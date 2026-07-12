from contextlib import contextmanager

import anthropic
from dotenv import load_dotenv

load_dotenv()  # charge .env (ANTHROPIC_API_KEY) avant toute utilisation

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api import db, pipeline, rag

app = FastAPI(title="Traitement de courrier administratif")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()


class Question(BaseModel):
    question: str
    n_results: int = 5


class ResponseUpdate(BaseModel):
    draft: str | None = None
    status: str | None = None
    category: str | None = None
    dossier_ref: str | None = None


class Categories(BaseModel):
    categories: list[str]


@contextmanager
def ai_errors():
    """Convertit les erreurs d'authentification Anthropic en message clair."""
    try:
        yield
    except (anthropic.AuthenticationError, TypeError):
        # Le SDK lève TypeError à la construction du client si aucune clé n'est trouvée
        raise HTTPException(
            status_code=500,
            detail="Clé API Anthropic manquante ou invalide : définis ANTHROPIC_API_KEY dans .env",
        )


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Extraction simple (historique) ----------


@app.post("/extract-text")
async def extract_text(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Le fichier doit être un PDF (reçu : {file.content_type})",
        )
    content = await file.read()
    pages = pipeline.read_pdf_pages(content)
    return {
        "filename": file.filename,
        "num_pages": len(pages),
        "text": "\n".join(p["text"] for p in pages),
        "pages": pages,
    }


# ---------- Base documentaire (RAG) ----------


@app.post("/documents")
async def add_document(
    file: UploadFile,
    doc_type: str = Form("historique"),
    dossier_ref: str = Form(""),
):
    """Extrait le texte d'un PDF et l'indexe dans la base vectorielle."""
    if doc_type not in ("modele", "historique", "dossier"):
        raise HTTPException(
            status_code=400,
            detail="doc_type doit être : modele, historique ou dossier",
        )
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Le fichier doit être un PDF (reçu : {file.content_type})",
        )
    content = await file.read()
    pages = pipeline.read_pdf_pages(content)
    text = "\n\n".join(p["text"] for p in pages)
    num_chunks = rag.index_document(file.filename, text, doc_type, dossier_ref)
    if num_chunks == 0:
        raise HTTPException(
            status_code=422,
            detail="Aucun texte extractible dans ce PDF (document scanné ?)",
        )
    return {
        "filename": file.filename,
        "doc_type": doc_type,
        "dossier_ref": dossier_ref,
        "chunks": num_chunks,
    }


@app.get("/documents")
def list_documents():
    """Liste les documents indexés."""
    return rag.list_documents()


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Supprime un document de la base vectorielle (tous ses extraits)."""
    deleted = rag.delete_document(filename)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document introuvable")
    return {"filename": filename, "deleted_chunks": deleted}


@app.post("/search")
def search(q: Question):
    """Recherche les passages les plus proches de la question (sans LLM, pour déboguer)."""
    return rag.search(q.question, q.n_results)


@app.post("/ask")
def ask(q: Question):
    """Répond à une question à partir des documents indexés (RAG complet)."""
    with ai_errors():
        return rag.ask(q.question, q.n_results)


# ---------- Boîte de réception ----------


@app.post("/inbox")
async def receive_response(
    file: UploadFile | None = None,
    text: str | None = Form(None),
):
    """Réceptionne un courrier (fichier image/PDF ou texte collé), le numérise et le classifie."""
    if file is None and not (text and text.strip()):
        raise HTTPException(status_code=400, detail="Fournis un fichier ou un texte")

    with ai_errors():
        if text and text.strip():
            raw_text, filename = text.strip(), None
        else:
            content = await file.read()
            raw_text = pipeline.digitize(content, file.content_type)
            filename = file.filename

        if not raw_text:
            raise HTTPException(status_code=422, detail="Aucun texte lisible dans ce document")

        analysis = pipeline.analyze(raw_text, db.get_categories())

    return db.insert_response(
        filename=filename,
        raw_text=raw_text,
        category=analysis["category"],
        urgency=analysis["urgency"],
        dossier_ref=analysis["dossier_ref"],
        summary=analysis["summary"],
    )


@app.get("/inbox")
def list_inbox(category: str | None = None, status: str | None = None):
    return db.list_responses(category, status)


@app.get("/inbox/{response_id}")
def get_inbox_item(response_id: int):
    row = db.get_response(response_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Courrier introuvable")
    return row


@app.post("/inbox/{response_id}/draft")
def generate_draft(response_id: int):
    """Génère un brouillon de réponse via RAG + IA."""
    row = db.get_response(response_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Courrier introuvable")

    query = row["summary"] or row["raw_text"][:500]
    dossier_pieces = (
        rag.search(query, 4, where={"dossier_ref": row["dossier_ref"]})
        if row["dossier_ref"]
        else []
    )
    modeles = rag.search(query, 4, where={"doc_type": {"$in": ["modele", "historique"]}})

    with ai_errors():
        draft = pipeline.generate_draft(
            row["raw_text"], row["category"], dossier_pieces, modeles
        )

    updated = db.update_response(response_id, draft=draft, status="brouillon")
    return {
        **updated,
        "rag_sources": {"dossier": dossier_pieces, "modeles": modeles},
    }


@app.put("/inbox/{response_id}")
def update_inbox_item(response_id: int, body: ResponseUpdate):
    """Édite le brouillon, le statut, la catégorie ou la référence dossier."""
    if db.get_response(response_id) is None:
        raise HTTPException(status_code=404, detail="Courrier introuvable")
    if body.status and body.status not in db.STATUSES:
        raise HTTPException(status_code=400, detail=f"Statut invalide (attendu : {db.STATUSES})")
    return db.update_response(response_id, **body.model_dump())


# ---------- Catégories & stats ----------


@app.get("/categories")
def get_categories():
    return db.get_categories()


@app.put("/categories")
def set_categories(body: Categories):
    if not body.categories:
        raise HTTPException(status_code=400, detail="La liste ne peut pas être vide")
    db.set_categories(body.categories)
    return db.get_categories()


@app.get("/stats")
def get_stats():
    return db.stats()
