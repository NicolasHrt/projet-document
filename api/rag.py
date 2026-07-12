from pathlib import Path

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_or_create_collection("documents")

_model = None


def get_model() -> SentenceTransformer:
    # Chargé à la première utilisation : ~1 Go téléchargé au premier lancement
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Découpe en morceaux de ~chunk_size caractères, en coupant sur les paragraphes."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > chunk_size and current.strip():
            chunks.append(current.strip())
            current = current[-overlap:]
        current += p + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def index_document(
    filename: str, text: str, doc_type: str = "historique", dossier_ref: str = ""
) -> int:
    """Découpe, calcule les embeddings et stocke le document. Renvoie le nombre de chunks.

    doc_type : "modele" (réponse type), "historique" (courrier passé) ou "dossier" (pièce de dossier).
    """
    # Ré-uploader un fichier du même nom remplace l'ancienne version
    collection.delete(where={"source": filename})

    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = get_model().encode(chunks).tolist()
    collection.add(
        ids=[f"{filename}-{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {"source": filename, "chunk": i, "doc_type": doc_type, "dossier_ref": dossier_ref}
            for i in range(len(chunks))
        ],
    )
    return len(chunks)


def search(question: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
    """Renvoie les chunks les plus proches de la question, avec source et distance.

    where : filtre Chroma optionnel sur les metadata, ex. {"dossier_ref": "D-2026-42"}.
    """
    if collection.count() == 0:
        return []
    q_embedding = get_model().encode([question]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=min(n_results, collection.count()),
        where=where,
    )
    if not results["documents"][0]:
        return []
    return [
        {
            "text": doc,
            "source": meta["source"],
            "chunk": meta["chunk"],
            "distance": round(dist, 3),
        }
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def delete_document(filename: str) -> int:
    """Supprime tous les chunks d'un document indexé. Renvoie le nombre de chunks supprimés."""
    existing = collection.get(where={"source": filename})
    if not existing["ids"]:
        return 0
    collection.delete(ids=existing["ids"])
    return len(existing["ids"])


def list_documents() -> list[dict]:
    """Liste les documents indexés avec leur type, dossier et nombre de chunks."""
    data = collection.get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for meta in data["metadatas"]:
        doc = docs.setdefault(
            meta["source"],
            {
                "source": meta["source"],
                "doc_type": meta.get("doc_type", "historique"),
                "dossier_ref": meta.get("dossier_ref", ""),
                "chunks": 0,
            },
        )
        doc["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["source"])


def ask(question: str, n_results: int = 5) -> dict:
    """Recherche les passages pertinents puis génère une réponse ancrée avec Claude."""
    sources = search(question, n_results)
    if not sources:
        return {
            "answer": "Aucun document indexé. Uploade d'abord un PDF via POST /documents.",
            "sources": [],
        }

    passages = "\n\n---\n\n".join(
        f"[Source : {s['source']}, extrait {s['chunk']}]\n{s['text']}" for s in sources
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system="Réponds uniquement à partir des extraits fournis. "
        "Cite la source de chaque information. "
        "Si les extraits ne contiennent pas la réponse, dis-le clairement.",
        messages=[
            {
                "role": "user",
                "content": f"Extraits de documents :\n\n{passages}\n\nQuestion : {question}",
            }
        ],
    )
    answer = next((b.text for b in response.content if b.type == "text"), "")
    return {"answer": answer, "sources": sources}
