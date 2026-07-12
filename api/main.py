import io

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from pypdf.errors import PdfReadError

app = FastAPI(title="API d'extraction de texte PDF")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-text")
async def extract_text(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Le fichier doit être un PDF (reçu : {file.content_type})",
        )

    content = await file.read()

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise HTTPException(
                status_code=422,
                detail="Le PDF est chiffré, impossible d'extraire le texte",
            )
        pages = [
            {"page": i + 1, "text": page.extract_text() or ""}
            for i, page in enumerate(reader.pages)
        ]
    except PdfReadError:
        raise HTTPException(status_code=422, detail="PDF invalide ou corrompu")

    return {
        "filename": file.filename,
        "num_pages": len(pages),
        "text": "\n".join(p["text"] for p in pages),
        "pages": pages,
    }
