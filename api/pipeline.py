import base64
import io
import json

import anthropic
from fastapi import HTTPException
from pypdf import PdfReader
from pypdf.errors import PdfReadError

MODEL = "claude-opus-4-8"

IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

URGENCIES = ["basse", "normale", "haute"]


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def read_pdf_pages(content: bytes) -> list[dict]:
    """Lit un PDF, renvoie le texte page par page. Lève une HTTPException si invalide."""
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise HTTPException(
                status_code=422,
                detail="Le PDF est chiffré, impossible d'extraire le texte",
            )
        return [
            {"page": i + 1, "text": page.extract_text() or ""}
            for i, page in enumerate(reader.pages)
        ]
    except PdfReadError:
        raise HTTPException(status_code=422, detail="PDF invalide ou corrompu")


def digitize(content: bytes, content_type: str) -> str:
    """Extrait le texte d'un fichier reçu : pypdf si texte natif, sinon transcription Claude."""
    if content_type == "application/pdf":
        pages = read_pdf_pages(content)
        text = "\n\n".join(p["text"] for p in pages).strip()
        if len(text) >= 100:
            return text
        # PDF scanné (pas de texte natif) → transcription par vision
        block = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(content).decode(),
            },
        }
    elif content_type in IMAGE_TYPES:
        block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": content_type,
                "data": base64.standard_b64encode(content).decode(),
            },
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {content_type} (attendu : PDF, JPEG, PNG, GIF, WebP)",
        )

    response = _client().messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": [
                    block,
                    {
                        "type": "text",
                        "text": "Transcris intégralement et fidèlement le texte de ce document, "
                        "sans commentaire ni mise en forme ajoutée. "
                        "Si une partie est illisible, indique [illisible].",
                    },
                ],
            }
        ],
    )
    return next((b.text for b in response.content if b.type == "text"), "").strip()


def analyze(text: str, categories: list[str]) -> dict:
    """Classifie le courrier et en extrait les informations clés (sortie structurée garantie)."""
    schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": categories},
            "urgency": {"type": "string", "enum": URGENCIES},
            "dossier_ref": {
                "type": "string",
                "description": "Référence ou numéro de dossier mentionné dans le courrier, chaîne vide si absent",
            },
            "summary": {"type": "string", "description": "Résumé du courrier en 2-3 phrases"},
        },
        "required": ["category", "urgency", "dossier_ref", "summary"],
        "additionalProperties": False,
    }
    response = _client().messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system="Tu analyses des courriers administratifs reçus. "
        "Classe chaque courrier dans la catégorie la plus pertinente, évalue son urgence "
        "(haute si délai court ou conséquence grave), et repère la référence de dossier si présente.",
        messages=[{"role": "user", "content": f"Courrier reçu :\n\n{text}"}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    return json.loads(next(b.text for b in response.content if b.type == "text"))


def generate_draft(raw_text: str, category: str, dossier_pieces: list[dict], modeles: list[dict]) -> str:
    """Génère un brouillon de réponse à partir du courrier reçu et des passages RAG."""

    def fmt(passages: list[dict]) -> str:
        if not passages:
            return "(aucun)"
        return "\n\n---\n\n".join(
            f"[{p['source']}, extrait {p['chunk']}]\n{p['text']}" for p in passages
        )

    prompt = (
        f"Courrier reçu (catégorie : {category}) :\n\n{raw_text}\n\n"
        f"=== Pièces du dossier concerné ===\n{fmt(dossier_pieces)}\n\n"
        f"=== Modèles et réponses passées (pour le style et les formulations) ===\n{fmt(modeles)}\n\n"
        "Rédige un brouillon de réponse à ce courrier."
    )

    response = _client().messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system="Tu rédiges des brouillons de réponses à des courriers administratifs, en français formel. "
        "Appuie-toi sur les pièces du dossier fournies pour le fond, et sur les modèles de réponses "
        "passées pour le style et les formulations. "
        "N'invente aucun fait : si une information nécessaire manque, mets un espace réservé entre "
        "crochets, par exemple [DATE À COMPLÉTER]. "
        "Renvoie uniquement le texte du courrier de réponse, sans commentaire.",
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in response.content if b.type == "text"), "").strip()
