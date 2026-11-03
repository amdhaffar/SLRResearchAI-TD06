from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


def extract_pdf_pages(source: str | Path | BinaryIO) -> list[str]:
    """Code fourni : extrait le texte page par page et signale les pages vides."""
    reader = PdfReader(source)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    if not pages:
        raise ValueError("Le PDF ne contient aucune page")
    if not any(pages):
        raise ValueError("Aucun texte extractible : un OCR est probablement nécessaire")
    return pages

