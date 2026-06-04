"""
Extrator de texto de arquivos PDF usando pdfplumber.
Preserva estrutura de parágrafos e lida com PDFs de slides.
"""

from pathlib import Path

import pdfplumber


def extract_pdf_text(file_path: str | Path) -> str:
    """
    Extrai todo o texto de um PDF, página por página.

    Args:
        file_path: Caminho para o arquivo .pdf

    Returns:
        Texto extraído concatenado.
    """
    text_parts: list[str] = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if page_text:
                text_parts.append(page_text.strip())

    return "\n\n".join(text_parts)


def extract_all_pdfs_in_folder(folder: str | Path) -> dict[str, str]:
    """
    Extrai texto de todos os PDFs em uma pasta.

    Args:
        folder: Caminho da pasta

    Returns:
        Dicionário {nome_arquivo: texto_extraido}
    """
    folder = Path(folder)
    results: dict[str, str] = {}

    for pdf_file in sorted(folder.glob("*.pdf")):
        text = extract_pdf_text(pdf_file)
        if text.strip():
            results[pdf_file.name] = text

    return results
