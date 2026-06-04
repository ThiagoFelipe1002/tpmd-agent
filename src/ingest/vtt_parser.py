"""
Parser de arquivos .vtt (WebVTT - legendas de vídeo).
Remove timestamps, IDs de cue e cabeçalho, retornando apenas o texto limpo.
"""

import re
from pathlib import Path


def parse_vtt(file_path: str | Path) -> str:
    """
    Lê um arquivo .vtt e retorna o texto limpo sem timestamps ou metadados.

    Args:
        file_path: Caminho para o arquivo .vtt

    Returns:
        Texto limpo concatenado de todas as legendas.
    """
    content = Path(file_path).read_text(encoding="utf-8")
    lines = content.splitlines()

    text_lines: list[str] = []
    skip_next = False

    for line in lines:
        line = line.strip()

        # Pular cabeçalho WEBVTT e metadados
        if line.startswith("WEBVTT") or line.startswith("X-TIMESTAMP-MAP"):
            continue

        # Pular timestamps (ex: 00:00:05.930 --> 00:00:06.920)
        if re.match(r"^\d{2}:\d{2}:\d{2}[\.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[\.,]\d{3}", line):
            skip_next = False
            continue

        # Pular IDs de cue (linhas com apenas números)
        if re.match(r"^\d+$", line):
            continue

        # Pular linhas vazias
        if not line:
            continue

        text_lines.append(line)

    # Juntar linhas, removendo duplicatas consecutivas (artefato comum em VTTs)
    deduplicated: list[str] = []
    for line in text_lines:
        if not deduplicated or deduplicated[-1] != line:
            deduplicated.append(line)

    return " ".join(deduplicated)
