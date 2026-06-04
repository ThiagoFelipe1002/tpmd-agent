"""
Script 01 - Geração de documentação Markdown por tópico.

Para cada pasta de conteúdo, combina o conteúdo VTT e PDF
e usa GPT para gerar um documento Markdown rico e estruturado.

Os arquivos são salvos em data/docs/, organizados por módulo.

Uso:
    python scripts/01_generate_docs.py
    python scripts/01_generate_docs.py --resume   # Pula tópicos já processados
"""

import argparse
import sys
import re
from pathlib import Path

from dotenv import load_dotenv
import os
from openai import OpenAI
from tqdm import tqdm

# Garante que o projeto raiz está no sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.doc_generator import collect_lesson_folders, generate_lesson_doc


def sanitize_filename(name: str) -> str:
    """Remove caracteres inválidos para nomes de arquivo."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(". ")
    return name[:100]  # Limita o tamanho


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Gera documentação Markdown dos tópicos.")
    parser.add_argument("--resume", action="store_true", help="Pula tópicos já processados.")
    parser.add_argument(
        "--course-path",
        default=os.getenv("COURSE_PATH", ""),
        help="Caminho para a pasta raiz com os arquivos de conteúdo.",
    )
    parser.add_argument(
        "--output-path",
        default=os.getenv("DOCS_OUTPUT_PATH", "data/docs"),
        help="Pasta de saída para os markdowns.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DOC_GEN_MODEL", "gpt-4.1-mini"),
        help="Modelo OpenAI para geração.",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERRO: OPENAI_API_KEY não encontrada. Copie .env.example para .env e configure.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    output_base = Path(args.output_path)
    output_base.mkdir(parents=True, exist_ok=True)

    lesson_folders = collect_lesson_folders(args.course_path)
    print(f"Encontradas {len(lesson_folders)} pastas para processar.")

    skipped = 0
    processed = 0
    errors = 0

    for lesson_folder in tqdm(lesson_folders, desc="Gerando docs"):
        module_name = sanitize_filename(lesson_folder.parent.name)
        lesson_name = sanitize_filename(lesson_folder.name)

        # Cria subpasta por módulo
        module_output = output_base / module_name
        module_output.mkdir(parents=True, exist_ok=True)

        output_file = module_output / f"{lesson_name}.md"

        if args.resume and output_file.exists():
            skipped += 1
            continue

        try:
            content = generate_lesson_doc(
                lesson_folder=lesson_folder,
                client=client,
                model=args.model,
            )
            if content:
                output_file.write_text(content, encoding="utf-8")
                processed += 1
            else:
                tqdm.write(f"  [VAZIO] {lesson_folder.name} - nenhum conteúdo encontrado.")
        except Exception as e:
            errors += 1
            tqdm.write(f"  [ERRO] {lesson_folder.name}: {e}")

    print(f"\nConcluído: {processed} gerados | {skipped} pulados | {errors} erros")
    print(f"Docs salvos em: {output_base.resolve()}")


if __name__ == "__main__":
    main()
