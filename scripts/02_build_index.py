"""
Script 02 - Construção do índice FAISS.

Lê todos os arquivos .md gerados em data/docs/,
faz chunking inteligente e cria o índice vetorial FAISS.

Uso:
    python scripts/02_build_index.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
import os
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.indexer import build_index


def main():
    load_dotenv()

    provider = os.getenv("AI_PROVIDER", "openai").lower()
    docs_path = os.getenv("DOCS_OUTPUT_PATH", "data/docs")
    vectorstore_path = os.getenv("VECTORSTORE_PATH", "vectorstore")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("ERRO: GEMINI_API_KEY não encontrada. Configure o arquivo .env.")
            sys.exit(1)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERRO: OPENAI_API_KEY não encontrada. Configure o arquivo .env.")
            sys.exit(1)

    docs_dir = Path(docs_path)
    md_files = list(docs_dir.rglob("*.md"))

    if not md_files:
        print(f"Nenhum arquivo .md encontrado em '{docs_dir.resolve()}'.")
        print("Execute primeiro: python scripts/01_generate_docs.py")
        sys.exit(1)

    print(f"Encontrados {len(md_files)} documentos Markdown para indexar.")

    build_index(
        md_files=md_files,
        vectorstore_path=vectorstore_path,
        embedding_model=embedding_model,
        api_key=api_key,
        provider=provider,
    )

    print(f"\nÍndice FAISS salvo em: {Path(vectorstore_path).resolve()}")


if __name__ == "__main__":
    main()
