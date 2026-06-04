"""
Indexador FAISS para a base de conhecimento do TPMD Agent.

Responsável por:
- Carregar e fazer chunking dos arquivos .md
- Gerar embeddings via OpenAI ou Gemini
- Salvar/carregar o índice FAISS
- Suporte a checkpoint para retomar indexação interrompida (Gemini free tier)
"""

import pickle
from pathlib import Path
import time

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tqdm import tqdm


class RateLimitedGeminiEmbeddings(Embeddings):
    """
    Wrapper com rate limiting e checkpoint para embeddings do Gemini.

    Free tier: 1000 req/dia, 100 req/min.
    Salva progresso a cada lote — ao rodar novamente, retoma de onde parou.
    """

    def __init__(
        self,
        api_key: str,
        batch_size: int = 90,
        delay: float = 65.0,
        checkpoint_path: str | Path = "vectorstore/.embed_checkpoint.pkl",
    ):
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
        )
        self.batch_size = batch_size
        self.delay = delay
        self.checkpoint_path = Path(checkpoint_path)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        total = len(texts)
        num_batches = (total + self.batch_size - 1) // self.batch_size

        # Carrega checkpoint se existir
        start_batch = 0
        results: list[list[float]] = []
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "rb") as f:
                ckpt = pickle.load(f)
            if ckpt.get("total") == total:
                results = ckpt["results"]
                start_batch = ckpt["last_batch"] + 1
                print(f"  Checkpoint encontrado: retomando do lote {start_batch + 1}/{num_batches} "
                      f"({len(results)}/{total} chunks já processados).")
            else:
                print("  Checkpoint encontrado mas com contagem diferente — iniciando do zero.")

        if start_batch >= num_batches:
            print("  Todos os lotes já foram processados (checkpoint completo).")
            return results

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for idx in range(start_batch, num_batches):
            i = idx * self.batch_size
            batch = texts[i : i + self.batch_size]
            done = min(i + self.batch_size, total)
            print(f"  Lote {idx + 1}/{num_batches} — chunks {i + 1}-{done}/{total}...")
            results.extend(self._embeddings.embed_documents(batch))

            # Salva checkpoint após cada lote
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump({"total": total, "results": results, "last_batch": idx}, f)

            if idx + 1 < num_batches:
                print(f"  Aguardando {self.delay:.0f}s (rate limit free tier)...")
                time.sleep(self.delay)

        # Remove checkpoint ao concluir
        self.checkpoint_path.unlink(missing_ok=True)
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)


def build_index(
    md_files: list[Path],
    vectorstore_path: str | Path,
    embedding_model: str = "text-embedding-3-small",
    api_key: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    provider: str = "openai",
) -> FAISS:
    """
    Constrói o índice FAISS a partir dos arquivos .md.

    Args:
        md_files: Lista de arquivos .md para indexar
        vectorstore_path: Pasta onde o índice FAISS será salvo
        embedding_model: Modelo de embeddings
        api_key: Chave da API
        chunk_size: Tamanho máximo de cada chunk em caracteres
        chunk_overlap: Sobreposição entre chunks
        provider: "openai" ou "gemini"

    Returns:
        Instância do FAISS carregada.
    """
    if provider == "gemini":
        embeddings = RateLimitedGeminiEmbeddings(api_key=api_key)
    else:
        embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
    )

    all_docs: list[Document] = []

    for md_file in tqdm(md_files, desc="Carregando e dividindo docs"):
        text = md_file.read_text(encoding="utf-8")
        module = md_file.parent.name
        lesson = md_file.stem

        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[
                {
                    "source": str(md_file),
                    "module": module,
                    "lesson": lesson,
                }
            ],
        )
        all_docs.extend(chunks)

    print(f"Total de chunks gerados: {len(all_docs)}")
    print("Gerando embeddings e construindo índice FAISS...")

    texts = [doc.page_content for doc in all_docs]
    metadatas = [doc.metadata for doc in all_docs]

    vectors = embeddings.embed_documents(texts)

    print("Montando índice FAISS...")
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embeddings,
        metadatas=metadatas,
    )

    vectorstore_path = Path(vectorstore_path)
    vectorstore_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_path))

    print(f"Índice salvo com {len(all_docs)} chunks.")
    return vectorstore


def load_index(
    vectorstore_path: str | Path,
    embedding_model: str = "text-embedding-3-small",
    api_key: str | None = None,
    provider: str = "openai",
) -> FAISS:
    """
    Carrega um índice FAISS existente.

    Args:
        vectorstore_path: Pasta onde o índice FAISS está salvo
        embedding_model: Modelo de embeddings (deve ser o mesmo usado na criação)
        api_key: Chave da API
        provider: "openai" ou "gemini"

    Returns:
        Instância do FAISS carregada.
    """
    if provider == "gemini":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
        )
    else:
        embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore
