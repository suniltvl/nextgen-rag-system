"""Build or load Chroma indexes using local embedder on Drive."""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from colab.lib.data_loader import deduplicate_data

SEPARATORS = ["\n\n", "\n", " ", ".", ","]


def build_embeddings(embedder_path: Path) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=str(embedder_path),
        model_kwargs={"local_files_only": True},
    )


def build_retriever(
    rows: list[dict],
    *,
    embedder_path: Path,
    persist_directory: Path,
    chunk_size: int,
    chunk_overlap: int,
    search_type: str = "similarity",
    search_kwargs: dict | None = None,
    rebuild: bool = False,
):
    search_kwargs = search_kwargs or {"k": 3}
    if rebuild and persist_directory.exists():
        shutil.rmtree(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    dedup = deduplicate_data(rows)
    docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in dedup.items()
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    docs_chunks = splitter.split_documents(docs)
    embeddings = build_embeddings(embedder_path)

    if persist_directory.exists() and any(persist_directory.iterdir()) and not rebuild:
        print(f"Loading existing vector database at {persist_directory}...")
        vector_db = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings,
        )
    else:
        print(f"Creating vector database at {persist_directory} ({len(docs_chunks)} chunks)...")
        vector_db = Chroma.from_documents(
            documents=docs_chunks,
            embedding=embeddings,
            persist_directory=str(persist_directory),
        )

    return vector_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
