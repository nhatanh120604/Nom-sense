"""
Standalone script to ingest documents into Pinecone for the Nom-sense project.
This script is designed to be portable and does not depend on the 'app' module.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Third-party imports
try:
    from dotenv import load_dotenv
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_core.documents import Document
    from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pinecone import Pinecone
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError as e:
    print(f"Error: Missing required packages. Please install: {e}")
    sys.exit(1)

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger("ingest_pinecone_standalone")

# Constants
BOOK_TITLE_BY_FOLDER: Dict[str, str] = {
    "Book1": "Khái luận văn tự học Chữ Nôm",
    "Book2": "Ngôn ngữ. Văn tự. Ngữ văn (Tuyển tập)",
}
DEFAULT_AUTHOR = "Nguyen Quang Hong"


# --- Minimal Settings Class ---
class IngestSettings(BaseSettings):
    """Minimal configuration for ingestion."""
    pinecone_api_key: str = Field(..., alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(..., alias="PINECONE_INDEX_NAME")
    pinecone_embedding_model: str = Field("multilingual-e5-large", alias="PINECONE_EMBEDDING_MODEL")
    pinecone_namespace: str = Field("nom_sense", alias="PINECONE_NAMESPACE")

    # Defaults
    data_dir: Path = Field(Path("Word"), alias="DATA_DIR")
    chunk_size: int = Field(1600, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(300, alias="CHUNK_OVERLAP")

    model_config = SettingsConfigDict(
        env_file=(".env", "config/.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def resolved_data_dir(self) -> Path:
        # Resolve relative to the script execution or expected root.
        # Assumes the script is run from project root, so 'Word' is ./Word.
        # If passed as arg, handle it. Simple version:
        return self.data_dir.expanduser().resolve()


# --- Helper Functions (Copied from app/rag/pipeline.py) ---
def iter_document_paths(data_dir: Path, extensions: Iterable[str]) -> List[Path]:
    candidates: set[Path] = set()
    for ext in extensions:
        candidates.update(path.resolve() for path in data_dir.glob(f"**/*{ext}"))
    return sorted(candidates)


def normalise_chapter_label(label: Optional[str]) -> str:
    if not label:
        return ""
    cleaned = re.sub(r"_+", " ", label)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or ""


def derive_chapter_metadata(path: Path) -> Dict[str, Any]:
    stem = path.stem
    parts = stem.split(".", 1)
    chapter_index: Optional[int] = None
    chapter_title = stem.replace("_", " ").strip()

    if len(parts) == 2 and parts[0].strip().isdigit():
        chapter_index = int(parts[0].strip())
        chapter_title = parts[1].strip() or chapter_title
    elif parts[0].strip().isdigit():
        chapter_index = int(parts[0].strip())

    label = chapter_title or stem
    return {"chapter_index": chapter_index, "chapter": label}


def load_documents(data_dir: Path, extensions: Sequence[str]) -> List[Document]:
    documents: List[Document] = []

    LOGGER.info(f"Scanning for documents in: {data_dir}")
    paths = iter_document_paths(data_dir, extensions)
    LOGGER.info(f"Found {len(paths)} files.")

    for path in paths:
        chapter_meta = derive_chapter_metadata(path)
        chapter_label = normalise_chapter_label(chapter_meta.get("chapter"))

        book_key = path.parent.name
        book_title = BOOK_TITLE_BY_FOLDER.get(book_key, book_key.replace("_", " "))

        try:
            loader = PyMuPDFLoader(str(path))
            raw_docs = loader.load()
        except Exception as e:
            LOGGER.warning(f"Failed to load {path}: {e}")
            continue

        for doc in raw_docs:
            text = doc.page_content.strip()
            if not text:
                continue

            page = doc.metadata.get("page_number")
            if page is None:
                page = doc.metadata.get("page")

            if page is not None:
                page_number = int(page) + 1
            else:
                page_number = None

            doc_meta: Dict[str, Any] = {
                "book_title": book_title or "",
                "page_number": page_number if page_number is not None else "",
                "author": DEFAULT_AUTHOR,
                "chapter": chapter_label,
                "file_name": path.name,
                "source": str(path),
            }

            if page_number:
                doc_meta["citation_label"] = f"{book_title} – {chapter_label} - p.{page_number}"
            else:
                doc_meta["citation_label"] = f"{book_title} – {chapter_label}"

            doc.metadata.update(doc_meta)
            documents.append(doc)

    return documents


# --- Main Ingestion Logic ---
def main():
    # Load env vars
    load_dotenv()

    try:
        settings = IngestSettings()
    except Exception as e:
        LOGGER.error(f"Configuration Error: {e}")
        LOGGER.info("Make sure you have a .env file with PINECONE_API_KEY and PINECONE_INDEX_NAME")
        return

    LOGGER.info(f"Targeting Pinecone Index: {settings.pinecone_index_name}")
    LOGGER.info(f"Embedding Model: {settings.pinecone_embedding_model}")

    # 1. Load Documents
    try:
        documents = load_documents(settings.resolved_data_dir, (".pdf",))
        if not documents:
            LOGGER.error(f"No documents found in {settings.resolved_data_dir}")
            return
    except Exception as e:
        LOGGER.error(f"Error loading documents: {e}")
        return

    # 2. Split Documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    LOGGER.info(f"Prepared {len(chunks)} text chunks.")

    # 3. Setup Pinecone
    try:
        embeddings = PineconeEmbeddings(
            model=settings.pinecone_embedding_model,
            pinecone_api_key=settings.pinecone_api_key
        )

        vectorstore = PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=embeddings,
            pinecone_api_key=settings.pinecone_api_key
        )

        # Init Pinecone client just for deletion support
        pc = Pinecone(api_key=settings.pinecone_api_key)

    except Exception as e:
        LOGGER.error(f"Failed to initialize Pinecone: {e}")
        return

    # 4. Wipe Index (Force Rebuild)
    LOGGER.info(f"Deleting all vectors in namespace '{settings.pinecone_namespace}' before ingesting...")
    try:
        index = pc.Index(settings.pinecone_index_name)
        # Only delete strictly the namespace for this project
        index.delete(delete_all=True, namespace=settings.pinecone_namespace)
        LOGGER.info("Namespace cleared.")
    except Exception as e:
        LOGGER.warning(f"Failed to delete index contents (might be empty/missing): {e}")

    # 5. Batch Ingest
    LOGGER.info("Starting ingestion...")
    batch_size = 32

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        LOGGER.info(f"Ingesting batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} ({len(batch)} chunks)...")
        try:
            vectorstore.add_documents(batch, namespace=settings.pinecone_namespace)
        except Exception as e:
            LOGGER.error(f"Error ingesting batch {i}: {e}")
            time.sleep(10)
            try:
                 vectorstore.add_documents(batch)
            except Exception as retry_e:
                 LOGGER.error(f"Retry failed for batch {i}: {retry_e}")

        # Rate limit pause
        time.sleep(2)

    LOGGER.info("Ingestion successfully completed.")


if __name__ == "__main__":
    main()
