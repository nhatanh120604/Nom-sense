"""
RAG Pipeline implementation using LangChain and Pinecone.
Handles document loading, embedding, retrieval, and LLM generation.
"""
from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Any
from urllib.parse import quote, quote_plus

from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

from ..schemas import SourceChunk
from ..settings import Settings

LOGGER = logging.getLogger(__name__)


BOOK_TITLE_BY_FOLDER: Dict[str, str] = {
    "Book1": "Khái luận văn tự học Chữ Nôm",
    "Book2": "Ngôn ngữ. Văn tự. Ngữ văn (Tuyển tập)",
}

# Hardcoded author for current books as requested
DEFAULT_AUTHOR = "Nguyen Quang Hong"


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
    for path in iter_document_paths(data_dir, extensions):
        chapter_meta = derive_chapter_metadata(path)
        chapter_label = normalise_chapter_label(chapter_meta.get("chapter"))

        book_key = path.parent.name
        book_title = BOOK_TITLE_BY_FOLDER.get(book_key, book_key.replace("_", " "))

        loader = PyMuPDFLoader(str(path))
        for doc in loader.load():
            text = doc.page_content.strip()
            if not text:
                continue

            # Strict Metadata Schema: [book-title, page-number, author, chapter]
            # Default empty strings for missing values

            page = doc.metadata.get("page_number")
            if page is None:
                page = doc.metadata.get("page")

            # Normalize Page Number
            if page is not None:
                page_number = int(page) + 1
            else:
                page_number = None

            doc_meta: Dict[str, Any] = {
                "book_title": book_title or "",
                "page_number": page_number if page_number is not None else "", # Pinecone metadata prefers strings or numbers, but consistent types are good. Let's use int or empty string? Pinecone metadata values must be string, number, boolean, or list of strings. Empty string is safe.
                "author": DEFAULT_AUTHOR,
                "chapter": chapter_label,
                "file_name": path.name,
                "source": str(path),
            }

            # Cleanup: ensure 'page_number' is consistent (e.g. keep as int, handle None separately if needed, or just use what we have if it's valid type)
            # Pinecone allows int.

            # Create citation label for LLM context
            if page_number:
                doc_meta["citation_label"] = f"{book_title} – {chapter_label} - p.{page_number}"
            else:
                doc_meta["citation_label"] = f"{book_title} – {chapter_label}"

            doc.metadata.update(doc_meta)

            # Remove purely internal or large keys if not needed in metadata filters
            # doc.metadata.pop("total_pages", None)

            documents.append(doc)

    if not documents:
        raise ValueError(f"No supported documents were loaded from {data_dir}.")

    return documents


def unique_citations(docs: Sequence[Document]) -> List[str]:
    citations: List[str] = []
    for doc in docs:
        label = doc.metadata.get("citation_label") or doc.metadata.get("source") or "Unknown"
        if label not in citations:
            citations.append(str(label))
    return citations


def format_docs(docs: Sequence[Document]) -> str:
    formatted: List[str] = []
    for doc in docs:
        label = doc.metadata.get("citation_label") or "Unknown Source"
        formatted.append(f"Source: {label}\n{doc.page_content}")
    return "\n\n".join(formatted) if formatted else "No supporting context retrieved."


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_env()

        LOGGER.info("Initializing Pinecone RAG Service...")

        # Pinecone Embeddings (Serverless Inference)
        self.embeddings = PineconeEmbeddings(
            model=settings.pinecone_embedding_model,
            pinecone_api_key=settings.pinecone_api_key
        )

        # Pinecone Client for Reranking and ID generation
        self.pc = Pinecone(api_key=settings.pinecone_api_key)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        self.llm = ChatOpenAI(model=settings.chat_model, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
   Bạn là trợ lý hỏi–đáp chỉ sử dụng NGỮ CẢNH (các đoạn trích đã truy xuất).
YÊU CẦU:
Trả lời theo cấu trúc dưới như sau:

Kết luận: kết luận tóm tắt ở đây
(Xuống dòng ở đây)
Giải thích: giải thích và minh chứng chữ Nôm ở đây

Các điểm cần nhớ:
NẾU CHỮ NÔM CÓ TRONG NGỮ CẢNH VÀ LIÊN QUAN TỚI CÂU HỎI: PHẢI COPY VÀO CÂU TRẢ LỜI KÈM THEO GIẢI THÍCH.
Nếu trong NGỮ CẢNH có chữ Nôm, bạn phải luôn ưu tiên sử dụng chữ Nôm và tiếng Việt phối hợp với nhau trong câu trả lời, ví dụ  trả lời tiếng Việt + chữ Nôm làm  ví dụ, lấy các chữ Nôm từ ngữ cảnh để trả lời. Đặc biệt nếu trong NGỮ CẢNH có chữ Nôm liên quan đến câu hỏi, bạn cần trả lời chi tiết hơn, có thể liệt kê các ví dụ cụ thể từ ngữ cảnh (chữ Nôm).
Nếu cùng đoạn chứa cả chữ Quốc Ngữ và chữ Nôm, trình bày cả hai dạng (Quốc Ngữ trước, chữ Nôm trong ngoặc
2) Nếu NGỮ CẢNH thiếu/không liên quan, trả lời: “Không tìm thấy thông tin này trong sách/nguồn đã cho.”
3) Không suy diễn, không đưa kiến thức ngoài NGỮ CẢNH.
4) Luôn giữ nguyên thuật ngữ gốc khi cần.

Đây là  ví dụ 1 câu hỏi và câu trả lời mẫu:
Câu hỏi : Chứng tích sớm nhất về chữ Nôm là gì?
Trả lời:
Kết luận: Là các bia đá thời Lý
Giải thích:
Bia mộ bà họ Lê (1174) có chữ Nôm mượn Hán và chữ Nôm tự tạo.
Bia chùa Tháp Miếu (1210) có hơn hai chục chữ Nôm, như:
𥺹 oản (bộ mễ + uyển),
䊷 chài (bộ mịch + tài),
土而 nhe (bộ thổ + nhi),
񢂞 bơi (bộ thuỷ + bi).
""".strip(),
                ),
                ("user", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

        self._vectorstore: Optional[PineconeVectorStore] = None

    def load_source_documents(self) -> List[Document]:
        return load_documents(self.settings.resolved_data_dir, (".pdf",))

    def ensure_vectorstore(self) -> PineconeVectorStore:
        if self._vectorstore is not None:
            return self._vectorstore

        # Assume the index exists (User responsibility to create index).
        # Global Search: Use the default namespace (None) or a specific one if configured.
        # Use declared namespace for project isolation.

        self._vectorstore = PineconeVectorStore(
            index_name=self.settings.pinecone_index_name,
            embedding=self.embeddings,
            pinecone_api_key=self.settings.pinecone_api_key,
            namespace=self.settings.pinecone_namespace
        )
        return self._vectorstore

    def ingest(self, force_rebuild: bool = False) -> None:
        """
        Ingest documents into Pinecone.
        Note: force_rebuild=True in Pinecone context typically implies "delete all and re-upload".
        Current implementation simply adds documents. Deletion is resource-intensive on a remote index.
        """
        vectorstore = self.ensure_vectorstore()
        documents = self.load_source_documents()
        chunks = self.splitter.split_documents(documents)

        if force_rebuild:
            LOGGER.info("Delete all vectors in index before ingesting...")
            # Deleting everything in the namespace (assuming default namespace)
            index = self.pc.Index(self.settings.pinecone_index_name)
            try:
                index.delete(delete_all=True)
            except Exception as e:
                LOGGER.warning("Failed to delete index contents: %s", e)

        LOGGER.info("Adding %s chunks to Pinecone...", len(chunks))

        # Batching to avoid Rate Limits (250k tokens/min).
        # Assumption: ~500 tokens per chunk average.
        # Batch size 32 = ~16k tokens.
        # Sleeping 2 seconds between batches.
        batch_size = 32
        import time

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            LOGGER.info(f"Ingesting batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} ({len(batch)} chunks)...")
            try:
                vectorstore.add_documents(batch)
            except Exception as e:
                LOGGER.error(f"Error ingesting batch {i}: {e}")
                # Wait longer if we hit an error (likely rate limit)
                time.sleep(10)
                # Retry once? For now just log and continue/sleep
                try:
                     vectorstore.add_documents(batch)
                except Exception as retry_e:
                     LOGGER.error(f"Retry failed for batch {i}: {retry_e}")

            time.sleep(2)

        LOGGER.info("Ingestion complete.")

    def _build_source_payload(self, doc: Document) -> SourceChunk:
        # Use flattened metadata structure
        meta = doc.metadata
        return SourceChunk(
            label=meta.get("citation_label", ""),
            page_number=meta.get("page_number", ""),
            chapter=meta.get("chapter", ""),
            book_title=meta.get("book_title", ""),
            file_name=meta.get("file_name", ""),
            source_path=meta.get("source", ""),
            text=doc.page_content,
            viewer_url=self._build_viewer_url(
                file_name=meta.get("file_name"),
                page_number=meta.get("page_number"),
                snippet=doc.page_content
            ),
        )

    def _build_viewer_url(
        self, *, file_name: Optional[str], page_number: Any, snippet: str
    ) -> Optional[str]:
        if not self.settings.serve_docs or not file_name:
            return None
        base = self.settings.docs_mount_path.rstrip("/") or "/docs"
        url = f"{base}/{quote(file_name)}"
        fragments: List[str] = []

        # page_number in metadata might be int or str, need to handle both
        if page_number is not None and str(page_number) != "":
            fragments.append(f"page={page_number}")

        if snippet:
            search_term = quote_plus(snippet[:120])
            fragments.append(f"search={search_term}")
        if fragments:
            return f"{url}#{'&'.join(fragments)}"
        return url

    def ask(
        self,
        *,
        question: str,
        additional_context: Optional[str] = None,
        top_k: Optional[int] = None,
        pool_size: Optional[int] = None,
        temperature: Optional[float] = None,
        rerank: bool = True,
    ) -> Dict[str, object]:
        vectorstore = self.ensure_vectorstore()

        # 1. Retrieval (High Recall)
        k = pool_size or self.settings.retriever_k
        # Plain similarity search (no collection/namespace filtering implies Global Search)
        candidate_docs = vectorstore.similarity_search(question, k=k)

        # 2. Reranking (High Precision)
        chosen_top_k = top_k or self.settings.rerank_top_k
        final_docs = candidate_docs

        if rerank and candidate_docs:
            docs_content = [d.page_content for d in candidate_docs]
            try:
                # Pinecone Inference Rerank
                result = self.pc.inference.rerank(
                    model=self.settings.pinecone_rerank_model,
                    query=question,
                    documents=docs_content,
                    top_n=chosen_top_k,
                    return_documents=False # We map back by index
                )

                # Map back results to document objects
                reranked_docs = []
                for item in result.data:
                    idx = item.index
                    doc = candidate_docs[idx]
                    # Optionally attach score: doc.metadata["rerank_score"] = item.score
                    reranked_docs.append(doc)
                final_docs = reranked_docs
            except Exception as e:
                LOGGER.error("Pinecone Reranking failed: %s. Falling back to raw similarity.", e)
                final_docs = candidate_docs[:chosen_top_k]
        else:
            final_docs = candidate_docs[:chosen_top_k]

        # 3. Generation
        context_text = format_docs(final_docs)
        if additional_context:
            extra = additional_context.strip()
            context_text = f"{extra}\n\n{context_text}" if context_text else extra

        previous_temperature = self.llm.temperature
        if temperature is not None:
            self.llm.temperature = temperature
        try:
            response = self.llm.invoke(
                self.prompt.format_messages(context=context_text, question=question)
            )
        finally:
            if temperature is not None:
                self.llm.temperature = previous_temperature

        answer = response.content.strip()
        citations = unique_citations(final_docs)
        sources = [self._build_source_payload(doc) for doc in final_docs]
        return {
            "answer": answer,
            "citations": citations,
            "sources": sources,
        }
