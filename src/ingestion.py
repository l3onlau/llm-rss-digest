import asyncio
import logging
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_community.document_loaders import RSSFeedLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from config import settings
from src.models import get_embeddings

logger = logging.getLogger("RSS_Agent")


async def run_ingestion() -> Tuple[Optional[any], Optional[any]]:
    """Orchestrates loading, chunking, and indexing."""
    logger.info(f"📡 Polling {len(settings.DOMAINS)} RSS feeds...")

    raw_docs = await _fetch_feeds()
    if not raw_docs:
        logger.warning("❌ No documents found in feeds.")
        return None, None

    split_docs = _chunk_documents(raw_docs)
    return _create_retrievers(split_docs)


async def _fetch_feeds() -> List[Document]:
    """Async wrapper for blocking RSS loader."""

    def blocking_load():
        try:
            # RSSFeedLoader handles multiple URLs natively
            loader = RSSFeedLoader(urls=settings.DOMAINS)
            return loader.load()
        except Exception as e:
            logger.error(f"Failed to load RSS feeds: {e}")
            return []

    return await asyncio.to_thread(blocking_load)


def _chunk_documents(docs: List[Document]) -> List[Document]:
    """Splits documents based on semantic similarity breakpoints."""
    logger.info("🔪 Chunking Documents (Semantic)...")
    try:
        text_splitter = SemanticChunker(
            get_embeddings(), breakpoint_threshold_type="percentile"
        )
        split_docs = text_splitter.split_documents(docs)
    except Exception as e:
        logger.warning(f"Semantic chunking failed ({e}), falling back to raw docs.")
        split_docs = docs

    # Clean metadata for ChromaDB compatibility
    split_docs = filter_complex_metadata(split_docs)
    logger.info(f"   -> Generated {len(split_docs)} chunks.")
    return split_docs


def _create_retrievers(docs: List[Document]):
    """Builds Hybrid Search Retrievers (Dense + Sparse)."""
    logger.info("💾 Building/Updating Vector Store...")

    # Dense Retriever (Vector)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=settings.CHROMA_PATH,
    )
    chroma_retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.K_RETRIEVAL}
    )

    # Sparse Retriever (Keyword/BM25)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = settings.K_RETRIEVAL

    return chroma_retriever, bm25_retriever
