import asyncio
import logging
from typing import List, Tuple, Optional
from langchain_community.document_loaders import RSSFeedLoader
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

from config import settings
from src.models import ModelManager

logger = logging.getLogger("RSS_Agent")


class IngestionEngine:
    def __init__(self, model_manager: ModelManager):
        self.models = model_manager

    async def run(self) -> Tuple[Optional[any], Optional[any]]:
        """Orchestrates loading, chunking, and indexing."""
        logger.info(f"📡 Polling {len(settings.DOMAINS)} RSS feeds...")

        raw_docs = await self._fetch_feeds()
        if not raw_docs:
            logger.warning("❌ No documents found in feeds.")
            return None, None

        split_docs = self._chunk_documents(raw_docs)
        return self._create_retrievers(split_docs)

    async def _fetch_feeds(self) -> List[Document]:
        def blocking_load():
            try:
                loader = RSSFeedLoader(urls=settings.DOMAINS)
                return loader.load()
            except Exception as e:
                logger.error(f"Failed to load RSS feeds: {e}")
                return []

        return await asyncio.to_thread(blocking_load)

    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        logger.info("🔪 Chunking Documents (Semantic)...")
        try:
            # Note: SemanticChunker is computation heavy
            text_splitter = SemanticChunker(
                self.models.embeddings, breakpoint_threshold_type="percentile"
            )
            split_docs = text_splitter.split_documents(docs)
        except Exception as e:
            logger.warning(f"Semantic chunking failed ({e}), falling back to raw docs.")
            split_docs = docs

        split_docs = filter_complex_metadata(split_docs)
        logger.info(f"   -> Generated {len(split_docs)} chunks.")
        return split_docs

    def _create_retrievers(self, docs: List[Document]):
        logger.info("💾 Building/Updating Vector Store...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.models.embeddings,
            persist_directory=settings.CHROMA_PATH,
        )
        chroma_retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.K_RETRIEVAL}
        )

        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = settings.K_RETRIEVAL

        return chroma_retriever, bm25_retriever
