import logging
import sys
import os
import asyncio
from typing import TypedDict, List, Tuple, Optional

import torch
from pydantic import BaseModel, Field

# LangChain / HuggingFace Imports
from langchain_community.document_loaders import RSSFeedLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.utils import filter_complex_metadata
from langgraph.graph import StateGraph, END

# Transformers / Sentence Transformers
from sentence_transformers import CrossEncoder
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

from config import settings

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RSS_Agent")


# --- Data Models ---
class ExtractionResult(BaseModel):
    is_relevant: bool = Field(
        description="Set to True ONLY if content matches the user profile specific needs."
    )
    relevant_facts: str = Field(
        description="A concise bulleted list of extracted facts. Leave empty if not relevant."
    )
    relevance_score: int = Field(
        description="Integer score 1-10. 10 is a perfect match."
    )


class AgentState(TypedDict):
    original_query: str
    current_query: str
    user_profile: str
    candidate_docs: List[Document]
    reranked_docs: List[Document]
    extracted_data: List[str]
    final_digest: str
    retry_count: int


# --- Service: Model Management ---
class ModelManager:
    """Singleton-like manager for heavy AI models to handle lazy loading."""

    def __init__(self):
        self._llm = None
        self._embeddings = None
        self._reranker = None
        self._tokenizer = None

    @property
    def embeddings(self):
        if not self._embeddings:
            logger.info(f"🧠 Loading Embeddings: {settings.EMBEDDING_MODEL}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def reranker(self):
        if not self._reranker:
            logger.info(f"⚖️ Loading Reranker: {settings.RERANKER_MODEL}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._reranker = CrossEncoder(settings.RERANKER_MODEL, device=device)
        return self._reranker

    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_ID)
        return self._tokenizer

    @property
    def llm(self):
        if not self._llm:
            logger.info(f"🧠 Loading LLM: {settings.LLM_MODEL_ID} (4-bit)...")

            # Ensure tokenizer is loaded
            _ = self.tokenizer

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=self._tokenizer,
                max_new_tokens=1024,
                temperature=0.1,
                return_full_text=False,
            )
            self._llm = HuggingFacePipeline(pipeline=pipe)
        return self._llm

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncates text to fit context window based on token count."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text

        return (
            self.tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True) + "..."
        )


# --- Service: Ingestion & Indexing ---
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
                # Add headers/timeout handling if needed in future
                loader = RSSFeedLoader(urls=settings.DOMAINS)
                return loader.load()
            except Exception as e:
                logger.error(f"Failed to load RSS feeds: {e}")
                return []

        return await asyncio.to_thread(blocking_load)

    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        logger.info("🔪 Chunking Documents (Semantic)...")
        try:
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
        # Chroma (Vector)
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.models.embeddings,
            persist_directory=settings.CHROMA_PATH,
        )
        chroma_retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.K_RETRIEVAL}
        )

        # BM25 (Keyword)
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = settings.K_RETRIEVAL

        return chroma_retriever, bm25_retriever


# --- Graph Nodes ---
class AgentNodes:
    def __init__(self, model_manager: ModelManager, vector_retriever, bm25_retriever):
        self.models = model_manager
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def retrieve(self, state: AgentState) -> AgentState:
        query = state["current_query"]
        logger.info(f"🔎 Retrieving: '{query}'")

        # Hybrid Search
        v_docs = self.vector_retriever.invoke(query)
        b_docs = self.bm25_retriever.invoke(query)

        # Deduplicate based on content hash or direct string match
        # Using a dictionary preserves order of insertion (Python 3.7+)
        unique_docs = {d.page_content: d for d in v_docs + b_docs}
        combined = list(unique_docs.values())

        logger.info(f"   -> Found {len(combined)} unique candidates")
        return {"candidate_docs": combined}

    def rerank(self, state: AgentState) -> AgentState:
        docs = state["candidate_docs"]
        if not docs:
            return {"reranked_docs": []}

        logger.info("⚖️ Reranking candidates...")
        pairs = [[state["current_query"], d.page_content] for d in docs]

        # Batch prediction
        scores = self.models.reranker.predict(pairs)

        # Zip, sort by score desc, extract top K
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_k = [d for d, s in scored_docs[: settings.K_FINAL]]

        if scored_docs:
            logger.info(f"   -> Top Match Score: {scored_docs[0][1]:.4f}")

        return {"reranked_docs": top_k}

    def extract(self, state: AgentState) -> AgentState:
        logger.info("🧪 Extracting Intelligence...")
        parser = PydanticOutputParser(pydantic_object=ExtractionResult)

        prompt = ChatPromptTemplate.from_template(
            settings.PROMPTS["extraction_system"]
        ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self.models.llm | parser
        valid_extracts = []

        for doc in state["reranked_docs"]:
            try:
                # Safety truncation
                safe_content = self.models.truncate_text(
                    doc.page_content, settings.MAX_INPUT_TOKENS
                )

                res = chain.invoke(
                    {
                        "user_profile": state["user_profile"],
                        "content": safe_content,
                    }
                )

                if (
                    res.is_relevant
                    and res.relevance_score >= settings.MIN_RELEVANCE_SCORE
                ):
                    source = doc.metadata.get("title", "Unknown Source")
                    snippet = (
                        f"**Source:** {source}\n"
                        f"**Relevance:** {res.relevance_score}/10\n"
                        f"**Facts:**\n{res.relevant_facts}"
                    )
                    valid_extracts.append(snippet)
                    logger.info(f"   -> Extracted info from '{source[:20]}...'")

            except Exception as e:
                # Log debug to avoid spamming console on parsing errors
                logger.debug(f"Extraction error on doc chunk: {e}")
                continue

        return {"extracted_data": valid_extracts}

    def rewrite_query(self, state: AgentState) -> AgentState:
        logger.info("🔄 Strategy: Expanding Search Query...")

        msg = settings.PROMPTS["rewrite_template"].format(
            original=state["original_query"], current=state["current_query"]
        )

        # Simple string invocation
        new_query = self.models.llm.invoke(msg).strip().replace('"', "")
        logger.info(f"   -> New Query: {new_query}")

        return {"current_query": new_query, "retry_count": state["retry_count"] + 1}

    def summarize(self, state: AgentState) -> AgentState:
        logger.info("📝 Generating Final Brief...")
        if not state["extracted_data"]:
            return {"final_digest": "No significant intelligence found after analysis."}

        context = "\n\n".join(state["extracted_data"])
        safe_context = self.models.truncate_text(context, 3000)

        chain = (
            ChatPromptTemplate.from_template(settings.PROMPTS["summarize_system"])
            | self.models.llm
            | StrOutputParser()
        )

        result = chain.invoke(
            {"context": safe_context, "user_profile": state["user_profile"]}
        )

        return {"final_digest": result}

    def decide_next_step(self, state: AgentState) -> str:
        """Conditional routing logic."""
        if state["extracted_data"]:
            return "summarize"
        if state["retry_count"] < settings.MAX_RETRIES:
            return "rewrite"
        return END


# --- Application Builder ---
def build_graph(chroma, bm25, model_manager):
    nodes = AgentNodes(model_manager, chroma, bm25)
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("rerank", nodes.rerank)
    workflow.add_node("extract", nodes.extract)
    workflow.add_node("rewrite", nodes.rewrite_query)
    workflow.add_node("summarize", nodes.summarize)

    # Set Entry Point
    workflow.set_entry_point("retrieve")

    # Add Standard Edges
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "extract")

    # Add Conditional Edges
    workflow.add_conditional_edges(
        "extract",
        nodes.decide_next_step,
        {"summarize": "summarize", "rewrite": "rewrite", END: END},
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("summarize", END)

    return workflow.compile()


async def main():
    print("Initializing AI Regulatory Digest Agent...")

    # Observability Setup
    os.environ["PHOENIX_PROJECT_NAME"] = "RSS_Digest_Agent"
    os.environ["PHOENIX_HOST"] = f"{settings.PHOENIX_HOST}:{settings.PHOENIX_PORT}"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
        f"{settings.PHOENIX_HOST}:{settings.PHOENIX_PORT}"
    )
    LangchainInstrumentor().instrument()
    print(
        f"Observability: Arize Phoenix (local @ {settings.PHOENIX_HOST}:{settings.PHOENIX_PORT})"
    )

    # Initialise OpenTelemetry → Phoenix
    trace.set_tracer_provider(TracerProvider())
    otlp_exporter = OTLPSpanExporter(endpoint=f"localhost:4317", insecure=True)
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    # Init Models and Ingestion
    models = ModelManager()
    ingestor = IngestionEngine(models)

    chroma, bm25 = await ingestor.run()

    if not chroma:
        sys.exit("Critical Error: Knowledge base construction failed.")

    # Build and Run Graph
    app = build_graph(chroma, bm25, models)

    initial_state = {
        "original_query": settings.QUERY,
        "current_query": settings.QUERY,
        "user_profile": settings.USER_PROFILE,
        "retry_count": 0,
        "extracted_data": [],
        "candidate_docs": [],
        "reranked_docs": [],
    }

    try:
        result = await app.ainvoke(initial_state)
        print("\n" + "=" * 60)
        print("📢 FINAL EXECUTIVE BRIEF")
        print("=" * 60)
        print(result.get("final_digest", "No data generated."))
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
