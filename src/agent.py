import logging
from typing import TypedDict, List
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END

from config import settings
from src.models import ModelManager

logger = logging.getLogger("RSS_Agent")


class ExtractionResult(BaseModel):
    is_relevant: bool = Field(description="True ONLY if content matches user profile.")
    relevant_facts: str = Field(
        description="Concise bulleted list of facts. Empty if irrelevant."
    )
    relevance_score: int = Field(description="Score 1-10. 10 is perfect match.")


class AgentState(TypedDict):
    original_query: str
    current_query: str
    user_profile: str
    candidate_docs: List[Document]
    reranked_docs: List[Document]
    extracted_data: List[str]
    final_digest: str
    retry_count: int


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

        # Deduplicate
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
        scores = self.models.reranker.predict(pairs)

        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_k = [d for d, s in scored_docs[: settings.K_FINAL]]

        return {"reranked_docs": top_k}

    def extract(self, state: AgentState) -> AgentState:
        logger.info("🧪 Extracting Intelligence...")
        parser = PydanticOutputParser(pydantic_object=ExtractionResult)

        prompt = ChatPromptTemplate.from_template(
            settings.prompts.EXTRACTION_SYSTEM
        ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self.models.llm | parser
        valid_extracts = []

        for doc in state["reranked_docs"]:
            try:
                safe_content = self.models.truncate_text(
                    doc.page_content, settings.MAX_INPUT_TOKENS
                )
                res = chain.invoke(
                    {"user_profile": state["user_profile"], "content": safe_content}
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
                logger.debug(f"Extraction error: {e}")
                continue

        return {"extracted_data": valid_extracts}

    def rewrite_query(self, state: AgentState) -> AgentState:
        logger.info("🔄 Strategy: Expanding Search Query...")
        msg = settings.prompts.REWRITE_TEMPLATE.format(
            original=state["original_query"], current=state["current_query"]
        )
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
            ChatPromptTemplate.from_template(settings.prompts.SUMMARIZE_SYSTEM)
            | self.models.llm
            | StrOutputParser()
        )
        result = chain.invoke(
            {"context": safe_context, "user_profile": state["user_profile"]}
        )
        return {"final_digest": result}

    def decide_next_step(self, state: AgentState) -> str:
        if state["extracted_data"]:
            return "summarize"
        if state["retry_count"] < settings.MAX_RETRIES:
            return "rewrite"
        return END


def build_graph(chroma_retriever, bm25_retriever, model_manager):
    nodes = AgentNodes(model_manager, chroma_retriever, bm25_retriever)
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("rerank", nodes.rerank)
    workflow.add_node("extract", nodes.extract)
    workflow.add_node("rewrite", nodes.rewrite_query)
    workflow.add_node("summarize", nodes.summarize)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "extract")

    workflow.add_conditional_edges(
        "extract",
        nodes.decide_next_step,
        {"summarize": "summarize", "rewrite": "rewrite", END: END},
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("summarize", END)

    return workflow.compile()
