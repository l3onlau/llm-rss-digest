import logging
from functools import partial
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

import src.models as models
from config import settings

logger = logging.getLogger("RSS_Agent")


# --- Data Structures ---
class ExtractionResult(BaseModel):
    is_relevant: bool = Field(
        description="True if content contains facts relevant to the Query."
    )
    relevant_facts: str = Field(
        description="Concise bulleted list of facts. Empty if irrelevant."
    )
    relevance_score: int = Field(description="Score 1-10. 10 is perfect match.")


class AgentState(TypedDict):
    original_query: str
    current_query: str
    candidate_docs: List[Document]
    reranked_docs: List[Document]
    extracted_data: List[str]
    final_digest: str
    retry_count: int


# --- Nodes ---
def retrieve_node(state: AgentState, vector_retriever, bm25_retriever) -> AgentState:
    query = state["current_query"]
    logger.info(f"🔎 Retrieving: '{query}'")

    v_docs = vector_retriever.invoke(query)
    b_docs = bm25_retriever.invoke(query)

    # Deduplicate based on content
    unique_docs = {d.page_content: d for d in v_docs + b_docs}
    combined = list(unique_docs.values())

    logger.info(f"   -> Found {len(combined)} unique candidates")
    return {"candidate_docs": combined}


def rerank_node(state: AgentState) -> AgentState:
    docs = state["candidate_docs"]
    if not docs:
        return {"reranked_docs": []}

    logger.info("⚖️ Reranking candidates...")
    reranker = models.get_reranker()
    pairs = [[state["current_query"], d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    # Sort by score descending
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_k = [d for d, s in scored_docs[: settings.K_FINAL]]

    return {"reranked_docs": top_k}


def extract_node(state: AgentState) -> AgentState:
    logger.info("⛏️ Extracting Intelligence...")
    parser = PydanticOutputParser(pydantic_object=ExtractionResult)
    llm = models.get_llm()

    prompt = ChatPromptTemplate.from_template(
        settings.prompts.EXTRACTION_SYSTEM
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    valid_extracts = []

    for doc in state["reranked_docs"]:
        try:
            safe_content = models.truncate_text(
                doc.page_content, settings.MAX_INPUT_TOKENS
            )
            res = chain.invoke(
                {"query": state["original_query"], "content": safe_content}
            )

            if res.is_relevant and res.relevance_score >= settings.MIN_RELEVANCE_SCORE:
                source = doc.metadata.get("title", "Unknown Source")
                snippet = (
                    f"**Source:** {source}\n"
                    f"**Relevance:** {res.relevance_score}/10\n"
                    f"**Facts:**\n{res.relevant_facts}"
                )
                valid_extracts.append(snippet)
                logger.info(f"   -> Extracted info from '{source[:20]}...'")

        except Exception as e:
            logger.debug(f"Extraction error on doc: {e}")
            continue

    return {"extracted_data": valid_extracts}


def rewrite_query_node(state: AgentState) -> AgentState:
    logger.info("🔄 Strategy: Expanding Search Query...")
    msg = settings.prompts.REWRITE_TEMPLATE.format(
        original=state["original_query"], current=state["current_query"]
    )
    llm = models.get_llm()
    new_query = llm.invoke(msg).strip().replace('"', "")
    logger.info(f"   -> New Query: {new_query}")
    return {"current_query": new_query, "retry_count": state["retry_count"] + 1}


def summarize_node(state: AgentState) -> AgentState:
    logger.info("📝 Generating Final Brief...")
    if not state["extracted_data"]:
        return {"final_digest": "No significant intelligence found after analysis."}

    context = "\n\n".join(state["extracted_data"])
    safe_context = models.truncate_text(context, 3000)
    llm = models.get_llm()

    chain = (
        ChatPromptTemplate.from_template(settings.prompts.SUMMARIZE_SYSTEM)
        | llm
        | StrOutputParser()
    )
    result = chain.invoke({"context": safe_context, "query": state["original_query"]})
    return {"final_digest": result}


def decide_next_step(state: AgentState) -> str:
    if state["extracted_data"]:
        return "summarize"
    if state["retry_count"] < settings.MAX_RETRIES:
        return "rewrite"
    return END


# --- Graph Builder ---
def build_graph(chroma_retriever, bm25_retriever):
    workflow = StateGraph(AgentState)

    retrieve_with_deps = partial(
        retrieve_node, vector_retriever=chroma_retriever, bm25_retriever=bm25_retriever
    )

    workflow.add_node("retrieve", retrieve_with_deps)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("summarize", summarize_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "extract")

    workflow.add_conditional_edges(
        "extract",
        decide_next_step,
        {"summarize": "summarize", "rewrite": "rewrite", END: END},
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("summarize", END)

    return workflow.compile()
