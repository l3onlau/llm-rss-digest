from dotenv import load_dotenv
import os
from typing import TypedDict, List
from operator import itemgetter
import argparse

import torch
from langchain_community.document_loaders import RSSFeedLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for flexibility."""
    parser = argparse.ArgumentParser(description="LLM-based RSS News Digest")
    parser.add_argument("--query", type=str, default="Top world news highlights", help="Query for news retrieval")
    parser.add_argument("--k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens for LLM generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM sampling temperature")
    return parser.parse_args()

def main() -> None:
    """Main execution flow: load config, process RSS, build RAG, run workflow."""
    args = parse_args()
    load_dotenv()

    # Config loading with defaults
    llm_model_name = os.getenv("LLM_MODEL_NAME", "microsoft/Phi-4-mini-instruct")
    user_profile = os.getenv("USER_PROFILE", "Busy professional seeking quick world news highlights.")
    embedding_model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

    # RSS loading
    urls = [u.strip() for u in os.getenv("DOMAINS", "").split(",") if u.strip()]
    if not urls:
        raise ValueError("No RSS domains provided in .env")
    loader = RSSFeedLoader(urls=urls)
    data = loader.load()

    # Document splitting and cleaning
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(data)
    cleaned_splits = [clean_doc(doc) for doc in splits]

    # Vector store setup
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = Chroma.from_documents(documents=cleaned_splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})

    # LLM setup with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=True,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Workflow definition
    workflow = build_workflow(llm)

    # Input and execution
    input_data = {"query": args.query, "user_profile": user_profile}
    initial_state = build_retrieval_chain(retriever).invoke(input_data)
    final_state = workflow.invoke(initial_state)

    print("\n=== FINAL SUMMARY ===")
    print(final_state["final_summary"].strip())
    print("====================")

def clean_doc(doc: Document) -> Document:
    """Clean document metadata to ensure serializability."""
    cleaned = {}
    for k, v in doc.metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            cleaned[k] = v
        elif isinstance(v, list):
            cleaned[k] = v[0] if len(v) == 1 else ", ".join(map(str, v)) if v else None
        else:
            cleaned[k] = str(v)
    return Document(page_content=doc.page_content, metadata=cleaned)

class SummaryState(TypedDict):
    """State for summary workflow."""
    user_profile: str
    query: str
    documents: List[Document]
    extracted_text: str
    final_summary: str

def extract_critical_updates(llm, state: SummaryState) -> dict:
    """Extract raw relevant text segments based on user profile."""
    context = "\n---\n".join(doc.page_content for doc in state["documents"])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Critical filter: Extract ONLY raw text segments highly relevant to USER PROFILE. No rephrasing/additions.\n\nUSER PROFILE: {user_profile}\n\nARTICLES:\n{context}",
            ),
            ("human", "Extract critical raw updates."),
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"context": context, "user_profile": state["user_profile"]})
    return {"extracted_text": result}

def generate_final_summary(llm, state: SummaryState) -> dict:
    """Synthesize extracted text into actionable bullet points."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Synthesize EXTRACTED TEXT into high-priority, actionable bullet points matching USER PROFILE.\n\nUSER PROFILE: {user_profile}\n\nEXTRACTED TEXT:\n{extracted_text}",
            ),
            ("human", "Generate final news summary."),
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"extracted_text": state["extracted_text"], "user_profile": state["user_profile"]})
    return {"final_summary": result}

def build_retrieval_chain(retriever):
    """Build retrieval chain for initial state."""
    return RunnablePassthrough.assign(
        documents=itemgetter("query") | retriever, user_profile=itemgetter("user_profile")
    ) | RunnablePassthrough.assign(
        query=itemgetter("query"),
        user_profile=itemgetter("user_profile"),
        documents=itemgetter("documents"),
        extracted_text=lambda _: "",
        final_summary=lambda _: "",
    )

def build_workflow(llm):
    """Construct LangGraph workflow."""
    workflow = StateGraph(SummaryState)
    workflow.add_node("extract", lambda state: extract_critical_updates(llm, state))
    workflow.add_node("summarize", lambda state: generate_final_summary(llm, state))
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "summarize")
    workflow.add_edge("summarize", END)
    return workflow.compile()

if __name__ == "__main__":
    main()