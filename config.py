from typing import List, Optional, Dict
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Data Sources ---
    DOMAINS: List[str] = ["https://feeds.bbci.co.uk/news/technology/rss.xml"]

    # --- Query & Profile ---
    QUERY: str = "AI regulation"
    USER_PROFILE: str = "Tech Analyst"

    # --- Models ---
    LLM_MODEL_ID: str = "microsoft/Phi-4-mini-instruct"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # --- RAG Parameters ---
    CHROMA_PATH: str = "./chroma_db"
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200
    K_RETRIEVAL: int = 15
    K_FINAL: int = 5  # Increased slightly for better context
    MAX_RETRIES: int = 2
    MIN_RELEVANCE_SCORE: int = 4  # Threshold for keeping a fact

    # --- Resource Management ---
    MAX_INPUT_TOKENS: int = 1500  # Max tokens to feed extraction node

    # --- Observability ---
    PHOENIX_PORT: int = 6006
    PHOENIX_HOST: str = "http://localhost"

    # --- PROMPTS ---
    # Storing prompts here allows for easier tuning without touching code
    PROMPTS: Dict[str, str] = {
        "extraction_system": (
            "You are an expert Intelligence Analyst. Analyze the provided content against "
            "the User Profile.\n\n"
            "USER PROFILE: {user_profile}\n"
            "CONTENT_CHUNK: {content}\n\n"
            "{format_instructions}"
        ),
        "rewrite_template": (
            "The user asked for: '{original}'.\n"
            "The previous search '{current}' yielded no relevant intelligence.\n"
            "Generate ONE new, alternative search query that is broader or uses synonyms.\n"
            "Output ONLY the query string."
        ),
        "summarize_system": (
            "<|system|>\n"
            "You are an executive analyst. Create a high-level briefing based ONLY on the provided intelligence.\n"
            "Cite sources where possible.\n"
            "<|end|>\n"
            "<|user|>\n"
            "USER PROFILE: {user_profile}\n\n"
            "INTELLIGENCE REPORTS:\n"
            "{context}\n\n"
            "TASK: Write a structured executive summary.\n"
            "<|end|>\n"
            "<|assistant|>"
        ),
    }

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
