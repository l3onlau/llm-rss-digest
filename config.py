from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class PromptSettings:
    """Centralized Prompt Templates."""

    EXTRACTION_SYSTEM: str = (
        "You are an expert Intelligence Analyst. Analyze the provided content "
        "relevance to the Query: '{query}'.\n\n"
        "CONTENT_CHUNK: {content}\n\n"
        "{format_instructions}"
    )

    REWRITE_TEMPLATE: str = (
        "The user asked for: '{original}'.\n"
        "The previous search '{current}' yielded no relevant intelligence.\n"
        "Generate ONE new, alternative search query that is broader or uses synonyms.\n"
        "Output ONLY the query string."
    )

    SUMMARIZE_SYSTEM: str = (
        "<|system|>\n"
        "You are an executive analyst. Create a high-level briefing based ONLY on the provided intelligence.\n"
        "Focus on answering the Query: '{query}'.\n"
        "Cite sources where possible.\n"
        "<|end|>\n"
        "<|user|>\n"
        "INTELLIGENCE REPORTS:\n"
        "{context}\n\n"
        "TASK: Write a structured executive summary.\n"
        "<|end|>\n"
        "<|assistant|>"
    )


class Settings(BaseSettings):
    # --- Content Settings ---
    DOMAINS: List[str] = [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml",
    ]
    QUERY: str = "Economic trends"

    # --- Model IDs ---
    LLM_MODEL_ID: str = "microsoft/Phi-4-mini-instruct"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # --- Vector DB & Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent
    CHROMA_PATH: str = str(BASE_DIR / "chroma_db")
    RAGAS_ADAPTER_PATH: Optional[str] = str(BASE_DIR / "ragas_adapter")

    # --- RAG Parameters ---
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200
    K_RETRIEVAL: int = 15
    K_FINAL: int = 5
    MAX_RETRIES: int = 2
    MIN_RELEVANCE_SCORE: int = 4
    MAX_INPUT_TOKENS: int = 1500

    # --- Telemetry & Eval ---
    PHOENIX_PORT: int = 6006
    PHOENIX_HOST: str = "http://localhost"
    ENABLE_TRACING: bool = True
    ENABLE_EVALUATION: bool = True
    RAGAS_METRICS: List[str] = ["faithfulness", "answer_relevance"]
    EVALUATION_THRESHOLD: float = 0.6

    prompts: PromptSettings = PromptSettings()

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
