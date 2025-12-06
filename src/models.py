import logging
import torch
from functools import lru_cache
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from sentence_transformers import CrossEncoder
from config import settings

logger = logging.getLogger("RSS_Agent")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_embeddings():
    """Singleton accessor for Embeddings."""
    device = get_device()
    logger.info(f"🧠 Loading Embeddings ({device}): {settings.EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_reranker():
    """Singleton accessor for Reranker."""
    device = get_device()
    logger.info(f"⚖️ Loading Reranker ({device}): {settings.RERANKER_MODEL}")
    return CrossEncoder(settings.RERANKER_MODEL, device=device)


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(settings.LLM_MODEL_ID)


@lru_cache(maxsize=1)
def get_llm():
    """Singleton accessor for LLM."""
    device = get_device()
    logger.info(f"🧠 Loading LLM: {settings.LLM_MODEL_ID}...")

    model_kwargs = {"device_map": "auto"}

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_ID, **model_kwargs
        )

        # HuggingFace pipeline needs explicit device argument for MPS/CPU if not using device_map="auto" fully
        pipe_device = device if device != "cuda" else None

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=get_tokenizer(),
            max_new_tokens=1024,
            temperature=0.1,
            return_full_text=False,
            device=pipe_device,
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        raise


def truncate_text(text: str, max_tokens: int) -> str:
    """Helper function to truncate text based on token count."""
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True) + "..."
