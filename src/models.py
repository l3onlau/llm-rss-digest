import logging
import torch
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


class ModelManager:
    """
    Singleton-like manager for heavy AI models.
    Handles lazy loading and hardware acceleration (CUDA/MPS/CPU).
    """

    def __init__(self):
        self._llm = None
        self._embeddings = None
        self._reranker = None
        self._tokenizer = None
        self._device = self._get_device()

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        return "cpu"

    @property
    def embeddings(self):
        if not self._embeddings:
            logger.info(
                f"🧠 Loading Embeddings ({self._device}): {settings.EMBEDDING_MODEL}"
            )
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": self._device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def reranker(self):
        if not self._reranker:
            logger.info(
                f"⚖️ Loading Reranker ({self._device}): {settings.RERANKER_MODEL}"
            )
            self._reranker = CrossEncoder(settings.RERANKER_MODEL, device=self._device)
        return self._reranker

    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_ID)
        return self._tokenizer

    @property
    def llm(self):
        if not self._llm:
            logger.info(f"🧠 Loading LLM: {settings.LLM_MODEL_ID}...")

            model_kwargs = {"device_map": "auto"}

            # Only use BitsAndBytes if CUDA is available
            if self._device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = bnb_config
            else:
                # Load normally for CPU/Mac (consider using lighter models for these targets)
                model_kwargs["torch_dtype"] = torch.float16

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    settings.LLM_MODEL_ID, **model_kwargs
                )

                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    max_new_tokens=1024,
                    temperature=0.1,
                    return_full_text=False,
                    # MPS sometimes requires explicit device handling in pipeline
                    device=self._device if self._device != "cuda" else None,
                )
                self._llm = HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                logger.error(f"Failed to load LLM: {e}")
                raise

        return self._llm

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncates text to fit context window based on token count."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        return (
            self.tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True) + "..."
        )
