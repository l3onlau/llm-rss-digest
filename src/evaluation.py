import logging
import math
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy

import src.models as models
from config import settings

logger = logging.getLogger("RSS_Agent")


def run_evaluation(state: dict):
    """
    Runs Ragas evaluation. Dynamically injects LoRA adapter if available.
    Returns: Tuple(bool, dict) -> (is_passed, scores)
    """
    logger.info("📊 Starting Ragas Evaluation...")

    if (
        not state.get("extracted_data")
        or "No significant intelligence" in state["final_digest"]
    ):
        logger.warning("⚠️ No extracted data to evaluate. Skipping Ragas.")
        return False, {}

    llm_pipeline = models.get_llm()
    base_model = llm_pipeline.pipeline.model
    adapter_loaded = False
    adapter_name = adapter_name = Path(settings.RAGAS_ADAPTER_PATH).stem

    # --- Adapter Injection ---
    if settings.RAGAS_ADAPTER_PATH and Path(settings.RAGAS_ADAPTER_PATH).exists():
        try:
            logger.info(
                f"🔌 Loading RAGAS Adapter from {settings.RAGAS_ADAPTER_PATH}..."
            )

            # Check if model supports PEFT loading
            if hasattr(base_model, "load_adapter"):
                base_model.load_adapter(
                    settings.RAGAS_ADAPTER_PATH, adapter_name=adapter_name
                )
                base_model.set_adapter(adapter_name)
                adapter_loaded = True
                logger.info("✅ Adapter injected successfully.")
            else:
                logger.warning(
                    "⚠️ Model does not support dynamic adapter loading. Using Base Model."
                )
        except Exception as e:
            logger.error(f"❌ Failed to load adapter: {e}. Proceeding with Base Model.")

    # --- Data Prep ---
    full_query = f"{state['original_query']} for user profile: {state['user_profile']}"
    data = {
        "user_input": [full_query],
        "response": [state["final_digest"]],
        "retrieved_contexts": [state["extracted_data"]],
    }
    dataset = Dataset.from_dict(data)

    evaluator_llm = LangchainLLMWrapper(llm_pipeline)
    evaluator_embeddings = LangchainEmbeddingsWrapper(models.get_embeddings())

    metrics = []
    if "faithfulness" in settings.RAGAS_METRICS:
        metrics.append(Faithfulness(llm=evaluator_llm))
    if "answer_relevance" in settings.RAGAS_METRICS:
        metrics.append(
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        )

    is_passed = True
    scores = {}

    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        scores = results

        # Normalize scores (handle NaNs)
        rel_score = scores.get("answer_relevance", 0.0) or 0.0
        faith_score = scores.get("faithfulness", 0.0) or 0.0

        # Safe handling of math.nan
        if math.isnan(rel_score):
            rel_score = 0.0
        if math.isnan(faith_score):
            faith_score = 0.0

        print("\n" + "=" * 30)
        print(f"📈 RAGAS SCORE (Adapter: {adapter_loaded})")
        print(f"   - Relevance: {rel_score:.2f}")
        print(f"   - Faithfulness: {faith_score:.2f}")
        print("=" * 30)

        if rel_score < settings.EVALUATION_THRESHOLD:
            logger.error(
                f"❌ REJECTED: Relevance {rel_score:.2f} < {settings.EVALUATION_THRESHOLD}"
            )
            is_passed = False

        if faith_score < settings.EVALUATION_THRESHOLD:
            logger.error(
                f"❌ REJECTED: Faithfulness {faith_score:.2f} < {settings.EVALUATION_THRESHOLD}"
            )
            is_passed = False

    except Exception as e:
        logger.error(f"Ragas evaluation failed: {e}")
        is_passed = False

    finally:
        # --- Cleanup Adapter ---
        if adapter_loaded:
            try:
                logger.info("🔌 Unloading RAGAS Adapter...")
                base_model.delete_adapter(adapter_name)
                logger.info("✅ Reverted to Base Model.")
            except Exception as e:
                logger.error(f"⚠️ Failed to unload adapter: {e}")

    return is_passed, scores
