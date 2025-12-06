import logging
import math
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from config import settings
import src.models as models

logger = logging.getLogger("RSS_Agent")


def run_evaluation(state: dict):
    """
    Runs Ragas evaluation and determines if output passes quality checks.
    Returns: Tuple(bool, dict) -> (is_passed, scores)
    """
    logger.info("📊 Starting Ragas Evaluation...")

    if (
        not state.get("extracted_data")
        or "No significant intelligence" in state["final_digest"]
    ):
        logger.warning("⚠️ No extracted data to evaluate. Skipping Ragas.")
        return False, {}

    full_query = f"{state['original_query']} for user profile: {state['user_profile']}"
    contexts = state["extracted_data"]

    data = {
        "user_input": [full_query],
        "response": [state["final_digest"]],
        "retrieved_contexts": [contexts],
    }
    dataset = Dataset.from_dict(data)

    # Access singleton models directly
    evaluator_llm = LangchainLLMWrapper(models.get_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(models.get_embeddings())

    metrics = []
    if "faithfulness" in settings.RAGAS_METRICS:
        metrics.append(Faithfulness(llm=evaluator_llm))
    if "answer_relevance" in settings.RAGAS_METRICS:
        metrics.append(
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        )

    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        scores = results

        # Helper to safely get scores
        rel_score = scores.get("answer_relevance", 0.0)
        faith_score = scores.get("faithfulness", 0.0)

        if math.isnan(rel_score):
            rel_score = 0.0
        if math.isnan(faith_score):
            faith_score = 0.0

        # Calculate average
        valid_scores = [
            v
            for k, v in scores.items()
            if isinstance(v, (int, float)) and not math.isnan(v)
        ]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        print("\n" + "=" * 30)
        print(
            f"📈 RAGAS SCORE: {avg_score:.2f} (Thresh: {settings.EVALUATION_THRESHOLD})"
        )
        print(f"   - Relevance: {rel_score}")
        print(f"   - Faithfulness: {faith_score}")
        print("=" * 30)

        is_passed = True
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

        return is_passed, scores

    except Exception as e:
        logger.error(f"Ragas evaluation failed: {e}")
        return False, {}
