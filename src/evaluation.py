import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from config import settings

logger = logging.getLogger("RSS_Agent")


class RagasEvaluator:
    def __init__(self, model_manager):
        self.models = model_manager

    def run_eval(self, state):
        logger.info("📊 Starting Ragas Evaluation...")

        full_query = (
            f"{state['original_query']} for user profile: {state['user_profile']}"
        )
        contexts = (
            state["extracted_data"]
            if state["extracted_data"]
            else [d.page_content for d in state["reranked_docs"]]
        )

        data = {
            "user_input": [full_query],
            "response": [state["final_digest"]],
            "retrieved_contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)

        evaluator_llm = LangchainLLMWrapper(self.models.llm)
        evaluator_embeddings = LangchainEmbeddingsWrapper(self.models.embeddings)

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
            print("\n" + "=" * 30)
            print("📈 RAGAS EVALUATION REPORT")
            print("=" * 30)
            print(results)
            return results
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            return None
