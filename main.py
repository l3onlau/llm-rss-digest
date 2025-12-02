import sys
import os
import asyncio
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from config import settings
from src.models import ModelManager
from src.ingestion import IngestionEngine
from src.agent import build_graph
from src.evaluation import RagasEvaluator

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def setup_telemetry():
    if not settings.ENABLE_TRACING:
        return

    os.environ["PHOENIX_PROJECT_NAME"] = "RSS_Digest_Agent"
    os.environ["PHOENIX_HOST"] = f"{settings.PHOENIX_HOST}:{settings.PHOENIX_PORT}"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
        f"{settings.PHOENIX_HOST}:{settings.PHOENIX_PORT}"
    )

    trace.set_tracer_provider(TracerProvider())
    otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))


async def main():
    print("🚀 Initializing AI Regulatory Digest Agent...")
    setup_telemetry()

    # Initialize Core Components
    models = ModelManager()
    ingestor = IngestionEngine(models)

    # Load Data
    chroma, bm25 = await ingestor.run()
    if not chroma:
        sys.exit("Critical Error: Knowledge base construction failed.")

    # Build Agent
    app = build_graph(chroma, bm25, models)

    initial_state = {
        "original_query": settings.QUERY,
        "current_query": settings.QUERY,
        "user_profile": settings.USER_PROFILE,
        "retry_count": 0,
        "extracted_data": [],
        "candidate_docs": [],
        "reranked_docs": [],
    }

    try:
        result = await app.ainvoke(initial_state)

        print("\n" + "=" * 60)
        print("📢 FINAL EXECUTIVE BRIEF")
        print("=" * 60)
        print(result.get("final_digest", "No data generated."))

        if settings.ENABLE_EVALUATION and result.get("extracted_data"):
            evaluator = RagasEvaluator(models)
            evaluator.run_eval(result)

    except Exception as e:
        logging.error(f"Graph execution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
