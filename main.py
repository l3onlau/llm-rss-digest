import asyncio
import logging
import os
import sys

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from config import settings
from src.agent import build_graph
from src.evaluation import run_evaluation
from src.ingestion import run_ingestion

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RSS_Agent")


def setup_telemetry():
    """Initializes OpenTelemetry traces if enabled."""
    if not settings.ENABLE_TRACING:
        return

    os.environ["PHOENIX_PROJECT_NAME"] = "RSS_Digest_Agent"
    os.environ["PHOENIX_HOST"] = f"{settings.PHOENIX_HOST}:{settings.PHOENIX_PORT}"

    # Standard OTLP Exporter setup
    trace.set_tracer_provider(TracerProvider())
    otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))
    logger.info("🔭 Telemetry enabled (Phoenix/OTEL).")


async def main():
    print("🚀 Initializing AI Regulatory Digest Agent...")
    setup_telemetry()

    # 1. Ingestion Phase
    chroma, bm25 = await run_ingestion()
    if not chroma:
        sys.exit("Critical Error: Knowledge base construction failed. Check RSS feeds.")

    # 2. Build Agent
    app = build_graph(chroma, bm25)

    initial_state = {
        "original_query": settings.QUERY,
        "current_query": settings.QUERY,
        "retry_count": 0,
        "extracted_data": [],
        "candidate_docs": [],
        "reranked_docs": [],
    }

    try:
        # 3. Execution Phase
        result = await app.ainvoke(initial_state)
        final_digest = result.get("final_digest", "No data generated.")

        # 4. Evaluation Phase
        if settings.ENABLE_EVALUATION and result.get("extracted_data"):
            print("\n🧐 Validating Output Quality...")
            is_valid = run_evaluation(result)

            header = (
                "📢 FINAL EXECUTIVE BRIEF (Verified)"
                if is_valid
                else "⛔ OUTPUT REJECTED"
            )
            print("\n" + "=" * 60)
            print(header)
            print("=" * 60)

            if is_valid:
                print(final_digest)
            else:
                print("The generated response did not meet quality standards.")
        else:
            print("\n" + "=" * 60)
            print("📢 FINAL EXECUTIVE BRIEF (Unverified)")
            print("=" * 60)
            print(final_digest)

    except Exception as e:
        logger.error(f"Graph execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
