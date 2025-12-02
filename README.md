# 📰 LLM RSS Digest Agent

**An intelligent, autonomous agent that monitors RSS feeds, filters noise based on a User Profile, and generates executive briefings using local LLMs.**

This project demonstrates an advanced **Agentic RAG (Retrieval-Augmented Generation)** workflow using LangGraph, LangChain, and local Hugging Face models. It moves beyond simple "chat with data" to perform active intelligence analysis: fetching, chunking, retrieving, reranking, extracting, summarizing, and self-evaluating.

---

## 🚀 Key Features

* **Advanced RAG Pipeline:** Implements **Hybrid Search** (BM25 + Chroma Vector) and **Cross-Encoder Reranking** for high-precision retrieval.
* **Agentic Workflow:** Built on **LangGraph**, allowing the system to self-correct. If a search yields no results, the agent rewrites its own query and tries again.
* **Modular Architecture:** Clean separation of concerns with a dedicated `src` directory for models, ingestion, and agent logic.
* **Local LLM Inference:** Runs entirely on your hardware (or cloud GPU) using 4-bit quantization (BitsAndBytes) for efficiency. Default model: `Microsoft Phi-4-mini`.
* **Structured Intelligence:** Uses **Pydantic** to force the LLM to output structured data (Boolean relevance, specific fact bullets) rather than unstructured text.
* **Evaluation & Telemetry:** Integrated with **Ragas** for quality metrics (Faithfulness, Answer Relevance) and **OpenTelemetry/Phoenix** for tracing.

---

## 🛠️ AI Engineering Concepts Implemented

This repository serves as a reference implementation for several key AI Engineering skills:

| Skill / Concept | Implementation Details | Reference |
| :--- | :--- | :--- |
| **Agentic State Management** | Uses `LangGraph` to manage state (`AgentState`) containing queries, docs, and retry counts. | `src/agent.py` |
| **Hybrid Retrieval** | Combines Dense retrieval (Chroma/Embeddings) with Sparse retrieval (BM25/Keywords). | `src/ingestion.py` |
| **Re-Ranking** | Uses a Cross-Encoder (`BAAI/bge-reranker`) to score the relevancy of retrieved documents. | `src/agent.py` |
| **Semantic Chunking** | Uses embedding breakpoints to split text where topics logically shift, rather than fixed characters. | `src/ingestion.py` |
| **Model Quantization** | Loads LLMs in 4-bit precision (`nf4`) using `BitsAndBytes` to reduce VRAM usage. | `src/models.py` |
| **RAG Evaluation** | Automates quality testing using `Ragas` to score Faithfulness and Relevance of the final output. | `src/evaluation.py` |
| **Observability** | Implements OpenTelemetry tracing to monitor agent latency and decision paths. | `main.py` |

---

## ⚙️ Architecture

The agent follows a cyclical graph architecture:

1.  **Ingest:** Async fetching of RSS feeds $\rightarrow$ Semantic Chunking $\rightarrow$ Indexing (ChromaDB + BM25).
2.  **Retrieve:** Hybrid search based on the `current_query`.
3.  **Rerank:** Top K results are re-scored by a Cross-Encoder.
4.  **Extract:** The LLM analyzes each chunk against the `USER_PROFILE`.
5.  **Decision Node:**
    * *Found Intel?* $\rightarrow$ **Summarize**.
    * *No Intel?* $\rightarrow$ **Rewrite Query** (Broaden terms/Synonyms) $\rightarrow$ Loop back to **Retrieve**.
6.  **Evaluate:** (Optional) Runs Ragas metrics on the generated result.

---

## 📂 Project Structure

```text
├── config.py           # Configuration (Prompts, Model IDs, Thresholds)
├── main.py             # Entry point: Setup, Telemetry, and Execution
├── README.md           # Documentation
└── src
    ├── agent.py        # LangGraph nodes, State definition, and Flow logic
    ├── evaluation.py   # Ragas evaluation logic
    ├── ingestion.py    # RSS loading, Semantic Chunking, and Vector Store creation
    └── models.py       # Singleton Manager for LLM, Embeddings, and Reranker
````

## 📦 Installation

### Prerequisites

  * Python 3.10+
  * CUDA-enabled GPU (Recommended for local LLM inference)

### Setup

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/llm-rss-digest.git](https://github.com/your-username/llm-rss-digest.git)
    cd llm-rss-digest
    ```

2.  **Install dependencies:**

    ```bash
    pip install langchain langchain-community langchain-huggingface langchain-chroma \
    langchain-experimental langgraph transformers sentence-transformers \
    pydantic-settings bitsandbytes accelerate ragas opentelemetry-api \
    opentelemetry-sdk opentelemetry-exporter-otlp
    ```

    *(Note: You may need to install `torch` specifically for your CUDA version).*

3.  **Environment Variables (Optional):**
    Create a `.env` file for LangSmith or Phoenix tracing.

    ```env
    # For Phoenix/OTEL
    PHOENIX_PORT=6006
    PHOENIX_HOST=http://localhost
    ```

-----

## 🖥️ Usage

1.  **Configure the Agent:**
    Open `config.py` in the root directory to set your target topic and profile.

    ```python
    # config.py
    QUERY: str = "AI regulation"
    USER_PROFILE: str = "Tech Analyst looking for EU compliance risks"
    DOMAINS: List[str] = ["[https://feeds.bbci.co.uk/news/technology/rss.xml](https://feeds.bbci.co.uk/news/technology/rss.xml)"]
    ENABLE_EVALUATION: bool = True
    ```

2.  **Run the Agent:**
    Execute the main script from the root directory:

    ```bash
    python main.py
    ```

3.  **Output:**

      * The script will log the retrieval process (Polling -\> Chunking -\> Reranking).
      * A **Final Executive Brief** will be printed to the console.
      * If enabled, a **Ragas Evaluation Report** will follow.

-----

## ⚠️ Hardware Notes

This project defaults to using **Microsoft Phi-4-mini-instruct**.

  * **VRAM:** Requires approximately 4GB-6GB VRAM with 4-bit quantization.
  * **CPU Mode:** Possible but significantly slower. To run on CPU, remove `load_in_4bit=True` config in `src/models.py` and adjust device settings in `ModelManager`.