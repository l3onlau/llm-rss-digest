# 📰 LLM RSS Digest Agent

**An intelligent, autonomous agent that monitors RSS feeds, filters noise based on a User Profile, and generates executive briefings using local LLMs.**

This project demonstrates an advanced **Agentic RAG (Retrieval-Augmented Generation)** workflow using LangGraph, LangChain, and local Hugging Face models. It moves beyond simple "chat with data" to perform active intelligence analysis: fetching, chunking, retrieving, reranking, extracting, and summarizing.

-----

## 🚀 Key Features

  * **Advanced RAG Pipeline:** Implements **Hybrid Search** (BM25 + Chroma Vector) and **Cross-Encoder Reranking** for high-precision retrieval.
  * **Agentic Workflow:** Built on **LangGraph**, allowing the system to self-correct. If a search yields no results, the agent rewrites its own query and tries again.
  * **Local LLM Inference:** Runs entirely on your hardware (or cloud GPU) using 4-bit quantization (BitsAndBytes) for efficiency. Default model: `Microsoft Phi-4-mini`.
  * **Structured Intelligence:** Uses **Pydantic** to force the LLM to output structured data (Boolean relevance, specific fact bullets) rather than unstructured text.
  * **Observability:** Integrated with **LangSmith** for tracing agent thought processes and latency.

-----

## 🛠️ AI Engineering Concepts Implemented

This repository serves as a reference implementation for several key AI Engineering skills:

| Skill / Concept | Implementation Details | Reference |
| :--- | :--- | :--- |
| **Agentic State Management** | Uses `LangGraph` to manage state (`AgentState`) containing queries, docs, and retry counts. | `llm-rss-digest.py` (Lines 66-76) |
| **Hybrid Retrieval** | Combines Dense retrieval (Chroma/Embeddings) with Sparse retrieval (BM25/Keywords) to capture both semantic meaning and exact matches. | `llm-rss-digest.py` (Lines 226-239) |
| **Re-Ranking** | Uses a Cross-Encoder (`BAAI/bge-reranker`) to score the relevancy of retrieved documents, significantly improving context quality before the LLM sees it. | `llm-rss-digest.py` (Lines 242-259) |
| **Semantic Chunking** | Instead of fixed character splitting, it uses embedding breakpoints to split text where topics logically shift. | `llm-rss-digest.py` (Lines 185-197) |
| **Model Quantization** | Loads LLMs in 4-bit precision (`nf4`) using `BitsAndBytes` to reduce VRAM usage while maintaining performance. | `llm-rss-digest.py` (Lines 123-130) |
| **Structured Extraction** | Enforces schema compliance using `PydanticOutputParser`, ensuring downstream functions receive clean JSON-like objects. | `llm-rss-digest.py` (Lines 262-270) |

-----

## ⚙️ Architecture

The agent follows a cyclical graph architecture:

1.  **Ingest:** Async fetching of RSS feeds $\rightarrow$ Semantic Chunking $\rightarrow$ Indexing (ChromaDB + BM25).
2.  **Retrieve:** Hybrid search based on the `current_query`.
3.  **Rerank:** Top K results are re-scored by a Cross-Encoder.
4.  **Extract:** The LLM analyzes each chunk against the `USER_PROFILE`.
5.  **Decision Node:**
      * *Found Intel?* $\rightarrow$ **Summarize**.
      * *No Intel?* $\rightarrow$ **Rewrite Query** (Broaden terms/Synonyms) $\rightarrow$ Loop back to **Retrieve**.

-----

## 📦 Installation

### Prerequisites

  * Python 3.10+
  * CUDA-enabled GPU (Recommended for local LLM inference)

### Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/llm-rss-digest.git
    cd llm-rss-digest
    ```

2.  **Install dependencies:**

    ```bash
    pip install langchain langchain-community langchain-huggingface langchain-chroma langchain-experimental langgraph transformers sentence-transformers pydantic-settings bitsandbytes accelerate
    ```

    *(Note: You may need to install `torch` specifically for your CUDA version).*

3.  **Environment Variables (Optional):**
    Create a `.env` file for LangSmith tracing (optional).

    ```env
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=your_langchain_api_key
    ```

-----

## 🖥️ Usage

1.  **Configure the Agent:**
    Open `config.py` to set your target topic and profile.

    ```python
    # config.py
    QUERY: str = "AI regulation"
    USER_PROFILE: str = "Tech Analyst looking for EU compliance risks"
    DOMAINS: List[str] = ["https://feeds.bbci.co.uk/news/technology/rss.xml"]
    ```

2.  **Run the Agent:**

    ```bash
    python llm-rss-digest.py
    ```

3.  **Output:**
    The script will log the retrieval process and output a final Markdown-formatted executive brief in the console.

-----

## 📂 Project Structure

  * **`config.py`**: Centralized configuration using `pydantic_settings`. Controls model selection (`phi-4`), thresholds (`MIN_RELEVANCE_SCORE`), and system prompts.
  * **`llm-rss-digest.py`**: The main application logic.
      * `ModelManager`: Handles lazy loading of heavy models (LLM, Embeddings, Reranker).
      * `IngestionEngine`: Fetches RSS and handles Semantic Chunking.
      * `AgentNodes`: The functional logic for the graph (Retrieve, Extract, Rewrite).
      * `build_graph`: Definitions for the LangGraph state machine.

-----

## ⚠️ Hardware Notes

This project defaults to using **Microsoft Phi-4-mini-instruct**.

  * **VRAM:** Requires approximately 4GB-6GB VRAM with 4-bit quantization.
  * **CPU Mode:** Possible but significantly slower. To run on CPU, remove `load_in_4bit=True` config in `ModelManager` and adjust device settings.