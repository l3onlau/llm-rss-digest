# 📰 LLM RSS Digest Agent (v2.0)

**An intelligent, autonomous agent that monitors RSS feeds, filters noise based on a User Profile, and generates executive briefings using local LLMs.**

This project demonstrates an advanced **Agentic RAG (Retrieval-Augmented Generation)** workflow using LangGraph, LangChain, and local Hugging Face models. It moves beyond simple "chat with data" to perform active intelligence analysis: fetching, chunking, retrieving, reranking, extracting, summarizing, and self-evaluating.

-----

## 🚀 Key Features

  * **Hybrid Search Architecture:** Combines **Dense Vector Search** (ChromaDB/Embeddings) with **Sparse Keyword Search** (BM25) to capture both semantic meaning and exact keyword matches.
  * **Agentic Self-Correction:** Built on **LangGraph**. If a search yields no results, the agent rewrites its own query (using synonyms/broader terms) and attempts retrieval again.
  * **Custom Adapter Support:** Supports dynamically loading **LoRA adapters** during the evaluation phase to act as specialized "Judge" models.
  * **Quality Guardrails:** Integrated with **Ragas**. The agent scores its own output for *Faithfulness* and *Answer Relevance* before presenting it to the user.
  * **Structured Intelligence:** Uses **Pydantic** to force the LLM to output structured data (Boolean relevance, specific fact bullets) rather than unstructured text.

-----

## 🛠️ AI Engineering Concepts

This repository serves as a reference implementation for several advanced AI Engineering skills:

| Concept | Implementation | File |
| :--- | :--- | :--- |
| **Hybrid Retrieval** | Combines `Chroma` (Vector) and `BM25Retriever` (Keyword). | `src/ingestion.py` |
| **Re-Ranking** | Uses a Cross-Encoder (`BAAI/bge-reranker`) to re-score top-K documents. | `src/agent.py` |
| **Fine-Tuning (LoRA)** | Script to train a specialized adapter for Ragas evaluation. | `train_ragas_adapter.py` |
| **Dynamic Adapters** | Injecting LoRA adapters at runtime for specific tasks without reloading the base model. | `src/evaluation.py` |
| **Semantic Chunking** | Splitting text based on embedding breakpoints rather than fixed characters. | `src/ingestion.py` |
| **Quantization** | Loading LLMs in 4-bit precision (`nf4`) to optimize VRAM. | `src/models.py` |

-----

## ⚙️ Workflow Architecture

1.  **Ingest:** Async fetching of RSS feeds $\rightarrow$ Semantic Chunking $\rightarrow$ Indexing (ChromaDB + BM25).
2.  **Retrieve:** Hybrid search based on the `current_query`.
3.  **Rerank:** Top results are re-scored by a Cross-Encoder to remove false positives.
4.  **Extract:** The LLM analyzes each chunk against the `USER_PROFILE` and extracts bullet points.
5.  **Decision Loop:**
      * *Found Intel?* $\rightarrow$ **Summarize**.
      * *No Intel?* $\rightarrow$ **Rewrite Query** $\rightarrow$ Loop back to **Retrieve**.
6.  **Evaluate:** The system injects a "Judge" adapter to score the final summary using Ragas metrics.

-----

## 📦 Installation

### Prerequisites

  * Python 3.10+
  * CUDA-enabled GPU (Recommended) or Mac M1/M2 (MPS supported)

### Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/llm-rss-digest.git
    cd llm-rss-digest
    ```

2.  **Install dependencies:**

    ```bash
    pip install langchain langchain-community langchain-huggingface langchain-chroma \
    langchain-experimental langgraph transformers sentence-transformers \
    pydantic-settings bitsandbytes accelerate ragas opentelemetry-api \
    opentelemetry-sdk opentelemetry-exporter-otlp trl peft
    ```

3.  **Environment Variables (Optional):**
    Create a `.env` file for tracing configuration.

    ```env
    PHOENIX_PORT=6006
    PHOENIX_HOST=http://localhost
    ```

-----

## 🖥️ Usage

### 1\. Basic Execution

Open `config.py` to set your topic (e.g., "AI Regulation") and User Profile (e.g., "Compliance Officer"). Then run:

```bash
python main.py
```

### 2\. Training a Custom Judge (Optional)

To improve the accuracy of the evaluation phase, you can fine-tune a LoRA adapter on a Q\&A dataset (like SQuAD) to act as a better judge.

```bash
python train_ragas_adapter.py
```

*This will save a LoRA adapter to `./ragas_adapter`. The main agent will automatically detect and load this adapter during the evaluation phase.*

-----

## ⚠️ Hardware Notes

  * **VRAM:** The default model (`Microsoft Phi-4-mini`) with 4-bit quantization requires \~4-6GB VRAM.
  * **MPS (Mac):** The code automatically detects Apple Silicon and runs in `float16` mode.
  * **CPU:** Supported, but significantly slower.

## 📂 Project Structure

```text
├── config.py                # Configuration & Prompts
├── main.py                  # Entry point & Orchestration
├── train_ragas_adapter.py   # LoRA Fine-tuning script
└── src
    ├── agent.py             # LangGraph Nodes & Flow
    ├── evaluation.py        # Ragas logic & Adapter injection
    ├── ingestion.py         # RSS Fetching, Chunking & Indexing
    └── models.py            # Model Singleton Manager
```