# LLM-RSS-Digest

## Functionality
- Loads RSS feeds from specified domains.
- Splits content into chunks using RecursiveCharacterTextSplitter.
- Embeds chunks via HuggingFaceEmbeddings.
- Stores in Chroma vector database.
- Retrieves top-k documents for query.
- Extracts relevant text segments matching user profile.
- Synthesizes into bullet-point summary using LLM.
- Employs LangGraph for workflow: extract → summarize.

## Dependencies
- Python 3.x.
- Packages: langchain-community, langchain-huggingface, langchain-chroma, langchain-text-splitters, langchain-core, langgraph, transformers, torch, dotenv, argparse.

## Setup
- Clone repository.
- Install dependencies: `pip install -r requirements.txt` (create if absent).
- Configure `.env` from `.env.template`:
  - DOMAINS: comma-separated RSS URLs.
  - HUGGINGFACE_EMBEDDING_MODEL: default "all-MiniLM-L6-v2".
  - CHUNK_SIZE: default 1000.
  - CHUNK_OVERLAP: default 200.
  - LLM_MODEL_NAME: default "microsoft/Phi-4-mini-instruct".
  - USER_PROFILE: default "Busy professional seeking quick world news highlights.".

## Execution
- Run script:
```bash
python llm-rss-digest.py --query "Top world news highlights" --k 10 --max_tokens 512 --temperature 0.7
```
- Outputs final summary in console.

## Workflow Details
- Retrieval: query → retriever → documents.
- Extraction: LLM filters raw segments relevant to profile.
- Summarization: LLM generates actionable bullets from extracts.
- Quantization: 4-bit for LLM efficiency.

## Customization
- Adjust args: query, k, max_tokens, temperature.
- Modify prompts in extract_critical_updates, generate_final_summary for specificity.

## Limitations
- No internet beyond RSS load.
- LLM output variability from sampling.
- Vector store ephemeral per run.