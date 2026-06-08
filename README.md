# Hierarchical RAG Memory System

A conversational AI with long-term memory, built from scratch — no LangChain, no vector DB libraries. Uses a two-level hierarchical retrieval architecture backed by SQLite and NumPy.

---

## Overview

Most RAG systems dump everything into a flat index and brute-force similarity search. This system organizes memories into topic clusters, doing coarse-to-fine retrieval:

1. **Master table** — stores a mean embedding per topic cluster + keywords
2. **Cluster subtables** — store individual memory entries (summaries of past exchanges)
3. **Query time** — first find relevant clusters, then re-rank within them

Retrieval uses hybrid scoring: `0.6 × cosine similarity + 0.4 × keyword overlap`

---

## Project Structure

```
├── main.py               # Main conversation loop
├── rag_system.py         # Memory storage, retrieval, embedding logic
├── doc_reader.py         # Document ingestion and chunk retrieval
├── retrieving_clf.py     # Classifier to check if memory retrieval is needed
├── loader.py             # Model clients, NLP utilities
└── Data/
    ├── complex_rag.db    # SQLite database (master + cluster tables)
    └── docs/
        ├── doc_ref.json  # Document registry
        └── npz_files/    # Stored chunk embeddings per document
```

---

## How It Works

### Memory System (`rag_system.py`)

**Saving a turn:**
- Conversation exchanges are summarized by an LLM into structured JSON (`summary` + `facts`)
- Each summary is embedded and compared against existing cluster centroids
- If similarity + keyword overlap exceeds threshold → insert into matching cluster(s)
- Otherwise → create a new cluster
- Cluster centroids are updated with a **rolling mean** (no full recomputation)

**Retrieving:**
- Query is embedded + keywords extracted
- Master table is searched for relevant clusters
- Matching clusters are searched for relevant individual summaries
- Results are passed as context to the LLM

### Document Mode (`doc_reader.py`)

Load PDFs or TXT files and query them directly. Two ingestion modes:

- **Raw chunks** — embeds the actual text, better for detail-heavy queries
- **Summaries** — LLM summarizes chunks first, better for high-level questions

Chunks are sized between 1700–2000 characters with sentence-level overlap at boundaries for context continuity.

Retrieval uses the same hybrid scoring as the memory system.

---

## Setup

### Requirements

```bash
pip install numpy pdfplumber spacy
python -m spacy download en_core_web_sm
```

You'll also need access to:
- An LLM API (used via OpenAI-compatible client in `loader.py`)
- An embeddings model endpoint

### Configuration

Set up your API clients in `loader.py`. The project uses:
- `openai/gpt-oss-120b` — main conversation model
- `nvidia/nemotron-3-nano-4b` — summarization model
- `text-embedding-embeddinggemma-300m` — embeddings

### Run

```bash
python main.py
```

---

## Key Design Decisions

| Decision | Reasoning |
|---|---|
| Hierarchical index | Avoids brute-force search across all memories as history grows |
| Rolling mean centroid | Efficient cluster updates without storing all embeddings at master level |
| Hybrid scoring (60/40) | Embeddings alone miss topic drift; keyword overlap adds precision |
| Dual doc modes | Tradeoff between detail coverage (raw) and noise reduction (summary) |
| SQLite over vector DB | Zero dependencies, good enough for personal-scale memory |

---

## Built With

- Python
- SQLite3
- NumPy
- pdfplumber
- spaCy
- Open-source LLM + embedding models