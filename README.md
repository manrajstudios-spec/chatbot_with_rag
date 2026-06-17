# Multi Agent Chat Bot

Yuzu is a personal AI companion built from scratch. She talks like a real friend, remembers things over time, can read documents, search the web, and respond using her own voice. Every core component — the retrieval system, the graph search, the chunker — is custom-built with no off-the-shelf RAG frameworks.

---

## Features

- **Conversational AI** — Powered by GPT-OSS 120B via Groq. Yuzu responds casually and naturally, not like an assistant.
- **Custom RAG (Long-Term Memory)** — A hierarchical retrieval system built on SQLite. Past conversations are stored, grouped by topic, and retrieved using semantic similarity.
- **Web Search** — Automatically detects when a query needs live information, searches DuckDuckGo, scrapes and chunks the pages, then retrieves only the relevant parts using HNSW graph search.
- **Document Reader** — Load PDFs or TXT files into the session. Documents are chunked, embedded, and stored in NPZ files for fast retrieval. Supports optional summarization.
- **Voice Interface** — Wake word detection ("Alexa") using OpenWakeWord, silence-aware recording via WebRTC VAD, speech-to-text via Faster Whisper (GPU), and text-to-speech via Edge TTS (Japanese Nanami voice, played with ffplay).
- **App Launcher** — Yuzu can open applications or websites from a user request. App names are fuzzy-matched against everything on your `$PATH`.
- **Query Router** — Every message is passed through an NLP routing layer that extracts topics, decides whether to search, and identifies apps to launch — all in a single structured JSON call.
- **Local Embedding** — All embeddings use a local Gemma3 300M model running in LM Studio (`text-embedding-embeddinggemma-300m`). Nothing is sent to an external embedding API.
- **Intent Classifier** — A locally fine-tuned transformer model classifies message intent before routing.
- **Custom HNSW** — A from-scratch approximate nearest-neighbor graph (Hierarchical Navigable Small World) written in pure Python/NumPy. Used for doc retrieval, web chunk retrieval, and RAG group lookup.

---

## Architecture
---
<img width="803" height="1258" alt="Main" src="https://github.com/user-attachments/assets/d5c626a4-d62d-4d93-9573-a4ccd74ed6c9" />
---

```
User Input (text or voice)
        │
        ▼
   Query Router
  ┌─────────────────────────────┐
  │  Topic Extraction           │
  │  Web Search Decision        │
  │  App Launch Detection       │
  └──────┬──────────────────────┘
         │
    ┌────┴────┐
    │         │
Web Search   Skip
    │
  HNSW chunk retrieval (temp, not saved)
         │
         ▼
    RAG Retrieval
  ┌──────────────────────────────────────┐
  │  Master Table (SQLite)               │
  │    └─ Sub-tables (per topic group)   │
  │         └─ Exchanges + Embeddings    │
  │  Mean embedding per group            │
  │  HNSW graph per group (saved)        │
  └──────────────────────────────────────┘
         │
    Doc Retrieval (if doc loaded)
  ┌──────────────────────────────────────┐
  │  NPZ file per doc                    │
  │  Chunks + keyword-merged embeddings  │
  │  HNSW graph (saved)                  │
  └──────────────────────────────────────┘
         │
         ▼
   Build context window
   [system prompt + retrieved memory + facts + web info + doc info + last N turns]
         │
         ▼
   GPT-OSS 120B (Groq) → Streamed reply
         │
         ▼
   TTS (Edge TTS → ffplay) [optional]
```

---

## RAG Design
---
<img width="1180" height="1352" alt="Rag" src="https://github.com/user-attachments/assets/3679799e-ce46-4980-a388-f3f37c32dace" />
---
The RAG system avoids summarization intentionally — summarizing exchanges loses detail and nuance, so raw exchanges are stored and retrieved directly.

**Storage structure:**

- `master_table` — one row per topic group. Stores the group's mean embedding (updated incrementally as new exchanges are added) and a list of topics.
- Sub-tables — one per group, named by UUID. Each row is a single exchange (user + assistant turn) with its embedding and topics.

**Retrieval flow:**

1. Embed the current conversation window (last N turns + extracted topics).
2. Cosine-compare against all group mean embeddings in `master_table` → pick top-k groups above threshold.
3. Inside each matched group, run HNSW graph search to retrieve the most relevant individual exchanges.
4. Return raw exchanges + user facts (stored separately in `facts.json`).



**Saving flow:**

Every `n` exchanges, the conversation history is flushed. An LLM call extracts facts and topics per exchange. Each exchange is embedded, compared against existing groups, and either added to a matching group or used to create a new one. The group's mean embedding and HNSW graph are updated.

---

## HNSW

Custom implementation in `HNSW.py`. No libraries used beyond NumPy and `heapq`.

- Builds a neighbor graph for a set of embeddings using cosine similarity.
- Each node connects to its top-8 nearest neighbors.
- The entry point (center node) is selected as the node closest to the mean embedding.
- Graph traversal uses a min-heap priority queue, visiting neighbors greedily by similarity.
- Graphs are serialized with `pickle` and saved/loaded per named collection (one per doc, one per RAG group).

---

## Document Pipeline
---
<img width="580" height="553" alt="docs" src="https://github.com/user-attachments/assets/d4c93080-396b-47e0-8f18-6680d4ac078f" />
---
1. User selects a PDF or TXT file via a file dialog.
2. Text is extracted (pdfplumber for PDFs) and cleaned.
3. Text is split into overlapping sentence-based chunks (1700–2000 chars, 2-sentence overlap).
4. Each chunk is embedded with keywords merged into the embedding input for better retrieval signal.
5. Optionally, chunks are passed through a local model to generate summaries — user's choice.
6. Embeddings, keywords, and mean embedding are saved to a `.npz` file.
7. An HNSW graph is built and saved for the document.

At query time, the user's message is embedded and the HNSW graph is used to retrieve the most relevant chunks.

---

## Web Search Pipeline
---
<img width="543" height="1181" alt="Web Search" src="https://github.com/user-attachments/assets/e39f254f-fac9-4ec2-a8ca-327bb0d337c0" />

---
1. Router extracts 1–3 optimized search queries from the conversation.
2. DuckDuckGo (`ddgs`) fetches top 2 results per query.
3. Pages are scraped with `trafilatura`. Falls back to snippet if scraping fails.
4. Scraped text is chunked with the same chunker used for documents.
5. Chunks are embedded and a temporary HNSW graph is built (not saved).
6. Only chunks above the similarity threshold are kept and passed into context.

---

## Voice Pipeline

| Stage | Tool |
|---|---|
| Wake word | OpenWakeWord (`alexa` model) |
| Recording | `sounddevice` + WebRTC VAD (aggressiveness 2) |
| Silence detection | 2 seconds of non-speech frames |
| Transcription | Faster Whisper (`base`, CUDA) |
| TTS | Edge TTS (`ja-JP-NanamiNeural`) |
| Playback | `ffplay` (subprocess, no display) |

---

## Models

| Role | Model | Where |
|---|---|---|
| Chat + Routing | `openai/gpt-oss-120b` | Groq API |
| Embeddings | `text-embedding-embeddinggemma-300m` | LM Studio (local) |
| Doc summaries | `nvidia/nemotron-3-nano-4b` | LM Studio (local) |
| Intent classifier | Fine-tuned transformer | Local (`/Model/intent_clf`) |
| STT | Faster Whisper `base` | Local (GPU) |
| Wake word | OpenWakeWord `alexa` | Local |
| TTS | Edge TTS `ja-JP-NanamiNeural` | Edge (streaming) |

---

## Project Structure

```
Yuzu-Ai-Companion/
├── main.py              # Entry point, CLI loop, context assembly
├── router.py            # Query routing, web search, app launching
├── rag_system.py        # Long-term memory (hierarchical RAG)
├── doc_reader.py        # Document loading, chunking, retrieval
├── HNSW.py              # Custom HNSW graph (built from scratch)
├── loader.py            # Shared clients (Groq, LM Studio), spaCy, KeyBERT
├── STT.py               # Wake word, VAD, recording, Whisper transcription
├── TTS.py               # Edge TTS + ffplay playback
├── retrieving_clf.py    # Intent classifier inference
│
├── Data/
│   ├── chat_data/
│   │   └── complex_rag.db       # SQLite RAG database
│   ├── docs/
│   │   ├── facts.json           # Persistent user facts
│   │   ├── doc_ref.json         # Loaded document registry
│   │   └── npz_files/           # Per-document embeddings
│   ├── Graph/
│   │   └── hnsw_data.pickle     # Saved HNSW graphs
│   └── Config/
│       └── config_json.json     # Thresholds and settings
│
└── Model/
    ├── intent_clf/              # Fine-tuned intent classifier
    └── intent_tokenizer/
```

---

## Setup

**Requirements:** Python 3.10+, CUDA GPU (for Faster Whisper), LM Studio running locally on port 1234.

```bash
Requirements Are In requirements.txt file 
```

Set your Groq API key in a `.env` file:
```
groq=your_groq_api_key_here
```

Load the embedding model in LM Studio and start the local server on `http://127.0.0.1:1234`.

**Run:**
```bash
python main.py
```

On startup, choose `1` for text input or `0` for voice (wake word: **"Alexa"**).

**In-session commands:**

| Input | Action |
|---|---|
| `q` | Save conversation to RAG and quit |
| `n` | Load a document into the session |
| `r` | Unload a loaded document |

---

## Config

`Data/Config/config_json.json` controls retrieval thresholds:

```json
{
  "web_search": 0.75,
  "master_tabel_threshold": 0.6,
  "within_tabel": 0.65
}
```

Tune these to control how aggressively memory and web chunks are retrieved.

---

## Notes

- Conversation history is kept in memory for the session. Every `n=20` exchanges, older turns are flushed to the RAG database automatically.
- Only the last 5 exchanges are sent to the router for topic/search extraction to keep routing fast and focused.
- Web search results are capped at 4000 characters before being passed to context.
- The HNSW graphs saved to disk are rebuilt in-place when new embeddings are added — the entire graph for that collection is recomputed.
- TTS is disabled by default (`make_sound` is commented out in `main.py`). Uncomment to enable.
