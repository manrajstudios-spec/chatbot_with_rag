# Multi-Agent ChatBot / Yuzu AI Companion

Yuzu is a local-first personal AI companion built from scratch in Python.

She can chat naturally, remember useful past conversations, read documents, search the web, open apps, use voice input, and respond through text or speech. The core systems — routing, memory retrieval, document retrieval, chunking, graph search, and context building — are custom-built without LangChain, LlamaIndex, or any ready-made RAG framework.

This project is mainly built as a standalone local AI assistant and learning project. It focuses on understanding how real assistant pipelines work internally: query routing, tool use, retrieval, memory, web search, document search, speech input, and response generation.

---

## Features

* **Conversational AI** — Uses `openai/gpt-oss-120b` through Groq for natural chat responses.
* **Custom Long-Term Memory RAG** — Stores useful conversation exchanges in SQLite, groups them by semantic similarity, and retrieves them when relevant.
* **Query Router** — Every message is routed through a structured JSON layer that decides whether the assistant needs memory, web search, app opening, or document retrieval.
* **Web Search** — Searches DuckDuckGo, scrapes pages with `trafilatura`, chunks the content, embeds it, and retrieves only the most relevant parts.
* **Document Reader** — Loads PDF/TXT files, extracts text, chunks it, embeds chunks with keyword-boosted text, and retrieves relevant document sections during chat.
* **Custom Graph Retrieval** — Uses a lightweight NumPy-based graph search system for document chunks, web chunks, and retrieval collections.
* **Voice Interface** — Supports wake word detection, speech/silence detection, Faster Whisper transcription, and optional text-to-speech.
* **App Launcher** — Can open local apps or websites from natural language using fuzzy matching over executables in `$PATH`.
* **Local Embeddings** — Uses a local OpenAI-compatible embedding server for embeddings instead of external embedding APIs.
* **CLI Experience** — Uses Rich panels and status spinners to make the terminal assistant feel smoother and easier to debug.

---

## Architecture

---

<img width="803" height="1258" alt="Main" src="https://github.com/user-attachments/assets/d5c626a4-d62d-4d93-9573-a4ccd74ed6c9" />
---

```text
User Input
   │
   ├── Written input
   └── Voice input
        └── Wake word → VAD → Whisper STT
   │
   ▼
Query Router
   │
   ├── Rewrites query if needed
   ├── Extracts topics
   ├── Decides if RAG is needed
   ├── Decides if web search is needed
   ├── Detects app/website opening requests
   └── Produces structured JSON
   │
   ▼
Context Gathering
   │
   ├── Long-term memory retrieval
   ├── Loaded document retrieval
   ├── Web search retrieval
   ├── Recent conversation turns
   └── User facts
   │
   ▼
Prompt Builder
   │
   ├── Yuzu system prompt
   ├── Retrieved memories
   ├── Retrieved document chunks
   ├── Retrieved web chunks
   ├── Recent context
   └── Current user query
   │
   ▼
Groq / GPT-OSS 120B
   │
   ▼
Streamed response
   │
   └── Optional TTS
   │
   ▼
Conversation saved back into memory
```

---

## RAG Design

---

<img width="1180" height="1352" alt="Rag" src="https://github.com/user-attachments/assets/3679799e-ce46-4980-a388-f3f37c32dace" />
---

The memory system stores raw conversation exchanges instead of only summaries. This is intentional because summaries can lose useful details, tone, and context.

The RAG system uses a two-level structure:

### 1. Master Table

The `master_table` stores one row per semantic group.

Each group has:

* a table name
* a mean embedding
* a count of stored exchanges
* related topics

The mean embedding acts like a rough semantic center for that group.

### 2. Group Tables

Each group has its own SQLite table containing individual exchanges.

Each row stores:

* the raw user + assistant exchange
* the exchange embedding
* topics
* date/time metadata

---

## Memory Retrieval Flow

```text
Current query + recent conversation
   │
   ▼
Embedding
   │
   ▼
Compare against master group mean embeddings
   │
   ▼
Select most relevant groups
   │
   ▼
Search inside matched groups
   │
   ▼
Return raw relevant exchanges + user facts
```

The assistant only uses memory when the router decides it is useful. This prevents old memories from being forced into every response.

---

## Memory Saving Flow

Conversation history is kept in the current session. Once the history grows past a limit, older exchanges are flushed into the RAG system.

During saving:

1. An LLM extracts useful exchanges.
2. It removes filler or useless turns.
3. Topics and facts are extracted.
4. Each useful exchange is embedded.
5. The exchange is compared with existing memory groups.
6. It is either added to a matching group or used to create a new group.
7. Group mean embeddings are updated.

---

## Custom Graph Search

The project uses a custom NumPy-based graph retrieval system.

It is HNSW-inspired, but it is intentionally lightweight and built for learning. It is not a full production HNSW implementation.

The graph system:

* normalizes embeddings
* connects each node to its nearest neighbors
* chooses a center node using the mean embedding
* performs greedy traversal through similar nodes
* retrieves relevant nodes without brute-forcing every item every time

This graph retrieval is used for:

* document chunks
* web search chunks
* retrieval collections

The system originally used fixed similarity thresholds, but was upgraded to greedy graph traversal because hard thresholds can miss useful chunks when embedding scores vary.

```text
Old approach:
query → compare all chunks → keep chunks above threshold

Current approach:
query → start from graph center → move through similar neighbors → collect relevant chunks
```

This makes retrieval more adaptive and less dependent on one manually tuned threshold.

---

## Document Pipeline

---

<img width="580" height="553" alt="docs" src="https://github.com/user-attachments/assets/d4c93080-396b-47e0-8f18-6680d4ac078f" />
---

The document reader allows the user to attach PDF or TXT files during a chat session.

Document ingestion flow:

1. User selects a PDF or TXT file through a file picker.
2. Text is extracted using PyMuPDF for PDFs or normal file reading for TXT files.
3. Text is cleaned.
4. Text is split into overlapping sentence-based chunks.
5. Keywords are extracted from each chunk.
6. Chunk text and keywords are merged before embedding.
7. Embeddings are stored.
8. A graph is built over the document chunks.
9. Document metadata and chunks are saved for later retrieval.

At query time:

```text
User query
   │
   ▼
Keyword extraction
   │
   ▼
Query embedding
   │
   ▼
Graph search over loaded document chunks
   │
   ▼
Relevant chunks added to context
```

Document summarization was removed from the main flow because it slowed down ingestion and could lose exact details. Raw chunks are used instead for more faithful document Q&A.

---

## Web Search Pipeline

---

<img width="543" height="1181" alt="Web Search" src="https://github.com/user-attachments/assets/e39f254f-fac9-4ec2-a8ca-327bb0d337c0" />
---

Web search is controlled by the router. The assistant does not search blindly for every message.

Flow:

1. Router decides whether web search is needed.
2. Router creates optimized search queries.
3. DuckDuckGo search is used through `ddgs`.
4. Search results are scraped with `trafilatura`.
5. If scraping fails, the result snippet is used as fallback.
6. Web text is chunked.
7. Chunks are embedded.
8. A temporary graph is built.
9. Relevant chunks are retrieved and passed into the final prompt.

Web search chunks are temporary and are not saved into long-term memory by default.

---

## Voice Pipeline

Voice mode is optional.

| Stage            | Tool           |
| ---------------- | -------------- |
| Wake word        | OpenWakeWord   |
| Audio recording  | `sounddevice`  |
| Speech detection | WebRTC VAD     |
| Transcription    | Faster Whisper |
| Text-to-speech   | Edge TTS       |
| Playback         | `ffplay`       |

Voice flow:

```text
Wake word
   │
   ▼
Start listening
   │
   ▼
Detect speech frames using VAD
   │
   ▼
Record until enough silence is detected
   │
   ▼
Transcribe with Faster Whisper
   │
   ▼
Send text into normal assistant pipeline
```

The assistant can run in written mode without needing to use the voice pipeline.

---

## App Launcher

Yuzu can open apps or websites from user requests.

The app launcher:

1. Extracts requested apps/sites from the router output.
2. Scans executables available in the system `$PATH`.
3. Uses fuzzy matching to find the closest app name.
4. Opens the matched app using `subprocess`.
5. Falls back to web search/opening if needed.

This allows messages like:

```text
open brave
open vscode
search this on web
```

to be handled by the same routing system.

---

## Models and Services

| Role                     | Model / Tool                                             | Location         |
| ------------------------ | -------------------------------------------------------- | ---------------- |
| Chat model               | `openai/gpt-oss-120b`                                    | Groq API         |
| Router model             | `openai/gpt-oss-120b`                                    | Groq API         |
| Memory extraction        | `openai/gpt-oss-120b`                                    | Groq API         |
| Embeddings               | `embeddinggemma:300m` / OpenAI-compatible local endpoint | Local            |
| Speech-to-text           | Faster Whisper `base`                                    | Local            |
| Wake word                | OpenWakeWord                                             | Local            |
| Voice activity detection | WebRTC VAD                                               | Local            |
| Text-to-speech           | Edge TTS                                                 | Online streaming |
| CLI UI                   | Rich                                                     | Local            |

---

## Project Structure

```text
Multi-Agent-ChatBot/
│
├── Data/
│   ├── chat_data/
│   │   └── complex_rag.db
│   ├── docs/
│   │   ├── facts.json
│   │   ├── doc_ref.json
│   │   └── context/
│   ├── Graph/
│   │   └── graph_data.pickle
│   └── Config/
│       └── config_json.json
│
├── Scripts/
│   ├── main_flow.py        # Main CLI loop and context assembly
│   ├── router.py           # Query routing, web search, app launching
│   ├── rag_system.py       # Long-term memory RAG
│   ├── doc_reader.py       # Document loading, chunking, retrieval
│   ├── graph_search.py     # Custom graph-based retrieval
│   ├── loader.py           # Shared clients, embeddings, keyword extraction
│   ├── STT.py              # Wake word, VAD, Whisper transcription
│   ├── TTS.py              # Edge TTS playback
│   ├── retrieving_clf.py   # Intent classifier inference
│   ├── training_clf.py     # Intent classifier training
│   └── make_data.py        # Dataset/helper script
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/manrajstudios-spec/Multi-Agent-ChatBot.git
cd Multi-Agent-ChatBot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

You may also need the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

### 4. Create `.env`

Create a `.env` file in the project root:

```env
groq_api=your_groq_api_key_here
```

### 5. Start the local embedding server

The current code expects an OpenAI-compatible local embedding endpoint.

Default endpoint used in code:

```text
http://localhost:11434/v1
```

Default embedding model:

```text
embeddinggemma:300m
```

Make sure the local embedding server is running before starting the assistant.

### 6. Run the assistant

Because the scripts currently use relative paths like `../Data/...`, run from inside the `Scripts` folder:

```bash
cd Scripts
python main_flow.py
```

On startup, choose:

| Input | Mode         |
| ----- | ------------ |
| `1`   | Written chat |
| `0`   | Voice chat   |

---

## In-Session Commands

| Command | Action                                     |
| ------- | ------------------------------------------ |
| `n`     | Load a document                            |
| `r`     | Unload a document                          |
| `q`     | Save current conversation context and quit |

---

## Configuration

`Data/Config/config_json.json` stores retrieval thresholds and settings.

Example:

```json
{
  "web_search": 0.75,
  "master_tabel_threshold": 0.6,
  "within_tabel": 0.65
}
```

These thresholds control how aggressively memory and web/document chunks are retrieved.

---

## Current Limitations

This is a standalone local project, so some parts are still being improved.

* Some exception handling is still rough.
* Web search can fail when pages block scraping.
* Router output depends on valid JSON from the model.
* Voice mode depends on microphone access, Faster Whisper, PyAudio/sounddevice, and `ffplay`.
* The graph search is custom and lightweight, not a full production HNSW implementation.
* The project is not packaged as an installable Python module yet.
* Some paths are still designed around running from the `Scripts` directory.

---

## Why This Project Exists

This project was built to understand how AI assistants actually work internally.

Instead of using a ready-made agent or RAG framework, the system builds the core pieces manually:

* routing
* long-term memory
* semantic grouping
* document retrieval
* web retrieval
* graph search
* voice input
* context assembly
* response streaming

The goal is not just to make a chatbot, but to learn how a full assistant pipeline connects end to end.
