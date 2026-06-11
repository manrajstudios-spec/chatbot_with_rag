import os
import json
import loader
import webbrowser
import subprocess
import trafilatura
import numpy as np
from ddgs import DDGS
from urllib.parse import quote
from doc_reader import make_chunks
from rapidfuzz import process, fuzz
from HNSW import make_graph,check_graph

all_apps = set()

for directory in os.environ["PATH"].split(":"):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            all_apps.add(file)

sys_prompt = '''
You are a prompt parser.

Analyze the user's message and respond ONLY with a valid JSON object.
No explanations, markdown, or extra text.

Output exactly these fields:
{
  "modified_prompt": string,
  "topics": string[],
  "search_queries": string[],
  "open_items": string[]
}

----------------------------
FIELD RULES
----------------------------

1. modified_prompt

- Rewrite the user's request into a clear, complete instruction.
- Preserve intent and meaning.
- Remove ONLY opening/launching actions (apps, websites, files, folders).
- Everything else MUST be preserved and rewritten.

IMPORTANT RULE:
- Set "modified_prompt" to "" ONLY IF the user request is purely about opening/launching something AND contains NO other task, question, or intent.

If ANY non-opening intent exists (question, search, explanation, task):
→ You MUST generate a modified_prompt (never empty)

----------------------------

2. topics

- Extract core semantic topics for retrieval/search/memory
- Use lowercase unless proper noun
- 1–4 words per topic
- Merge similar concepts
- No filler words

----------------------------

3. search_queries

Generate ONLY if external info is needed:
- current info, news, prices
- tutorials, APIs, docs
- real-world factual lookup
- trending or time-sensitive data

Otherwise return []

Keep queries short and search-engine optimized.
Add year when useful (2026).

----------------------------

4. open_items

Populate ONLY if user explicitly requests:
open / launch / start / run / go to / visit

- Normalize names:
  vs code → Visual Studio Code
  yt → YouTube
  chrome → Chrome
  github → GitHub
  spotify → Spotify

If none → []

----------------------------

HARD RULES
- Always return valid JSON
- Never omit any field
- Never explain anything
- Never add extra fields
- Never return invalid JSON

----------------------------

EXAMPLES

User: "open chrome and tell me how tcp works"

{
"modified_prompt": "Explain how TCP works.",
"topics": ["tcp", "networking"],
"search_queries": [],
"open_items": ["Chrome"]
}

User: "open chrome"

{
"modified_prompt": "",
"topics": ["chrome"],
"search_queries": [],
"open_items": ["Chrome"]
}

User: "how does transformers work"

{
"modified_prompt": "Explain how transformers work.",
"topics": ["transformers", "machine learning"],
"search_queries": [],
"open_items": []
}
'''
route_model = "nvidia/nemotron-3-nano-4b"
embedding_model = "text-embedding-embeddinggemma-300m"

def embed_chunks(chunks):
    embedding = loader.lm_client.embeddings.create(model=embedding_model, input=chunks)

    return embedding.data[0].embedding

def find_app(query, app_names, threshold=90):
    match = process.extractOne(query.lower(),app_names,scorer=fuzz.WRatio,score_cutoff=threshold)

    return match[0] if match else ""

def route_msg(query):
    response = loader.lm_client.chat.completions.create(model=route_model,messages=[{"role":"system","content":sys_prompt},{"role":"user","content":query}])

    raw = json.loads(response.choices[0].message.content)

    print(raw)

    to_ask = raw["modified_prompt"]
    search_queries = raw["search_queries"]
    open_items = raw["open_items"]
    topics = raw["topics"]
    searched = []

    if search_queries:
        searched = web_search(search_queries,to_ask)

    if open_items:
        for item in open_items:
            app_name = find_app(item,all_apps)
            try:
                subprocess.Popen([app_name])
            except:
                webbrowser.open(f"https://www.google.com/search?q={quote(item)}")

    if not to_ask and not open_items:
        to_ask = query
    return to_ask,searched,topics

def web_search(queries,to_ask):
    all_info = []
    query_embed = embed_chunks(to_ask)
    query_embed = np.array(query_embed, dtype=np.float32)

    for query in queries:
        with DDGS() as search:
            hits = list(search.text(query,max_results=2))

            cur_query = []

            for hit in hits:
                html = trafilatura.fetch_url(hit["href"])
                if not html:
                    all_info.append({"query":query,"content":hit["body"]})
                    continue

                content = trafilatura.extract(html) or hit["body"]

                if not content:
                    continue

                chunks = make_chunks(content)

                if not chunks:
                    continue

                embeddings = []

                for chunk in chunks:
                    embeddings.append(embed_chunks(chunk.strip()))

                embeddings = np.array(embeddings,dtype=np.float32)

                graph,start = make_graph(embeddings,"abc",False)

                ids = check_graph(query_embed,embeddings,graph,0.35,start)

                for i in ids:
                    cur_query.append(chunks[i].strip())
                    print(i)

            all_info.append({"query":query,"content":"\n".join(cur_query)})

    return all_info
