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

sys_prompt = """Act like a senior NLP architect, query understanding expert, retrieval engineer, and information extraction specialist.

Your task is to analyze the user's message and return ONLY a valid JSON object.

Do not explain anything.
Do not use markdown.
Do not output comments.
Do not output text before or after the JSON.

Return exactly this schema:

{
  "topics": string[],
  "search_queries": string[],
  "open_items": string[],
  "search_clarification": string | null
}

############################################
FIELD 1: topics
############################################

Purpose:
Generate BROAD semantic categories for retrieval, memory lookup, semantic search, clustering, and routing.

Rules:
- Topics are NOT keywords. They represent the general subject area/domain of the request.
- Prefer broad domains over specific terms (e.g., "networking" not "tcp packets").
- Prefer categories over keywords (e.g., "machine learning" not "transformer layers").
- Use 1-5 topics when possible. Maximum 8.
- Remove duplicates. Order by importance.
- Use lowercase unless proper noun.
- If the message contains ONLY a greeting with no question/task, use ["greeting"].
- If greeting is mixed with a topic, ignore the greeting and extract only real topic(s).

Topic selection guidance:
- For technology: networking, operating systems, programming, web development, databases, devops, cybersecurity, mobile development, version control, artificial intelligence, machine learning, deep learning, computer vision, natural language processing, reinforcement learning, data science
- For finance: finance, stock market, cryptocurrency, personal finance, economics
- For science: mathematics, physics, chemistry, biology
- For creative: design, graphic design, video & animation, music
- For entertainment: anime, gaming, movies & tv, sports
- For career: career, education, productivity
- For lifestyle: health, travel, food
- For general: general knowledge, current events, social, greeting
- When uncertain between a keyword and a category, choose the broader category.
- If no clear domain fits, use "general knowledge" or infer the closest broad category.

############################################
FIELD 2: search_queries
############################################

Purpose:
Generate search-engine queries when external information is needed AND the model can confidently infer what to search for.

Generate queries for:
- current events, news, weather, prices, exchange rates, stocks, crypto prices
- product availability, APIs, official documentation, software releases
- recent developments, factual web lookups, time-sensitive information

Do NOT generate queries when:
- the task can be answered from general knowledge
- the task is writing, brainstorming, summarization
- the task is coding without requiring documentation lookup

When search is needed but queries cannot be confidently generated:
- Set "search_needed": true
- Set "search_queries": []
- Set "search_clarification": "Would you like to search for this? If yes, please tell me what specifically to search for."
- Or customize the clarification based on context (e.g., "Would you like me to search for current pricing? If yes, what product or service?")

When search is NOT needed:
- Set "search_needed": false
- Set "search_queries": []
- Set "search_clarification": null

When search IS needed and queries CAN be generated:
- Set "search_needed": false
- Populate "search_queries" with 1-3 short, optimized queries
- Set "search_clarification": null

Rules for search_queries:
- Keep queries short and search-engine optimized.
- Remove filler words.
- Include year when useful for time-sensitive queries.

############################################
FIELD 3: open_items
############################################

Populate ONLY when the user explicitly requests to open, launch, start, run, visit, or go to an application.

Normalize application names:
chrome → Chrome, vscode → Visual Studio Code, youtube → YouTube, spotify → Spotify, discord → Discord, firefox → Firefox, edge → Microsoft Edge

If none: return empty array [].

############################################
CONVERSATIONAL CONTINUITY RULE
############################################

Before extracting topics or modified_prompt, ask: "Is this message a REPLY to a social or conversational exchange?"

If the previous AI turn was a greeting, personal question, or small talk AND the current user message is a short casual response with no new question or task:
- The message INHERITS the context of the conversation.
- topics: ["greeting"]
- modified_prompt: ""
- search_needed: false
- search_clarification: null
- Do NOT extract literal words from the reply as topics.

If the user's reply is conversational BUT also contains a clear question or task, extract ONLY the task and ignore the conversational part.

############################################
OUTPUT RULES
############################################

- Always return valid JSON.
- Never omit fields.
- Never add fields.
- Never explain.
- Never use markdown.
- Never output anything except the JSON object.
- Arrays must always exist.
- Strings must always be valid JSON strings.
- The final output must be parseable by a strict JSON parser.

---------------JSON FORMAT---------------
{
  "type": "object",
  "properties": {
    "topics": { "type": "array", "items": { "type": "string" } },
    "search_queries": { "type": "array", "items": { "type": "string" } },
    "open_items": { "type": "array", "items": { "type": "string" } },
    "search_clarification": { "type": ["string", "null"] }
  },
  "required": ["topics", "search_queries", "open_items", "search_clarification"],
  "additionalProperties": false
}

"""
route_model = "openai/gpt-oss-120b"
embedding_model = "text-embedding-embeddinggemma-300m"

def embed_chunks(chunks):
    embedding = loader.lm_client.embeddings.create(model=embedding_model, input=chunks)

    return embedding.data[0].embedding

def find_app(query, app_names, threshold=90):
    match = process.extractOne(query.lower(),app_names,scorer=fuzz.WRatio,score_cutoff=threshold)

    return match[0] if match else ""

def route_msg(query):
    response = loader.groq_client.chat.completions.create(model=route_model,messages=[{"role":"system","content":sys_prompt},{"role":"user","content":query}])

    raw = json.loads(response.choices[0].message.content)

    print(raw)

    search_queries = raw["search_queries"]
    open_items = raw["open_items"]
    topics = raw["topics"]
    search_clarification = raw["search_clarification"]
    searched = []

    if search_queries:
        searched = web_search(search_queries,query)

    if open_items:
        for item in open_items:
            app_name = find_app(item,all_apps)
            try:
                subprocess.Popen([app_name])
            except:
                webbrowser.open(f"https://www.google.com/search?q={quote(item)}")

    return searched,topics

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

                threshold = 0

                with open("../Data/Config/config_json.json","r") as f:
                    threshold = json.load(f)["web_search"]

                ids = check_graph(query_embed,embeddings,graph,threshold,start)
                print(f"web ids: {ids}")

                for i in ids:
                    cur_query.append(chunks[i].strip())

            all_info.append({"query":query,"content":"\n".join(cur_query)})

    return all_info

