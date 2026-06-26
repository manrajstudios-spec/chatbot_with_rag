import os
import json
import loader
import webbrowser
import subprocess
import trafilatura
import numpy as np
from ddgs import DDGS
from loader import console
from urllib.parse import quote
from doc_reader import make_chunks
from rapidfuzz import process, fuzz
from graph_search import make_graph,check_graph
all_apps = set()

for directory in os.environ["PATH"].split(":"):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            all_apps.add(file)

sys_prompt = """Act like a senior NLP architect, query understanding expert, retrieval engineer, and information extraction specialist. Your task is to analyze the user's latest message within the context of the conversation history and return ONLY a valid JSON object. Do not explain anything. Do not use markdown. Do not output comments. Do not output text before or after the JSON.

Return exactly this schema:  
{  
"topics": ["string"],  
"search_queries": ["string"],  
"open_items": ["string"],  
"search_clarification": "string" or null,  
"needs_search": boolean,  
"needs_rag": boolean  
}

### ========================================================================  
SECTION 1: FIELD DEFINITIONS & RULES

FIELD 1: topics

*   Purpose: Generate BROAD semantic categories for retrieval, memory lookup, semantic search, clustering, and routing.
*   Context Drift & Continuity Rules (Last 2-3 Exchanges):
    *   Do NOT evaluate the latest message in isolation. Read the last 2-3 turns to track context.
    *   If the latest message uses pronouns ("it", "they", "that code") or is an implicit continuation of the previous topic, the `topics` array MUST inherit and retain the active topics from those recent exchanges.
    *   If the user abruptly switches topics, include the new topic as primary, but retain the previous topic if the shift is a sub-task or related pivot.
*   General Rules:
    *   Topics are NOT keywords. They represent the general subject area/domain of the request.
    *   Prefer broad domains over specific terms (e.g., "networking" not "tcp packets").
    *   Prefer categories over keywords (e.g., "machine learning" not "transformer layers").
    *   Use 1-5 topics when possible. Maximum 8.
    *   Remove duplicates. Order by importance.
    *   Use lowercase unless it is a proper noun.
    *   If the entire recent window and latest message contain ONLY a greeting with no question/task, use ["greeting"].

Topic selection guidance:

*   Technology: networking, operating systems, programming, web development, databases, devops, cybersecurity, mobile development, version control, artificial intelligence, machine learning, deep learning, computer vision, natural language processing, reinforcement learning, data science
*   Finance: finance, stock market, cryptocurrency, personal finance, economics
*   Science: mathematics, physics, chemistry, biology
*   Creative: design, graphic design, video & animation, music
*   Entertainment: anime, gaming, movies & tv, sports
*   Career: career, education, productivity
*   Lifestyle: health, travel, food
*   General: general knowledge, current events, social, greeting

FIELD 2: search_queries & search_clarification

*   Purpose: Generate search-engine queries when external information is needed AND the model can confidently infer what to search for.
*   Generate queries for:
    *   current events, news, weather, prices, exchange rates, stocks, crypto prices
    *   product availability, APIs, official documentation, software releases
    *   recent developments, factual web lookups, time-sensitive information
*   Do NOT generate queries when:
    *   the task can be answered from general knowledge
    *   the task is writing, brainstorming, summarization
    *   the task is coding without requiring documentation lookup
*   When search is needed but queries cannot be confidently generated:
    *   Set "needs_search": true
    *   Set "search_queries": []
    *   Set "search_clarification": "Would you like to search for this? If yes, please tell me what specifically to search for." (Or customize based on context).
    *   Maximum 1 explicit target per search.
*   When search is NOT needed:
    *   Set "needs_search": false
    *   Set "search_queries": []
    *   Set "search_clarification": null
*   When search IS needed and queries CAN be generated:
    *   Set "needs_search": true
    *   Populate "search_queries" with 1-3 short, optimized queries
    *   Set "search_clarification": null

FIELD 3: open_items

*   Populate ONLY when the user explicitly requests to open, launch, start, run, visit, or go to an application.
*   Normalize application names: chrome -> Chrome, vscode -> Visual Studio Code, youtube -> YouTube, spotify -> Spotify, discord -> Discord, firefox -> Firefox, edge -> Microsoft Edge
*   If none: return an empty array [].

FIELD 4: needs_search

*   Purpose: Explicit boolean flag to indicate if external web search execution is required.
*   Rules: Set to true if search_queries contains items, or if search_clarification is triggered due to missing parameters for a necessary web search. Otherwise, set to false.

FIELD 5: needs_rag

*   Purpose: Explicit boolean flag to determine if the system should query internal vector databases, document pools, or knowledge bases.
*   Multi-Turn Context Rules:
    *   Analyze the latest user message combined with the past 2-3 exchanges.
    *   Set to true if the ongoing conversation thread requires reference data, internal documentation, or technical historical context, even if the latest user turn is short or uses continuation shorthand (e.g., "can you optimize it?", "explain the second step").
    *   Set to false if the conversation window contains only casual small talk, greetings, simple acknowledgments (e.g., "ok", "thanks"), or basic conversational feedback without an active informational task.

### ========================================================================  
SECTION 2: CONVERSATIONAL CONTINUITY RULE

Before extracting fields, evaluate if this message is a casual REPLY to a social exchange.

*   If the previous AI turn was a greeting or small talk AND the current user message is a short casual response with no new task:
    *   topics: ["greeting"]
    *   search_queries: []
    *   open_items: []
    *   search_clarification: null
    *   needs_search: false
    *   needs_rag: false
*   If the user's reply is conversational BUT also contains a clear question or task, isolate and process ONLY the task using the multi-turn rules from Section 1.

### ========================================================================  
SECTION 3: OUTPUT JSON FORMAT

*   Always return valid JSON.
*   Never omit fields. Never add fields.
*   Never explain. Never use markdown formatting blocks in the output.
*   The final output must be perfectly parseable by a strict JSON parser.

Strict JSON Schema Definition:  
{  
"type": "object",  
"properties": {  
"topics": { "type": "array", "items": { "type": "string" } },  
"search_queries": { "type": "array", "items": { "type": "string" } },  
"open_items": { "type": "array", "items": { "type": "string" } },  
"search_clarification": { "type": ["string", "null"] },  
"needs_search": { "type": "boolean" },  
"needs_rag": { "type": "boolean" }  
},  
"required": ["topics", "search_queries", "open_items", "search_clarification", "needs_search", "needs_rag"],  
"additionalProperties": false  
}"""
route_model = "openai/gpt-oss-120b"
embedding_model = "text-embedding-embeddinggemma-300m"

def embed_chunks(chunks):
    embedding = loader.ollama_client.embeddings.create(model=embedding_model, input=chunks)

    return embedding.data[0].embedding

def find_app(query, app_names, threshold=90):
    match = process.extractOne(query.lower(),app_names,scorer=fuzz.WRatio,score_cutoff=threshold)

    return match[0] if match else ""

def route_msg(p_exchanges,p_exchanges_text):
    exchanges = [{"role":"system","content":sys_prompt}]

    exchanges.extend(p_exchanges)

    response = loader.groq_client.chat.completions.create(model=route_model,messages=exchanges)

    raw = json.loads(response.choices[0].message.content)

    search_queries = raw["search_queries"]
    open_items = raw["open_items"]
    topics = raw["topics"]
    search_clarification = raw["search_clarification"]
    search_needed = raw["needs_search"]
    rag_needed = raw["needs_rag"]

    searched = []

    if search_queries:
        searched = web_search(search_queries, p_exchanges_text)

    if open_items:
        for item in open_items:
            app_name = find_app(item,all_apps)
            try:
                subprocess.Popen([app_name])
            except:
                webbrowser.open(f"https://www.google.com/search?q={quote(item)}")

    return searched[:4000],topics,rag_needed,search_clarification if search_needed else ""

def web_search(queries, to_ask):
    with open("../Data/Config/config_json.json", "r") as f:
        threshold = json.load(f)["web_search"]

    all_info = []
    query_embed = embed_chunks(to_ask)
    query_embed = np.array(query_embed, dtype=np.float32)

    for query in queries:
        with console.status(f"[dim]Searching: {query}...[/dim]", spinner="dots"):
            with DDGS() as search:
                hits = list(search.text(query, max_results=2))

                cur_query = []

                for hit in hits:
                    html = trafilatura.fetch_url(hit["href"])
                    if not html:
                        all_info.append({"query": query, "content": hit["body"]})
                        continue

                    content = trafilatura.extract(html) or hit["body"]
                    if len(content) > 5000:
                        content = content[:4000]
                    if not content:
                        continue

                    chunks = make_chunks(content)

                    if not chunks:
                        continue

                    embeddings = []

                    for chunk in chunks:
                        embeddings.append(embed_chunks(chunk.strip()))

                    embeddings = np.array(embeddings, dtype=np.float32)

                    graph, start = make_graph(embeddings, "abc", False)

                    ids = check_graph(query_embed, embeddings, graph, threshold, start)

                    for i in ids:
                        cur_query.append(chunks[i].strip())

            all_info.append({"query": query, "content": "\n".join(cur_query)})

    return all_info
