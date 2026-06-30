import os
import json
import webbrowser
import subprocess
import trafilatura
import numpy as np
from ddgs import DDGS
from urllib.parse import quote
from doc_reader import make_chunks
from rapidfuzz import process, fuzz
from graph_search import make_graph,check_graph
from loader import console,get_embedding,groq_client,main_model

all_apps = set()

for directory in os.environ["PATH"].split(":"):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            all_apps.add(file)

sys_prompt = """You are a query router and conversation understanding model.

Analyze the latest user message using recent conversation history.

Return ONLY valid JSON.
No markdown.
No explanations.
No text before or after JSON.

OUTPUT SCHEMA:
{
"modified_query": string or null,
"topics": string[],
"search_queries": string[],
"open_items": string[],
"search_clarification": string or null,
"needs_search": boolean,
"needs_rag": boolean
}

GENERAL RULES:

* Always output exactly the fields above.
* Do not add extra fields.
* Use valid JSON only.
* search_queries must contain maximum 2 items.
* If no value exists, use null or [].
* Analyze the latest user message as the main request.
* Use previous exchanges only for context, pronoun resolution, and continuity.

FIELD: modified_query

* If the latest message is greeting, filler, thanks, or small talk with no task, return null.
* If clear, rewrite it as one concise semantic sentence.
* If vague or uses pronouns like "it", "that", "this", resolve using recent history.
* If user asks only to open an app/site, return null.
* If user asks to open something plus another task, remove the open action and keep the task.
* Keep it optimized for embedding/search.
* Prefer: "user asks..." or "user wants..."

FIELD: topics
 * Purpose: Generate BROAD semantic categories for retrieval, memory lookup, semantic search, clustering, and routing. 
 * Context Drift & Continuity Rules (Last 2-3 Exchanges): 
 * Do NOT evaluate the latest message in isolation. Read the last 2-3 turns to track context. 
 * If the latest message uses pronouns ("it", "they", "that code") or is an implicit continuation of the previous topic, the topics array MUST inherit and retain the active topics from those recent exchanges. 
 * If the user abruptly switches topics, include the new topic as primary, but retain the previous topic if the shift is a sub-task or related pivot. 
 * General Rules: 
 * Topics are NOT keywords. They represent the general subject area/domain of the request. 
 * Prefer broad domains over specific terms (e.g., "networking" not "tcp packets"). 
 * Prefer categories over keywords (e.g., "machine learning" not "transformer layers"). 
 * Use 1-5 topics when possible. Maximum 8. * Remove duplicates. Order by importance. 
 * Use lowercase unless it is a proper noun. 
 * If the entire recent window and latest message contain ONLY a greeting with no question/task, use ["greeting"]. 
 
 Topic selection guidance: 
 * Technology: networking, operating systems, programming, web development, databases, devops, cybersecurity, mobile development, version control, artificial intelligence, machine learning, deep learning, computer vision, natural language processing, reinforcement learning, data science * Finance: finance, stock market, cryptocurrency, personal finance, economics * Science: mathematics, physics, chemistry, biology * Creative: design, graphic design, video & animation, music * Entertainment: anime, gaming, movies & tv, sports * Career: career, education, productivity * Lifestyle: health, travel, food * General: general knowledge, current events, social, greeting

FIELD: search_queries
Generate search queries ONLY when external web search is actually needed.

Search is needed for:

* latest/current/recent/live information
* news, weather, prices, stocks, crypto, exchange rates,media or any topic 
* product availability
* schedules, deadlines, laws, regulations
* APIs, libraries, docs, software/model releases
* specific companies, people, products, services, GitHub issues, fellowships, competitions, jobs
* when user explicitly says search, browse, look up, check online, or verify

Search is NOT needed for:

* general knowledge
* explanations
* writing/rewriting
* brainstorming
* coding/debugging with enough provided context
* casual chat
* tasks answerable from recent conversation or RAG

Rules:

* search_queries must be [] when search is not needed.
* Generate 1 or 2 queries only.
* Never generate more than 2 queries.
* Prefer 1 strong query.
* Queries should be short and specific.
* Do not duplicate queries.

FIELD: search_clarification

* If search is not needed: null.
* If search is needed and queries are clear: null.
* If user wants search but target is unclear: "Please specify what you want me to search for."

FIELD: open_items

* Populate only when user explicitly asks to open, launch, start, run, visit, or go to something.
* Normalize names:
  chrome -> Chrome
  vscode / vs code -> Visual Studio Code
  youtube -> YouTube
  spotify -> Spotify
  discord -> Discord
  firefox -> Firefox
  edge -> Microsoft Edge
  github -> GitHub
  gmail -> Gmail
  pycharm -> PyCharm
* If none, return [].

FIELD: needs_search

* true if search_queries is not empty.
* true if search_clarification is not null.
* otherwise false.

FIELD: needs_rag
Set true when internal memory/RAG/documents would help:

* user refers to previous conversation, saved memory, uploaded docs, old code, projects, repo, preferences, plans, or personal history
* user says things like "my project", "that code", "continue from there", "what we discussed"
* latest message needs historical/personal context beyond recent turns

Set false when:

* greeting/small talk/thanks
* simple general knowledge
* web search only
* open app/site only
* current message has enough context

FINAL CHECK:
Before output, ensure:

* valid JSON
* exactly 7 fields
* search_queries has max 2 items
* needs_search is true only when search_queries has items or search_clarification is not null
"""
route_model = "openai/gpt-oss-120b"
embedding_model = "text-embedding-embeddinggemma-300m"

def find_app(query, app_names, threshold=90):
    match = process.extractOne(query.lower(),app_names,scorer=fuzz.WRatio,score_cutoff=threshold)

    return match[0] if match else ""

def get_response(previous_exchanges,query):
    to_give = [{"role": "system", "content": sys_prompt}]

    for exchange in previous_exchanges:
        to_give.append({"role":"user","content":exchange["user"]})
        to_give.append({"role": "assistant", "content": exchange["assistant"]})

    to_give.append({"role":"user","content":query})

    response = groq_client.chat.completions.create(model=main_model,messages=to_give)

    return json.loads(response.choices[0].message.content.strip())

def route_msg(previous_exchanges, user_query,previous_exchanges_text):
    parsed = get_response(previous_exchanges, user_query)

    search_clarification = parsed["search_clarification"]
    modified_query = parsed["modified_query"]
    search_queries = parsed["search_queries"]
    search_needed = parsed["needs_search"]
    open_items = parsed["open_items"]
    rag_needed = parsed["needs_rag"]
    topics = parsed["topics"]

    searched = []

    console.print(f"Queries{search_queries}")

    if modified_query: previous_exchanges_text += modified_query
    else: previous_exchanges_text += user_query

    if search_queries:
        searched = web_search(search_queries, previous_exchanges_text)
        console.print(f"Searched{len(searched)}")
    if open_items:
        for item in open_items:
            app_name = find_app(item,all_apps)

            try:
                subprocess.Popen([app_name])
            except FileNotFoundError as e:
                webbrowser.open(f"https://www.google.com/search?q={quote(item)}")
                print(f"APP ERROR: {e}")

    return modified_query,rag_needed,search_needed,search_clarification,topics,searched

def web_search(queries, to_ask):
    all_info = []
    query_embed = np.array(get_embedding([to_ask])[0],dtype=np.float32)
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

                    if len(content) > 5000:content = content[:5000]

                    if not content:continue

                    chunks = make_chunks(content)

                    chunk_embeddings = []

                    for chunk in chunks:
                        chunk_embeddings.append(np.array(get_embedding([chunk.strip()])[0],dtype=np.float32))

                    chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)

                    graph, start,all_embeds_norm = make_graph(chunk_embeddings, "abc", False)

                    ids = check_graph(query_embed=query_embed, all_embeds=all_embeds_norm, graph=graph,center_node=start)

                    console.print(f"Ids: {ids}")
                    for i in ids:
                        cur_query.append(chunks[i].strip())

            all_info.append({"query": query, "content": "\n".join(cur_query)})

    return all_info
