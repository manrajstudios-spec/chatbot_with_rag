import uuid
import json
import loader
import sqlite3
import tiktoken
import numpy as np
from datetime import datetime
from sentence_transformers import util
from loader import console,get_response,get_embedding
from graph_search import compare_embed,make_graph,add_to_graph

# Fixes Left

# Handle input for rag and keeping good exchnages only

enc = tiktoken.get_encoding("cl100k_base")
summary_model = "openai/gpt-oss-120b"
embeddings_model = "text-embedding-embeddinggemma-300m"

sys_prompt = """You are a conversation memory extractor.

IMPORTANT OUTPUT RULES

- Output ONLY valid JSON. No explanations, no markdown, no extra text.
- Output must be a JSON object with three keys: useful_exchanges, topics, facts.

INPUT FORMAT

You will receive data in the following structure:

{
  "old_facts": ["previously extracted user facts", "..."],
  "exchanges": [
    {
      "index": 0,
      "user": "User message text",
      "assistant": "Assistant response text"
    }
  ]
}

Each exchange is tagged with an index number (0, 1, 2, ...).

OUTPUT FORMAT (STRICT)

{
  "useful_exchanges": [0, 2, 4],
  "topics": [["topic1", "topic2"], ["topic1"], ["topic1", "topic3"]],
  "facts": ["flat list of all facts: old untouched + modified + new"]
}

FIELD RULES

useful_exchanges:
- Include the index of an exchange ONLY if it contains meaningful information worth storing.
- DROP exchanges that are just greetings, small talk, filler, or simple one-off queries with no preference, goal, or user-specific information (e.g. "hi", "thanks", "how are you", "what is X" with no personal context).
- KEEP exchanges that reveal user preferences, goals, dislikes, tools, deadlines, constraints, or anything personal and reusable.

topics:
- STRICT: len(topics) MUST always equal len(useful_exchanges).
- Each entry in topics corresponds to the exchange at the same position in useful_exchanges.
- Each entry is a list of 1-5 broad semantic topic categories for that exchange.
- Topics are broad domains, not keywords. Use lowercase unless proper noun.
- Examples: ["machine learning"], ["programming", "python"], ["career", "education"]

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

facts:
- Compare new exchanges against old_facts.
- If an exchange updates an existing fact, overwrite it.
- If an exchange adds detail to an existing fact, merge them into one precise point.
- If an exchange introduces a brand new fact, append it.
- If an old fact was not touched by any exchange, keep it as-is.
- Output a single flat list of all resulting facts.
- Facts must be short atomic fragments. No paragraphs.
- Only include user-specific facts: preferences, goals, tools, constraints, timelines, numbers.
"""
complex_rag = sqlite3.connect("../Data/chat_data/complex_rag.db")
cursor_complex_rag = complex_rag.cursor()

cursor_complex_rag.execute("CREATE TABLE IF NOT EXISTS master_table("
                           "tables_name TEXT PRIMARY KEY, group_embeddings BLOB,count INT,topics TEXT"
                           ");")

def count_tokens(text):
    return len(enc.encode(text))

def load_facts():
    try:
        with open("../Data/docs/facts.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def write_facts(facts):
    with open("../Data/docs/facts.json", "w") as f:
        json.dump(facts, f,indent=4)

def divide_sections(exchanges):
    chunks = []
    cur_chunk = []

    for exchange in exchanges:
        curr = cur_chunk.copy()
        curr.append(exchange)
        cur_token_len = len(enc.encode(" ".join(curr)))

        if cur_token_len > 6000:
            chunks.append(cur_chunk)
            cur_chunk = [exchange]
        else:
            cur_chunk.append(exchange)

    embeddings = []

    for chunk in chunks:
        embeddings.extend(np.array(get_embedding(chunk),dtype=np.float32))

    embeddings = np.array(embeddings)
    norm_embeddings = embeddings / np.linalg.norm(embeddings,axis=1,keepdims=True)

    groups = util.community_detection(norm_embeddings,min_community_size=1,threshold=0.6)

    grouped_exchanges = [[exchanges[i] for i in group] for group in groups]

    return grouped_exchanges,groups

def rag_query(hist):
    msg = ""
    exchange_count = 0

    for i,m in enumerate(hist):
        if i % 2 == 0:
            msg += f"{exchange_count}: \n"
            msg += f"User: {m} \nASSISTANT: {hist[i+1]}\n"
            exchange_count +=1

    facts_msg = ", ".join(load_facts())

    parsed = get_response(query=msg,sys_prompt=f"{sys_prompt}\nFacts: {facts_msg}")

    exchanges = [f"{hist[i]} \n{hist[i+1]} \n" for i in parsed["useful_exchanges"]]

    sections,group_ids = divide_sections(exchanges)

    topics = parsed["topics"]

    sectioned_topics = [[topics[i] for i in idx] for idx in group_ids]

    return parsed,sections,sectioned_topics

def add_to_rag(hist):
    parsed,sections,sectioned_topics = rag_query(hist)

    facts = parsed["facts"]

    for section,topic in zip(sections,sectioned_topics):
        cur_section_text = "\n".join(section)
        cur_section_text += f"\nTopics --> {', '.join(topic)}\n"

        embedding = get_embedding([cur_section_text])[0]
        embedding = np.array(embedding,dtype=np.float32)

        groups = compare_embedding_master_table(embedding,5)

        if groups:
            for group in groups:
                pass


def get_embedding(exchange):
    response = loader.ollama_client.embeddings.create(model=embeddings_model, input=exchange)
    return response.data[0].embedding

def add_new_group(exchange, embedding, topics,date):
    group_name = str(uuid.uuid4()).replace("-", "")

    cursor_complex_rag.execute(f'CREATE TABLE "{group_name}"(id INTEGER PRIMARY KEY AUTOINCREMENT,exchange TEXT,embedding BLOB,topics TEXT,date TEXT); ')
    cursor_complex_rag.execute(
        f'INSERT INTO "{group_name}" (exchange, embedding,topics,date) VALUES (?, ?,?,?)',(exchange, np.array(embedding, dtype=np.float32).tobytes(), ", ".join(topics),date))
    complex_rag.commit()

    cursor_complex_rag.execute("INSERT INTO master_table (tables_name, group_embeddings, count, topics) VALUES (?, ?, ?, ?)",(group_name, np.array(embedding, dtype=np.float32).tobytes(), 1, ", ".join(topics)))

    make_graph(np.array(embedding,dtype=np.float32), group_name, True)
    complex_rag.commit()

def add_turn(hist):
    results,exchanges = rag_query(hist)
    topics = results["topics"]
    facts = results["facts"]
    for i, result in enumerate(results):
        cur_exchange = exchanges[i]
        f = result["facts"]
        cur_topics = result["topics"]

        cur_exchange += f"\nTopics: {", ".join(cur_topics)}"
        embedding = np.array(get_embedding(cur_exchange), dtype=np.float32).flatten()
        group_names = compare_embedding_master_table(embedding, 4)

        now = datetime.now()
        date = now.strftime('%A, %d %B %Y')

        if group_names:
            for name in group_names:
                cursor_complex_rag.execute(f'SELECT embedding FROM "{name}"')
                row = cursor_complex_rag.fetchall()

                embeddings_all = [np.frombuffer(r[0],dtype=np.float32) for r in row]

                add_to_graph(name,np.array(embeddings_all,dtype=np.float32),embedding)

                cursor_complex_rag.execute(f'INSERT INTO "{name}" (exchange, embedding,topics,date) VALUES (?, ?,?,?)',(cur_exchange, np.array(embedding, dtype=np.float32).tobytes(),", ".join(cur_topics),date))
                complex_rag.commit()

                cursor_complex_rag.execute('SELECT count,topics,group_embeddings FROM master_table WHERE tables_name = ?', (name,))

                row = cursor_complex_rag.fetchone()
                count = row[0]
                topics = row[1].split(", ")
                mean = np.frombuffer(row[2],dtype=np.float32)
                mean = np.array(mean,dtype=np.float32).flatten()
                all_topics = list(set(topics + cur_topics))

                new_mean = (((mean * count) + embedding) / (count + 1))

                cursor_complex_rag.execute(
                    'UPDATE master_table SET count=?, topics=?, group_embeddings=? WHERE tables_name=?',
                    (count+1, ", ".join(all_topics),new_mean.tobytes(), name))

                console.print("[dim]Old Group[/dim]")

        else:
            add_new_group(cur_exchange,embedding,cur_topics,date)
            console.print("[dim]New Group[/dim]")

    write_facts(f)

def compare_embedding_master_table(embedding, k):
    console.print("[dim]Getting matched Groups[/dim]")

    cursor_complex_rag.execute("SELECT tables_name,group_embeddings FROM master_table")
    rows = cursor_complex_rag.fetchall()

    if not rows:
        return []

    names = [row[0] for row in rows]

    embeddings =[np.frombuffer(row[1], dtype=np.float32)for row in rows]
    embeddings = np.array(embeddings,dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embedding)
    norms = np.where(norms == 0, 1e-9, norms)

    similarity = (embeddings @ embedding) / norms
    threshold = 0
    console.print(f"[dim]sims_master: {similarity}[/dim]")

    with open("../Data/Config/config_json.json", "r") as f:
        threshold = json.load(f)["master_tabel_threshold"]

    similarity_ids = similarity.argsort()[::-1]
    selected = [i for i in similarity_ids[:min(k, len(similarity_ids))] if similarity[i] > threshold]

    return [names[i] for i in selected]

def compare_embed_group(group_name, u_e):
    console.print("[dim]Re ranking in groups[/dim]")

    cursor_complex_rag.execute(f'SELECT exchange,embedding,topics FROM "{group_name}"')
    rows = cursor_complex_rag.fetchall()

    if not rows:
        return []

    summaries:list[str] = [row[0] for row in rows]

    embeddings = [np.frombuffer(row[1], dtype=np.float32)for row in rows]
    embeddings = np.array(embeddings,dtype=np.float32)

    threshold = 0

    with open("../Data/Config/config_json.json", "r") as f:
        threshold = json.load(f)["within_tabel"]

    ids = compare_embed(embeddings,u_e,group_name,threshold,5)
    console.print(f"[dim]ids: {ids}[/dim]")

    return [summaries[i] for i in ids]

def get_matches_rag(user, k, topics):
    embedding = get_embedding(user + f"Topics {topics}")
    embedding = np.array(embedding,dtype=np.float32).flatten()
    matches = compare_embedding_master_table(embedding, k)
    summaries=set()

    for group_name in matches:
        cur_summaries:list[str] = compare_embed_group(group_name, embedding)

        for x in cur_summaries:
            summaries.add(x)

    summary = " ".join(summaries)

    return summary,load_facts()

if __name__ == "__main__":
    console.print(get_matches_rag("Watched Thar Tensura ep", 10,[""]))