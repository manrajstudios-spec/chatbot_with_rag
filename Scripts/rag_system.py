import uuid
import json
import sqlite3
import tiktoken
import numpy as np
from datetime import datetime
from sentence_transformers import util
from loader import console,get_response,get_embedding

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

FIELD 1:
 topics 
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
                           "tables_name TEXT PRIMARY KEY, group_mean BLOB,count INT,topics TEXT"
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

            if len(enc.encode(exchange)) > 6000:
                exchange = exchange[:6000]

            cur_chunk = [exchange]
        else:
            cur_chunk.append(exchange)

    if cur_chunk:
        chunks.append(cur_chunk)

    embeddings = []

    for chunk in chunks:
        embeddings.extend(np.array(get_embedding(chunk),dtype=np.float32))

    embeddings = np.array(embeddings)
    norm_embeddings = embeddings / np.clip(np.linalg.norm(embeddings,axis=1,keepdims=True),1e-8,None)

    groups = util.community_detection(norm_embeddings,min_community_size=1,threshold=0.6)

    grouped_exchanges = [[exchanges[i] for i in group] for group in groups]

    return grouped_exchanges,groups,embeddings

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

    sections,group_ids,embedded_exchange = divide_sections(exchanges)

    topics = parsed["topics"]

    sectioned_topics = [[topics[i] for i in idx] for idx in group_ids]
    sectioned_embedded_exchanges = [[embedded_exchange[i] for i in idx] for idx in group_ids]

    return parsed,sections,sectioned_topics,sectioned_embedded_exchanges

def compare_embedding_master_table(query_embedding, k):
    console.print("[dim]Getting matched Groups[/dim]")

    cursor_complex_rag.execute("SELECT tables_name,group_mean FROM master_table")
    rows = cursor_complex_rag.fetchall()

    if not rows:
        return []

    names = [row[0] for row in rows]

    embeddings =[np.frombuffer(row[1], dtype=np.float32)for row in rows]
    embeddings = np.array(embeddings,dtype=np.float32)

    query_embedding_norm = query_embedding/np.clip(np.linalg.norm(query_embedding),1e-10,None)

    similarity = embeddings @ query_embedding_norm
    threshold = 0
    console.print(f"[dim]sims_master: {similarity}[/dim]")

    with open("../Data/Config/config_json.json", "r") as f:
        threshold = json.load(f)["master_tabel_threshold"]

    k = min(k, len(similarity))
    if k <= 0:
        return []

    similarity_ids = np.argpartition(similarity,-k)[-k:]
    selected = [i for i in similarity_ids if similarity[i] > threshold]

    return [names[i] for i in selected]

def compare_embed_group(group_name, query_embed):
    console.print("[dim]Re ranking in groups[/dim]")

    cursor_complex_rag.execute(f'SELECT exchange,sim FROM "{group_name}"')
    rows = cursor_complex_rag.fetchall()

    exchanges = [row[0] for row in rows]
    sims = np.array([row[1] for row in rows], dtype=np.float32)

    cursor_complex_rag.execute(f'SELECT group_mean FROM "master_table" WHERE tables_name=?',(group_name,))
    rows = cursor_complex_rag.fetchone()
    mean = np.frombuffer(rows[0], dtype=np.float32)

    query_norm = query_embed / np.clip(np.linalg.norm(query_embed), 1e-8, None)

    offset = 0.15
    query_sim = float(mean @ query_norm)
    ids = np.where(np.abs(sims - query_sim) <= offset)[0]

    if len(ids) == 0:
        ids = np.argsort(sims - query_sim)[:5]

    console.print(f"[dim]ids: {ids}[/dim]")

    return [exchanges[i] for i in ids]

def add_new_group(exchanges, embeddings, topics, date):
    group_name = str(uuid.uuid4()).replace("-", "")

    cursor_complex_rag.execute(
        f'CREATE TABLE "{group_name}" (id INTEGER PRIMARY KEY AUTOINCREMENT,exchange TEXT,embedding BLOB,topics TEXT,date TEXT,sim REAL); ')
    complex_rag.commit()

    norm_embeddings = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True),1e-8,None)
    mean = np.mean(norm_embeddings, axis=0)
    mean_norm = mean / np.clip(np.linalg.norm(mean), 1e-8, None)

    sims = norm_embeddings @ mean_norm

    for cur_embedding, cur_topics, cur_exchange,sim in zip(embeddings, topics, exchanges,sims):
        cursor_complex_rag.execute(f'INSERT INTO "{group_name}" (exchange,embedding,topics,date,sim) VALUES (?,?,?,?,?) ',
                                   (cur_exchange, cur_embedding.tobytes(), ", ".join(cur_topics), date,float(sim)))
        complex_rag.commit()

    all_topics = []
    for t in topics:
        all_topics.extend(t)
    all_topics = set(all_topics)

    cursor_complex_rag.execute(f'INSERT INTO master_table(tables_name,group_mean,count,topics) VALUES (?,?,?,?)',
                               (group_name, mean_norm.tobytes(), len(embeddings), ", ".join(all_topics)))
    complex_rag.commit()

def add_to_rag(hist):
    parsed,exchange_sections,sectioned_topics,embedded_exchanges = rag_query(hist)
    facts = parsed["facts"]
    write_facts(facts)

    now = datetime.now()
    date = now.strftime('%A, %d %B %Y')

    for section_exchanges,section_topics,section_embeddings in zip(exchange_sections,sectioned_topics,embedded_exchanges):
        section_embeddings_norm = section_embeddings / np.clip(np.linalg.norm(section_embeddings,axis=1,keepdims=True), 1e-8, None)

        cur_section_text = "\n".join(section_exchanges)
        flatten_cur_section_topics = []
        for t in section_topics:
            flatten_cur_section_topics.extend(t)

        flatten_cur_section_topics = list(set(flatten_cur_section_topics))

        cur_section_text += f"\nTopics --> {', '.join(flatten_cur_section_topics)}\n"

        embedded_section = get_embedding([cur_section_text])[0]
        embedded_section = np.array(embedded_section,dtype=np.float32)

        groups = compare_embedding_master_table(embedded_section,5)

        if groups:
            for group in groups:
                cursor_complex_rag.execute(f'SELECT embedding FROM "{group}"')
                rows = cursor_complex_rag.fetchall()

                cur_table_embeddings = [np.frombuffer(row[0],dtype=np.float32) for row in rows]
                cur_table_embeddings = np.array(cur_table_embeddings)

                for cur_embed,cur_exchange,cur_topics in zip(section_embeddings_norm,section_exchanges,section_topics):
                    cursor_complex_rag.execute(f'INSERT INTO "{group}" (exchange,embedding,topics,date) VALUES(?,?,?,?)',(cur_exchange,cur_embed,", ".join(cur_topics),date))
                    complex_rag.commit()

                cursor_complex_rag.execute(f'SELECT count,topics,group_mean FROM master_table WHERE tables_name = ?',(group,))

                row = cursor_complex_rag.fetchone()

                count, all_topics_str, mean_blob = row[0], row[1], row[2]

                all_topics = all_topics_str.split(", ")
                count += len(section_embeddings)

                table_norm_embedding = cur_table_embeddings / np.clip(np.linalg.norm(cur_table_embeddings, axis=1, keepdims=True),1e-8,None)
                stacked = np.vstack([table_norm_embedding,section_embeddings_norm])
                new_mean = np.mean(stacked,axis=0)
                new_mean_norm = new_mean/np.clip(np.linalg.norm(new_mean), 1e-8, None)

                sims = stacked @ new_mean_norm

                for row_id, sim in zip(range(len(sims)), sims):
                    cursor_complex_rag.execute(f'UPDATE "{group}" SET sim=? WHERE id=?',(float(sim), row_id+1))

                all_topics.extend(flatten_cur_section_topics)

                all_topics = set(all_topics)

                cursor_complex_rag.execute(f'UPDATE master_table SET count=?,topics=?,group_mean=? WHERE tables_name=?',(count,", ".join(all_topics),new_mean_norm.tobytes(),group))
                complex_rag.commit()
        else:
            add_new_group(section_exchanges,section_embeddings_norm,section_topics,date)

def get_matches_rag(previous_exchanges_query_text, k, topics):
    embedding = get_embedding([previous_exchanges_query_text + f"Topics {topics}"])[0]
    embedding = np.array(embedding,dtype=np.float32).flatten()
    matched_groups = compare_embedding_master_table(embedding, k)
    summaries=set()

    for group_name in matched_groups:
        cur_summaries:list[str] = compare_embed_group(group_name, embedding)

        for x in cur_summaries:
            summaries.add(x)

    rag_output = "\n".join(summaries)

    return rag_output,load_facts()

if __name__ == "__main__":
    console.print(get_matches_rag("Watched Thar Tensura ep", 10,[""]))