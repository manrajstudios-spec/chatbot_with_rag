import uuid
import json
import loader
import sqlite3
import numpy as np
from datetime import datetime
from loader import console
from graph_search import compare_embed,make_graph,add_to_graph



# Fixes Left

# Handle input for rag and keeping good exchnages only

summary_model = "openai/gpt-oss-120b"
embeddings_model = "text-embedding-embeddinggemma-300m"

sys_prompt = '''You are a conversation memory fact and topic extractor.

IMPORTANT OUTPUT RULES

*   Output ONLY valid JSON
*   No explanations, no markdown, no extra text
*   Output must be a JSON array
*   Each item in the output array corresponds to ONE useful exchange from the input list.
*   STRICT RULE: If an input exchange contains no meaningful information (e.g., just greetings, small talk, or filler), completely drop it. Do not include it in the final array.
*   STRICT RULE: Do not summarize the conversation or write descriptive summaries. Only extract discrete atomic facts and high-level topics.

INPUT FORMAT EXPECTED  
The system will receive data in the following structure:  
{  
"old_facts": ["previously extracted user facts", "..."],  
"exchanges": [  
{  
"user": "User message text",  
"assistant": "Assistant response text"  
}  
]  
}

OUTPUT FORMAT (STRICT)  
The final output must be a clean JSON array containing elements only for the useful, kept exchanges:  
[  
{  
"facts": ["clean atomic facts extracted from this exchange", "..."],  
"topics": ["high-level topics for this exchange"]  
}  
]

FIELD RULES: USER FACTS & STATE MANAGEMENT

1.  EXTRACT NEW FACTS:
    *   Extract ONLY explicit, meaningful information about the user.
    *   Must be short, punchy fragments (no paragraphs).
    *   Normalize terms and fix typos (e.g., pythn lerning -> learning Python).
2.  MERGE & UPDATE STATE (INTEGRATE WITH OLD FACTS):
    *   Compare new extractions against the provided list of old facts.
    *   If user preferences or situations have changed, OVERWRITE the old fact with the new tracking data.
    *   If a new fact provides more detail to an old fact, MERGE them into a single precise point.
    *   If a new fact is entirely new, APPEND it to the list.
3.  STRICTLY INCLUDE user-related details only:
    *   User current preferences, dislikes, and interests.
    *   User goals, deadlines, and timelines.
    *   Tools, models, or technologies the user actively uses or wants to learn.
    *   Numbers, constraints, or claims specific to the user's current situation.

FIELD RULES: EXCLUSIONS & FILTERING (CRITICAL)  
4. STRICTLY EXCLUDE UNWANTED EXCHANGES:

*   Do not process, evaluate, or output any item for input exchanges that are just greetings, small talk, or conversational filler (e.g., "hi", "hello", "how are you", "thanks", "ok", "you're welcome").
*   Omit any exchange that does not contain worth-storing facts or major topic shifts.

5.  STRICTLY EXCLUDE CONTENT-WISE:
    *   General knowledge, tech concepts, or industry debates (e.g., "Attention is better than LSTM").
    *   Redundant or conflicting outdated information.

############################################  
topics  
############################################  
Purpose:  
Generate BROAD semantic categories for retrieval, memory lookup, semantic search, clustering, and routing.

Rules:

*   Topics are NOT keywords. They represent the general subject area/domain of the request.
*   Prefer broad domains over specific terms (e.g., "networking" not "tcp packets").
*   Prefer categories over keywords (e.g., "machine learning" not "transformer layers").
*   Use 1-5 topics when possible. Maximum 8.
*   Remove duplicates. Order by importance.
*   Use lowercase unless proper noun.

Topic selection guidance:

*   For technology: networking, operating systems, programming, web development, databases, devops, cybersecurity, mobile development, version control, artificial intelligence, machine learning, deep learning, computer vision, natural language processing, reinforcement learning, data science
*   For finance: finance, stock market, cryptocurrency, personal finance, economics
*   For science: mathematics, physics, chemistry, biology
*   For creative: design, graphic design, video & animation, music
*   For entertainment: anime, gaming, movies & tv, sports
*   For career: career, education, productivity
*   For lifestyle: health, travel, food
*   For general: general knowledge, current events, social

############################################  
TOPIC SELECTION EXAMPLES  
############################################  
"explain tcp" → ["networking"]  
"how do transformers work in ml" → ["machine learning", "deep learning"]  
"best isekai anime" → ["anime"]  
"train a cnn on images" → ["deep learning", "computer vision"]  
"docker compose tutorial" → ["devops"] 
"bitcoin price today" → ["cryptocurrency"]  
"how to crack a faang interview" → ["career", "programming"]  
"explain backpropagation" → ["deep learning"]  
"what is inflation" → ["economics"]

Bad: ["tcp", "packets", "three-way handshake"]  
Good: ["networking"]

Bad: ["transformers", "attention", "layers"]  
Good: ["machine learning", "deep learning"]

Bad: ["naruto", "anime fights", "jutsu"]  
Good: ["anime"]
'''
complex_rag = sqlite3.connect("../Data/chat_data/complex_rag.db")
cursor_complex_rag = complex_rag.cursor()

cursor_complex_rag.execute("CREATE TABLE IF NOT EXISTS master_table("
                           "tables_name TEXT PRIMARY KEY, group_embeddings BLOB,count INT,topics TEXT"
                           ");")

def load_facts():
    try:
        with open("../Data/docs/facts.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def write_facts(facts):
    with open("../Data/docs/facts.json", "w") as f:
        json.dump(facts, f,indent=4)

def rag_query(hist):
    msg = ""

    for i,m in enumerate(hist):
        if i % 2 == 0:
            msg += f"{i}: User: {m} \nASSISTANT: {hist[i+1]}\n"

    facts_msg = ", ".join(load_facts())

    response = loader.groq_client.chat.completions.create(model=summary_model, messages=[{"role": "system", "content":sys_prompt + "\nFacts -> " + facts_msg}, {"role": "user", "content":msg}])
    raw = response.choices[0].message.content.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"Failed to parse raw output: {raw}")
        return []

def get_embedding(exchange):
    response = loader.lm_client.embeddings.create(model=embeddings_model, input=exchange)
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
    results = rag_query(hist)
    f = []

    for i, result in enumerate(results):
        console.print("[dim]Initiated Save[/dim]")
        cur_exchange = exchange
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