import uuid
import json
import loader
import sqlite3
import numpy as np
from datetime import datetime
from loader import console
from HNSW import compare_embed,make_graph

summary_model = "openai/gpt-oss-120b"
embeddings_model = "text-embedding-embeddinggemma-300m"

sys_prompt = '''
You are a conversation memory summarizer.

IMPORTANT OUTPUT RULES
- Output ONLY valid JSON
- No explanations, no markdown, no extra text
- Output must be a JSON array
- Each item corresponds to ONE exchange only
- Never merge multiple exchanges into one object

OUTPUT FORMAT (STRICT)
Each item must follow exactly:
{
  "facts": ["clean atomic facts", "..."],
  "topics": ["high-level topics"]
}

FIELD RULES
1. facts
- Extract ONLY explicit, meaningful information
- Must be short (no paragraphs)
- Normalize and fix typos (e.g. "pythn lerning" → "learning Python")
- Include:
  - preferences
  - goals
  - tools/models mentioned
  - numbers, constraints, or claims
- Exclude:
  - vague statements
  - conversational filler
  - redundant information

############################################
topics
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
- If no clear domain fits, use "general knowledge" or infer t

############################################
TOPIC SELECTION EXAMPLES
############################################

"explain tcp"
→ ["networking"]

"how do transformers work in ml"
→ ["machine learning", "deep learning"]

"best isekai anime"
→ ["anime"]

"train a cnn on images"
→ ["deep learning", "computer vision"]

"docker compose tutorial"
→ ["devops"]

"bitcoin price today"
→ ["cryptocurrency"]

"how to crack a faang interview"
→ ["career", "programming"]

"explain backpropagation"
→ ["deep learning"]

"what is inflation"
→ ["economics"]

"hi"
→ ["greeting"]

"hey bro"
→ ["greeting"]

Bad:
["tcp", "packets", "three-way handshake"]

Good:
["networking"]

Bad:
["transformers", "attention", "layers"]

Good:
["machine learning", "deep learning"]

Bad:
["naruto", "anime fights", "jutsu"]

Good:
["anime"]
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
    except:
        return []

def write_facts(facts):
    with open("../Data/docs/facts.json", "w") as f:
        json.dump(facts, f,indent=4)

def rag_query(hist):
    all = []
    msg = ""
    for i,m in enumerate(hist):
        if i % 2 == 0:
            msg += f"{i+1}: User: {m} \nASSISTANT: {hist[i+1]}"
            summ = f"{i+1}: User: {m} \nASSISTANT: {hist[i+1]}"
            all.append(summ)

    facts_msg = f"""\nOld Facts = {", ".join(load_facts())}
These are Old Facts Modify Them Or Delete Unusable Facts and give new facts containing older facts 
If none exist, use []"""

    response = loader.groq_client.chat.completions.create(model=summary_model, messages=[{"role": "system", "content":sys_prompt + " " + facts_msg}, {"role": "user", "content":msg}])
    raw = response.choices[0].message.content.strip()
    return json.loads(raw),all

def get_embedding(exchange):
    response = loader.lm_client.embeddings.create(model=embeddings_model, input=exchange)
    return response.data[0].embedding

def add_new_group(exchange, embedding, topics,date):
    group_name = str(uuid.uuid4()).replace("-", "")

    cursor_complex_rag.execute(f'CREATE TABLE "{group_name}"(id INTEGER PRIMARY KEY AUTOINCREMENT,exchange TEXT,embedding BLOB,topics TEXT,date TEXT); ')
    cursor_complex_rag.execute(
        f'INSERT INTO "{group_name}" (exchange, embedding,topics,date) VALUES (?, ?,?,?)',(exchange, np.array(embedding, dtype=np.float32).tobytes(), "' ".join(topics),date))
    complex_rag.commit()

    cursor_complex_rag.execute("INSERT INTO master_table (tables_name, group_embeddings, count, topics) VALUES (?, ?, ?, ?)",(group_name, np.array(embedding, dtype=np.float32).tobytes(), 1, ", ".join(topics)))

    make_graph(np.array(embedding,dtype=np.float32), group_name, True)
    complex_rag.commit()

def add_turn(hist):
    results,exchanges = rag_query(hist)
    f = []
    for i, (result, exchange) in enumerate(zip(results, exchanges)):
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

                cursor_complex_rag.execute(f'SELECT embedding FROM "{name}"')
                row = cursor_complex_rag.fetchall()

                embeddings_all = [np.frombuffer(r[0],dtype=np.float32) for r in row]
                make_graph(np.array(embeddings_all,dtype=np.float32),name,True)
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

    summaries = [row[0] for row in rows]

    embeddings = [np.frombuffer(row[1], dtype=np.float32)for row in rows]
    embeddings = np.array(embeddings,dtype=np.float32)

    threshold = 0

    with open("../Data/Config/config_json.json", "r") as f:
        threshold = json.load(f)["within_tabel"]

    ids = compare_embed(embeddings,u_e,group_name,threshold)
    console.print(f"[dim]ids: {ids}[/dim]")

    return [summaries[i] for i in ids]

def get_matches(user,k,topics):
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
    console.print(get_matches("Watched Thar Tensura ep", 10))