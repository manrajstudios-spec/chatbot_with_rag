import uuid
import json
import loader
import sqlite3
import numpy as np
from HNSW import compare_embed,make_graph

summary_model = "openai/gpt-oss-120b"
embeddings_model = "text-embedding-embeddinggemma-300m"

sys_prompt = '''
You are a conversation memory summarizer.

Your job is to convert each conversation exchange into structured JSON for long-term memory storage and retrieval.

IMPORTANT OUTPUT RULES
- Output ONLY valid JSON
- No explanations, no markdown, no extra text
- Output must be a JSON array
- Each item corresponds to ONE exchange only
- Never merge multiple exchanges into one object

OUTPUT FORMAT (STRICT)
Each item must follow exactly:
{
  "summary": "one sentence describing user intent and assistant action",
  "facts": ["clean atomic facts", "..."],
  "topics": ["high-level topics"]
}

FIELD RULES

1. summary
- Must be exactly ONE sentence
- Must include:
  - what the user wanted
  - what the assistant did/responded with
- Do not include unnecessary details
- Do not merge multiple exchanges

2. facts
- Extract ONLY explicit, meaningful information
- Must be short, atomic statements (no paragraphs)
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

Example facts:
- "Groq free tier gives 14k requests per day"
- "User is learning machine learning"
- "Interested in anime isekai"

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

SIMPLIFIED GOAL
Convert each exchange into clean, structured, searchable memory that is optimized for retrieval systems and embeddings.
'''
complex_rag = sqlite3.connect("../Data/chat_data/complex_rag.db")
cursor_complex_rag = complex_rag.cursor()

cursor_complex_rag.execute("CREATE TABLE IF NOT EXISTS master_table("
                           "tables_name TEXT PRIMARY KEY, group_embeddings BLOB,count INT,topics TEXT"
                           ");")
def as_matrix(x):
    x = np.array(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return x

def load_facts():
    try:
        with open("../Data/docs/facts.json", "r") as f:
            return json.load(f)
    except:
        return []

def write_facts(facts):
    with open("../Data/docs/facts.json", "w") as f:
        json.dump(facts, f,indent=4)

def get_summary(hist):
    all = []
    msg = ""
    for i,m in enumerate(hist):
        if i % 2 == 0:
            msg += f"{i+1}: User: {m} \nASSISTANT: {hist[i+1]}"
            summ = f"{i+1}: {m} \nASSISTANT: {hist[i+1]} \n\n"
            all.append(summ)

    facts_msg = f"""\nOld Facts = {", ".join(load_facts())}
These are Old Facts Modify Them Or Delete Un usable Facts and give new facts containing older facts 
If none exist, use []"""

    response = loader.groq_client.chat.completions.create(model=summary_model, messages=[{"role": "system", "content":sys_prompt+ facts_msg}, {"role": "user", "content":msg}])
    raw = response.choices[0].message.content.strip()
    return json.loads(raw),all

def get_embedding(summary):
    response = loader.lm_client.embeddings.create(model=embeddings_model, input=summary)
    return response.data[0].embedding

def add_new_group(summary,embedding,topics):
    group_name = str(uuid.uuid4()).replace("-", "")

    cursor_complex_rag.execute(f'CREATE TABLE "{group_name}"(id INTEGER PRIMARY KEY AUTOINCREMENT,summary TEXT,embedding BLOB,topics TEXT); ')
    cursor_complex_rag.execute(
        f'INSERT INTO "{group_name}" (summary, embedding,topics) VALUES (?, ?,?)',(summary, np.array(embedding, dtype=np.float32).tobytes(),"' ".join(topics)))
    complex_rag.commit()

    cursor_complex_rag.execute("INSERT INTO master_table (tables_name, group_embeddings, count, topics) VALUES (?, ?, ?, ?)",(group_name, np.array(embedding, dtype=np.float32).tobytes(), 1, ", ".join(topics)))

    make_graph(as_matrix([embedding]), group_name, True)
    complex_rag.commit()

def add_turn(hist):
    results,all = get_summary(hist)
    f = []
    for r in results:
        print("Initiated Save")
        summary = r["summary"]
        summary = all
        facts = r['facts']
        cur_topics = r["topics"]
        f = facts

        summary += f"KeyWords: {", ".join(cur_topics)}"
        embedding = np.array(get_embedding(summary), dtype=np.float32)
        group_names = compare_embedding_master_table(embedding, 4)

        if group_names:
            for name in group_names:
                cursor_complex_rag.execute(f'INSERT INTO "{name}" (summary, embedding,topics) VALUES (?, ?,?)',(summary, np.array(embedding, dtype=np.float32).tobytes(),", ".join(cur_topics)))
                complex_rag.commit()

                cursor_complex_rag.execute('SELECT count,topics,group_embeddings FROM master_table WHERE tables_name = ?', (name,))

                row = cursor_complex_rag.fetchone()
                count = row[0]
                topics = row[1].split(", ")
                mean = np.frombuffer(row[2],dtype=np.float32)

                all_topics = list(set(topics + cur_topics))

                new_mean = (mean * count + np.array(embedding, dtype=np.float32)) / (count + 1)

                cursor_complex_rag.execute(
                    'UPDATE master_table SET count=?, topics=?, group_embeddings=? WHERE tables_name=?',
                    (count+1, ", ".join(all_topics), np.array(new_mean,dtype=np.float32).tobytes(), name))

                cursor_complex_rag.execute(f'SELECT embedding FROM "{name}"')
                row = cursor_complex_rag.fetchall()

                embeddings_all = [np.frombuffer(r[0],dtype=np.float32) for r in row]
                make_graph(np.stack(embeddings_all,dtype=np.float32),name,True)

                print(f"Old Group ")

        else:
            add_new_group(summary,np.stack(embedding),cur_topics)
            print(f"New Group")

    write_facts(f)

def compare_embedding_master_table(embedding, k):
    print("Getting matched Groups")
    embedding = as_matrix(embedding)[0]
    cursor_complex_rag.execute("SELECT tables_name,group_embeddings FROM master_table")
    rows = cursor_complex_rag.fetchall()

    if not rows:
        return []

    names = [row[0] for row in rows]

    embeddings = as_matrix([np.frombuffer(row[1], dtype=np.float32)for row in rows])
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embedding)
    norms = np.where(norms == 0, 1e-9, norms)

    similarity = (embeddings @ embedding) / norms
    threshold = 0

    with open("../Data/Config/config_json.json", "r") as f:
        threshold = json.load(f)["master_tabel_threshold"]

    print(f"sims_master: {similarity}")
    similarity_ids = similarity.argsort()[::-1]
    selected = [i for i in similarity_ids[:min(k, len(similarity_ids))] if similarity[i] > threshold]

    return [names[i] for i in selected]

def compare_embed_group(group_name, u_e):
    print("Re ranking in groups")
    cursor_complex_rag.execute(f'SELECT summary,embedding,topics FROM "{group_name}"')
    rows = cursor_complex_rag.fetchall()

    if not rows:
        return []

    summaries = [row[0] for row in rows]

    embeddings = as_matrix([np.frombuffer(row[1], dtype=np.float32)for row in rows])

    u_e = np.array(u_e, dtype=np.float32)

    threshold = 0

    with open("../Data/Config/config_json.json", "r") as f:
        threshold = json.load(f)["within_tabel"]

    ids = compare_embed(embeddings,u_e,group_name,threshold)
    print(f"ids: {ids}")

    return [summaries[i] for i in ids]

def get_matches(user,k,topics):
    embedding = get_embedding(user + f"Topics {topics}")
    matches = compare_embedding_master_table(embedding, k)
    summaries=set()

    for group_name in matches:
        cur_summaries:list[str] = compare_embed_group(group_name, embedding)

        for x in cur_summaries:
            summaries.add(x)

    summary = " ".join(summaries)

    return summary,load_facts()

if __name__ == "__main__":
    print(get_matches("Watched Thar Tensura ep",10))