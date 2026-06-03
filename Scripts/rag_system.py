import json
import sqlite3
import numpy as np
from openai import OpenAI
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("groq")

client_groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

client_lm = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm_studio")
summary_model = "openai/gpt-oss-120b"
embeddings_model = "text-embedding-embeddinggemma-300m"

sys_prompt = """
You are a conversation memory summarizer.
Your job is to convert each conversation exchange into structured JSON.

IMPORTANT OUTPUT RULES
Output ONLY JSON
No explanations, no markdown, no extra text
Output must be a JSON array
Each item = one exchange
One object per exchange, never merge exchanges

OUTPUT FORMAT
Each item must look like this:
{
  "summary": "one sentence describing what user wanted and what assistant did",
  "facts": ["important factual details mentioned (or empty array)"],
  "topics": ["1 or 2 broad categories"],
  "tone": "casual | serious | emotional"
}

SIMPLE RULES

summary
One sentence only
Must include user goal + assistant response
Never merge two exchanges into one summary

facts
Extract specific numbers, tool names, stated preferences, or explicit claims
Example facts: "Groq free tier gives 14k requests per day", "user prefers isekai anime"
If none exist, use []

topics (VERY IMPORTANT)
Choose ONLY from this list:
anime
coding
personal
health
work
gaming
food
relationships
Use 1 or 2 only
Do NOT use names, tools, or specific things

tone
Pick ONE:
casual (normal chat)
serious (informational or task-based)
emotional (feelings, personal expression)

SIMPLIFIED GOAL
Turn each conversation exchange into a clean memory record that is easy to store and search.
"""

complex_rag = sqlite3.connect("../Data/complex_rag.db")
cursor_complex_rag = complex_rag.cursor()

cursor_complex_rag.execute("CREATE TABLE IF NOT EXISTS master_table("
                           "tables_name TEXT PRIMARY KEY, group_embeddings BLOB,count INT,topics TEXT"
                           ");")


def get_summary(hist):
    msg = ""
    for i,m in enumerate(hist):
        if i % 2 == 0:
            msg += f"{i+1}: User: {m} \nASSISTANT: {hist[i+1]}"

    response = client_groq.chat.completions.create(model=summary_model, messages=[{"role": "system", "content":sys_prompt}, {"role": "user", "content":msg}])
    raw = response.choices[0].message.content.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)

def get_embedding(summary):
    response = client_lm.embeddings.create(model=embeddings_model, input=summary)
    return response.data[0].embedding

def add_new_group(summary,embedding,topics,tone,t):
    group_name = str(uuid.uuid4()).replace("-", "")

    cursor_complex_rag.execute(f'CREATE TABLE "{group_name}"(id INTEGER PRIMARY KEY AUTOINCREMENT,summary TEXT,embedding BLOB,tone TEXT,topics TEXT); ')
    cursor_complex_rag.execute(
        f'INSERT INTO "{group_name}" (summary, embedding, tone,topics) VALUES (?, ?, ?,?)',(summary, np.array(embedding, dtype=np.float32).tobytes(), tone,t))
    complex_rag.commit()

    cursor_complex_rag.execute("INSERT INTO master_table (tables_name, group_embeddings, count, topics) VALUES (?, ?, ?, ?)",(group_name, np.array(embedding, dtype=np.float32).tobytes(), 1, ", ".join(topics)))
    complex_rag.commit()

def add_turn(hist):
    results = get_summary(hist)

    for r in results:
        print("Initiated Save")
        summary = r["summary"]
        tone = r['tone']
        facts = r['facts']
        cur_topics = r['topics']

        embedding = get_embedding(summary)
        group_names = compare_embedding(embedding,cur_topics,4)

        t = ", ".join(cur_topics)

        if group_names:
            for name in group_names:
                cursor_complex_rag.execute(f'INSERT INTO "{name}" (summary, embedding,tone,topics) VALUES (?, ?,?,?)',(summary, np.array(embedding, dtype=np.float32).tobytes(),tone,t))
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
                print(f"Old Group ")

        else:
            add_new_group(summary,embedding,cur_topics,tone,t)
            print(f"New Group")


def compare_embedding(embedding,topics,k):
    embedding = np.array(embedding,dtype=np.float32)

    cursor_complex_rag.execute("SELECT tables_name,group_embeddings,topics FROM master_table")

    rows = cursor_complex_rag.fetchall()
    if not rows:
        return []

    names = [row[0] for row in rows]
    e = [np.frombuffer(row[1],dtype=np.float32) for row in rows]
    embeddings = np.array(e)
    row_topics = [row[2].split(", ") for row in rows]

    topic_scores = [len(set(t) & set(topics)) for t in row_topics]

    similarity = embeddings @ embedding/ (np.linalg.norm(embeddings,axis=1) * np.linalg.norm(embedding))

    combined = similarity
    threshold = 0.7

    if topics:
        print(topics)
        topic_scores = np.array(topic_scores) / max(len(topics), 1)
        combined = (similarity * 0.6) + (topic_scores * 0.4)
        threshold = 0.45
        print(topics)

    combined_idx = combined.argsort()[::-1]

    selected = [i for i in combined_idx[:min(k,len(combined_idx))] if combined[i] > threshold]

    return [names[i] for i in selected]

def get_matches(user,k):

    embedding = get_embedding(user)
    matches = compare_embedding(embedding,[],k)
    summaries=[]

    for group_name in matches:
        cursor_complex_rag.execute(f'SELECT summary FROM "{group_name}" ORDER BY id DESC LIMIT 5')
        ss = cursor_complex_rag.fetchall()
        summaries.extend([s[0] for s in ss])

    summary = " ".join(summaries)
    print("rag here: "+ summary)
    return summary