import json
import sqlite3
import numpy as np
from openai import OpenAI
import uuid
import os
from dotenv import load_dotenv
from main_flow import get_key_words

load_dotenv()

api_key = os.getenv("groq")

client_groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

client_lm = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm_studio")
summary_model = "openai/gpt-oss-120b"
embeddings_model = "text-embedding-embeddinggemma-300m"

all_facts= []

sys_prompt = f"""
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
}

SIMPLE RULES

summary
One sentence only
Must include user goal + assistant response
Never merge two exchanges into one summary

facts
Extract specific numbers, tool names, stated preferences, or explicit claims
Example facts: "Groq free tier gives 14k requests per day", "user prefers isekai anime"
Old Facts = {", ".join(all_facts)}
These are Old Facts Modify Them Or Delete Un usable Facts and give new facts containing older facts 
If none exist, use []

SIMPLIFIED GOAL
Turn each conversation exchange into a clean memory record that is easy to store and search.
"""

complex_rag = sqlite3.connect("../Data/complex_rag.db")
cursor_complex_rag = complex_rag.cursor()

cursor_complex_rag.execute("CREATE TABLE IF NOT EXISTS master_table("
                           "tables_name TEXT PRIMARY KEY, group_embeddings BLOB,count INT,key_words TEXT"
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

def add_new_group(summary,embedding,key_words):
    group_name = str(uuid.uuid4()).replace("-", "")

    cursor_complex_rag.execute(f'CREATE TABLE "{group_name}"(id INTEGER PRIMARY KEY AUTOINCREMENT,summary TEXT,embedding BLOB,tone TEXT,key_words TEXT); ')
    cursor_complex_rag.execute(
        f'INSERT INTO "{group_name}" (summary, embedding,key_words) VALUES (?, ?,?)',(summary, np.array(embedding, dtype=np.float32).tobytes(),"' ".join(key_words)))
    complex_rag.commit()

    cursor_complex_rag.execute("INSERT INTO master_table (tables_name, group_embeddings, count, key_words) VALUES (?, ?, ?, ?)",(group_name, np.array(embedding, dtype=np.float32).tobytes(), 1, ", ".join(key_words)))
    complex_rag.commit()

def add_turn(hist):
    results = get_summary(hist)

    for r in results:
        print("Initiated Save")
        summary = r["summary"]
        facts = r['facts']
        cur_key_words = get_key_words(summary)

        embedding = get_embedding(summary)
        group_names = compare_embedding_master_table(embedding, cur_key_words, 4)


        if group_names:
            for name in group_names:
                cursor_complex_rag.execute(f'INSERT INTO "{name}" (summary, embedding,key_words) VALUES (?, ?,?)',(summary, np.array(embedding, dtype=np.float32).tobytes(),cur_key_words))
                complex_rag.commit()

                cursor_complex_rag.execute('SELECT count,key_words,group_embeddings FROM master_table WHERE tables_name = ?', (name,))

                row = cursor_complex_rag.fetchone()
                count = row[0]
                key_words = row[1].split(", ")
                mean = np.frombuffer(row[2],dtype=np.float32)

                all_keys = list(set(key_words + cur_key_words))

                new_mean = (mean * count + np.array(embedding, dtype=np.float32)) / (count + 1)

                cursor_complex_rag.execute(
                    'UPDATE master_table SET count=?, key_words=?, group_embeddings=? WHERE tables_name=?',
                    (count+1, ", ".join(all_keys), np.array(new_mean,dtype=np.float32).tobytes(), name))
                print(f"Old Group ")

        else:
            add_new_group(summary,embedding,cur_key_words)
            print(f"New Group")


def compare_embedding_master_table(embedding, key_words, k):
    embedding = np.array(embedding,dtype=np.float32)

    cursor_complex_rag.execute("SELECT tables_name,group_embeddings,key_words FROM master_table")

    rows = cursor_complex_rag.fetchall()
    if not rows:
        return []

    names = [row[0] for row in rows]
    e = [np.frombuffer(row[1],dtype=np.float32) for row in rows]
    embeddings = np.array(e)
    row_topics = [row[2].split(", ") for row in rows]

    key_words_score = [len(set(kw) & set(key_words)) for kw in row_topics]

    similarity = embeddings @ embedding/ (np.linalg.norm(embeddings,axis=1) * np.linalg.norm(embedding))

    combined = similarity
    threshold = 0.7

    if key_words:
        key_words_score = np.array(key_words_score) / max(len(key_words), 1)
        combined = (similarity * 0.6) + (key_words_score * 0.4)
        threshold = 0.45
        print(key_words)

    combined_idx = combined.argsort()[::-1]

    selected = [i for i in combined_idx[:min(k,len(combined_idx))] if combined[i] > threshold]

    return [names[i] for i in selected]

def compare_embed_group(group_name,u_e,u_kw,k):
    cursor_complex_rag.execute(f'SELECT summary,embedding,key_words FROM "{group_name}"')
    rows = cursor_complex_rag.fetchall()
    if not rows:
        return []
    summaries = [row[0] for row in rows]
    embeddings = [np.frombuffer(row[1],dtype=np.float32) for row in rows]
    key_words = []

    for row in rows:
        keys = row[2].split(", ")
        key_words.extend(keys)

    similarity = np.dot(u_e, embeddings) / (np.linalg.norm(u_e) * np.linalg.norm(embeddings)).flatten()
    threshold = 0.7
    combined = similarity

    if u_kw:
        threshold = 0.45
        key_words_score = [sum(1 for x in u_kw if x in k)/len(k) for k in key_words]
        combined = (similarity * 0.6) + (key_words_score * 0.4)

    combined_idx = combined.argsort()[::-1]

    selected = [i for i in combined_idx[:min(k, len(combined_idx))] if combined[i] > threshold]

    return [summaries[i] for i in selected]

def get_matches(user,k):
    keywords = get_key_words(user)
    embedding = get_embedding(user)
    matches = compare_embedding_master_table(embedding, keywords, k)
    summaries=set()

    for group_name in matches:
        cur_sumeries:list[str] = compare_embed_group(group_name, embedding, keywords, k)

        for x in cur_sumeries:
            summaries.add(x)

    summary = " ".join(summaries)
    print("rag here: "+ summary)
    return zip(summary,all_facts)