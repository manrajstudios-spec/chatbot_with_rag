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

SIMPLIFIED GOAL
Turn each conversation exchange into a clean memory record that is easy to store and search.
'''

complex_rag = sqlite3.connect("../Data/chat_data/complex_rag.db")
cursor_complex_rag = complex_rag.cursor()

cursor_complex_rag.execute("CREATE TABLE IF NOT EXISTS master_table("
                           "tables_name TEXT PRIMARY KEY, group_embeddings BLOB,count INT,key_words TEXT"
                           ");")

def load_facts():
    try:
        with open("../Data/docs/facts.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def write_facts(facts):
    with open("../Data/docs/facts.json", "w") as f:
        json.dump(facts, f,indent=4)

def get_summary(hist):
    msg = ""
    for i,m in enumerate(hist):
        if i % 2 == 0:
            msg += f"{i+1}: User: {m} \nASSISTANT: {hist[i+1]}"

    facts_msg = f"""\nOld Facts = {", ".join(load_facts())}
These are Old Facts Modify Them Or Delete Un usable Facts and give new facts containing older facts 
If none exist, use []"""

    response = loader.groq_client.chat.completions.create(model=summary_model, messages=[{"role": "system", "content":sys_prompt+ facts_msg}, {"role": "user", "content":msg}])
    raw = response.choices[0].message.content.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)

def get_embedding(summary):
    response = loader.lm_client.embeddings.create(model=embeddings_model, input=summary)
    return response.data[0].embedding

def add_new_group(summary,embedding,key_words):
    group_name = str(uuid.uuid4()).replace("-", "")

    cursor_complex_rag.execute(f'CREATE TABLE "{group_name}"(id INTEGER PRIMARY KEY AUTOINCREMENT,summary TEXT,embedding BLOB,key_words TEXT); ')
    cursor_complex_rag.execute(
        f'INSERT INTO "{group_name}" (summary, embedding,key_words) VALUES (?, ?,?)',(summary, np.array(embedding, dtype=np.float32).tobytes(),"' ".join(key_words)))
    complex_rag.commit()

    cursor_complex_rag.execute("INSERT INTO master_table (tables_name, group_embeddings, count, key_words) VALUES (?, ?, ?, ?)",(group_name, np.array(embedding, dtype=np.float32).tobytes(), 1, ", ".join(key_words)))

    make_graph(embedding,0.7,group_name)
    complex_rag.commit()

def add_turn(hist):
    results = get_summary(hist)
    f = []
    for r in results:
        print("Initiated Save")
        summary = r["summary"]
        facts = r['facts']
        f = facts
        cur_key_words = loader.extract_key_words(summary)
        cur_key_words = [kw for kw in cur_key_words if kw != "assistant"]

        summary += f"KeyWords: {", ".join(cur_key_words)}"
        embedding = get_embedding(summary)
        group_names = compare_embedding_master_table(embedding, cur_key_words, 4)

        if group_names:
            for name in group_names:
                cursor_complex_rag.execute(f'INSERT INTO "{name}" (summary, embedding,key_words) VALUES (?, ?,?)',(summary, np.array(embedding, dtype=np.float32).tobytes(),", ".join(cur_key_words)))
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

                cursor_complex_rag.execute(f'SELECT embedding FROM "{name}"')
                row = cursor_complex_rag.fetcall()

                embeddings_all = [np.frombuffer(r[0],dtype=np.float32) for r in row]
                make_graph(np.array(embeddings_all,dtype=np.float32),0.7,name)

                print(f"Old Group ")

        else:
            add_new_group(summary,embedding,cur_key_words)
            print(f"New Group")

    write_facts(f)

def compare_embedding_master_table(embedding, key_words, k):
    print("Getting matched Groups")
    embedding = np.array(embedding, dtype=np.float32)
    cursor_complex_rag.execute("SELECT tables_name,group_embeddings,key_words FROM master_table")
    rows = cursor_complex_rag.fetchall()

    if not rows:
        return []

    names = [row[0] for row in rows]

    embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows])
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embedding)
    norms = np.where(norms == 0, 1e-9, norms)
    similarity = (embeddings @ embedding) / norms
    threshold = 0.7
    similarity_ids = similarity.argsort()[::-1]
    selected = [i for i in similarity_ids[:min(k, len(similarity_ids))] if similarity[i] > threshold]

    return [names[i] for i in selected]


def compare_embed_group(group_name, u_e, u_kw, k):
    print("Re ranking in groups")
    cursor_complex_rag.execute(f'SELECT summary,embedding,key_words FROM "{group_name}"')
    rows = cursor_complex_rag.fetchall()

    if not rows:
        return []

    summaries = [row[0] for row in rows]

    embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows])
    u_e = np.array(u_e, dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(u_e)
    norms = np.where(norms == 0, 1e-9, norms)
    similarity = (embeddings @ u_e) / norms
    threshold = 0.7

    ids = compare_embed(embeddings,similarity,group_name,threshold)
    similarity = similarity[ids]
    similarity_ids = similarity.argsort()[::-1]

    return [summaries[i] for i in similarity_ids]

def get_matches(user,k):
    keywords = loader.extract_key_words(user)
    user += f"keywords: {keywords}"
    embedding = get_embedding(user)
    matches = compare_embedding_master_table(embedding, keywords, k)
    summaries=set()

    for group_name in matches:
        cur_summaries:list[str] = compare_embed_group(group_name, embedding, keywords, k)

        for x in cur_summaries:
            summaries.add(x)

    summary = " ".join(summaries)

    return summary,load_facts()

if __name__ == "__main__":
    print(get_matches("Watched Thar Tensura ep",10))