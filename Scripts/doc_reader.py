import os
import re
import json
import pdfplumber
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("groq")

client_lm = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm_studio")
summary_model = "granite-4.1-3b"
embeddings_model = "text-embedding-embeddinggemma-300m"

def load_json():
    try:
        with open("./Data/doc_ref.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def write_json(d):
    data = load_json()
    data.append(d)
    with open("./Data/doc_ref.json", "w") as f:
        json.dump(data, f)

def get_summary(query):
    sys_prompt = """ You are a document summarization assistant.

TASK:
Read the document chunk and return ONLY a single valid JSON object.

OUTPUT SCHEMA (must match exactly):

{
  "summary": "string",
  "key_words": ["string"]
}

RULES:
1. Return ONLY JSON.
2. Do NOT output markdown.
3. Do NOT output code fences.
4. Do NOT output explanations.
5. Do NOT output notes before or after the JSON.
6. The JSON must be parseable by json.loads().
7. "summary" must contain 2-4 sentences.
8. Preserve important facts, names, dates, numbers, and technical terms.
9. "key_words" must contain 3-7 keywords taken from the document.
10. Do not invent information that is not present in the document.

EXAMPLE OUTPUT:

{
  "summary": "Apple announced a new AI model in June 2025. The model achieved strong benchmark performance and will be integrated into several products.",
  "key_words": ["Apple", "AI model", "June 2025", "benchmark", "products"]
}

DOCUMENT CHUNK: """

    m = [{"role":"system","content":f"{sys_prompt} {query}"},{"role":"user","content":f"Doc Chunk: {query}"}]
    response = client_lm.chat.completions.create(model=summary_model, messages=m)

    raw = response.choices[0].message.content
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"Parsing Error: {e}")

def get_embedding(query):
    response = client_lm.embeddings.create(model=embeddings_model, input=query)

    return response.data[0].embedding

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def read_doc(path):
    text = ""

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return clean_text(text)

def make_chunks(text):
    paras = text.split("\n")
    chunks = []
    limit = 1200
    hard_limit = 2200
    offset = 200
    cur_chunk = ""

    for para in paras:
        if not cur_chunk:
            if len(para) >= limit:
                if len(para) - offset <= limit:
                    chunks.append(para)
                else:
                    cur_chunk += para
            else:
                if len(para) + offset >= limit:
                    chunks.append(para)
                else:
                    cur_chunk += para
            continue

        full = cur_chunk + "\n" + para
        length = len(para) + len(cur_chunk)

        if length >= hard_limit:
            index = 0
            temp = ""

            for t in full[::-1]:
                index += 1
                temp += t

                if length - index <= hard_limit:
                    if t == ".":
                        break
            temp = temp[::-1]
            cur_chunk = temp
            full = full[::-1][:index][::-1]
            chunks.append(full)
        else:

            if length <= limit:
                cur_chunk += para
                continue

            chunks.append(full)
            cur_chunk = ""

    return chunks

def make_summaries(chunks):
    summaries = []
    for chunk in chunks:
        s = get_summary(chunk)
        summaries.append(s)
    return summaries

def add_doc(doc_path):
    file_name = doc_path.split("/")[-1]

    data = load_json()
    if data:
        for d in data:
            if d['file_name'] == file_name:
                return

    text = read_doc(doc_path)
    chunks = make_chunks(text)

    summaries = make_summaries(chunks)

    key_words = []
    embeddings = []
    for summary in summaries:
        e = get_embedding(summary["summary"])
        embeddings.append(e["embedding"])
        key_words.append(summary["key_words"])

    np_keywords = np.array(key_words)
    np_embeddings = np.array(embeddings,dtype=np.float32)

    file_name_npz = f"../Data/docs/npz_files/{file_name}.npz"

    np.savez(file_name_npz,key_words=np_keywords,embeddings=np_embeddings)

    json_dict = {"file_name":file_name ,"npz_path":file_name_npz ,"summary":"\n".join([i["summary"] for i in summaries])}

    write_json(json_dict)


add_doc("../Data/docs/attentionpaper.pdf")