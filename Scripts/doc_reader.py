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

def get_summary(query):
    sys_prompt = """ You are a document summarization assistant.

Your task:
- Read the document chunk provided by the user.
- Extract the most important information.
- Respond ONLY with valid JSON in the exact format below. No extra text, no explanation, no markdown.

Output format:
{"summary": "<concise summary of the chunk>", "key_words": ["<keyword1>", "<keyword2>", "<keyword3>"]}

Rules:
- summary: 2-4 sentences. Keep facts, names, dates, and numbers.
- key_words: 3-7 important terms from the text only.
- Do not add any text before or after the JSON.
- Do not use markdown code blocks.

Document chunk: """
    m = [{"role":"system","content":f"{sys_prompt} {query}"},{"role":"user","content":f"Doc Chunk: {query}"}]
    response = client_lm.chat.completions.create(model=summary_model, messages=m)

    raw = response.choices[0].message.content.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)

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
    limit = 1000
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
    text = read_doc(doc_path)
    chunks = make_chunks(text)
    summaries = make_summaries(chunks)

    embeddings =[]
    for summary in summaries:
        e = get_embedding(summary["summary"])
        embeddings.append({"key_words":summary["key_words"],"embedding":e})

    




