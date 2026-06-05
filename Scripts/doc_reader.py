import os
import re
import json
import pdfplumber
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from tkinter import Tk,filedialog

root = Tk()
root.withdraw()

load_dotenv()
api_key = os.getenv("groq")

client_lm = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm_studio")
summary_model = "nvidia/nemotron-3-nano-4b"
embeddings_model = "text-embedding-embeddinggemma-300m"

max_attachments = 2

def ask_user(input_msg):
    while True:
        user_input = input(f"{input_msg}: ")
        if user_input:
            return user_input

def load_json():
    try:
        with open("../Data/docs/doc_ref.json", "r") as f:
            return json.load(f)
    except:
        return []

def write_json(d):
    data = load_json()
    data.append(d)
    with open("../Data/docs/doc_ref.json", "w") as f:
        json.dump(data, f,indent=4)

def get_summary(query):
    sys_prompt = """You are a document summarization assistant.

    TASK:
    Read the list of document chunks provided and return ONLY a single valid JSON array.
    Each element in the array must correspond to one chunk in the same order.

    OUTPUT SCHEMA (must match exactly):

    [
      {
        "summary": "string",
        "key_words": ["string"]
      }
    ]

    RULES:
    1. Return ONLY a JSON array.
    2. Do NOT output markdown.
    3. Do NOT output code fences.
    4. Do NOT output explanations.
    5. Do NOT output notes before or after the JSON.
    6. The JSON must be parseable by json.loads().
    7. Each "summary" must contain 2-4 sentences.
    8. Preserve important facts, names, dates, numbers, and technical terms.
    9. "key_words" must contain 3-7 keywords per chunk.
    10. Do not invent information not present in the document.
    11. The output array length MUST equal the number of input chunks.

    EXAMPLE OUTPUT:

    [
      {
        "summary": "Apple announced a new AI model in June 2025. The model achieved strong benchmark performance and will be integrated into several products.",
        "key_words": ["Apple", "AI model", "June 2025", "benchmark", "products"]
      },
      {
        "summary": "NVIDIA released a new GPU architecture focused on efficiency and training speed improvements.",
        "key_words": ["NVIDIA", "GPU", "architecture", "efficiency", "training speed"]
      }
    ]

    DOCUMENT CHUNKS:
    """

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
    len_batch = 3
    batches = [chunks[i:i+len_batch] for i in range(0, len(chunks), len_batch)]

    print(len(batches[-1]))
    print(len(chunks))

    summaries = []
    for batch in batches:
        s = get_summary(batch)
        summaries.append(s)

    s = [i for summary in summaries for i in summary]

    print(s[0]['key_words'])
    return s

def add_doc(doc_path):
    file_name = doc_path.split("/")[-1]
    file_name = file_name.split(".")[0]

    data = load_json()
    if data:
        for d in data:
            if d['file_name'] == file_name:
                return
    text = ""

    if doc_path.endswith(".pdf"):
        text = read_doc(doc_path)
    elif doc_path.endswith(".txt"):
        with open(doc_path,"r") as f:
            text = f.read()

    chunks = make_chunks(text)

    summaries = make_summaries(chunks)

    key_words = []
    embeddings = []

    for summary in summaries:
        e = get_embedding(summary["summary"])
        embeddings.append(e)

        if not summary["key_words"]:
            key_words.append([])
        else:
            key_words.append(summary["key_words"])

    np_keywords = np.array(key_words,dtype=object)
    np_embeddings = np.array(embeddings,dtype=np.float32)

    file_name_npz = f"../Data/docs/npz_files/{file_name}.npz"
    np.savez(file_name_npz,key_words=np_keywords,embeddings=np_embeddings)

    json_dict = {"file_name":file_name ,"npz_path":file_name_npz ,"summary":"\n ".join([i["summary"] for i in summaries])}

    write_json(json_dict)

def insert_doc():
    filepath = filedialog.askopenfilename()

    if not filepath:
        print("No file selected")
    else:
        read_docs(filepath)

def read_docs():
    selected_docs = []

    while True:
        # load docs
        docs = load_json()

        if docs:
            for i,doc in enumerate(docs):
                if doc in selected_docs: continue

                print(f"{i+1}: {doc["file_name"]} \n")
        else:
            print("No documents found")

        # ask user
        user_choice = ask_user("Enter The Number Mentioned Beside Docs OF the doc you wanna attach: (or press n to add new doc) ")

        if user_choice == "n":
            insert_doc()
            continue

        if user_choice and user_choice.isdigit():
            user_choice = int(user_choice)

            if user_choice > len(docs):
                print("Invalid Input: ")
                continue

            selected_docs.append(docs[user_choice-1])

            if len(selected_docs) < max_attachments:
                want_another_doc = ask_user("Want To Add another document? (0 for NO or 1 for YES) ")
                if want_another_doc == "0":
                    break
                elif want_another_doc == "1":
                    continue
                else:
                    print("Invalid Input: ")
                    continue
            else:
                print("Max Attachments Reached :( ")
                break
        else:
            print("Invalid Input: ")
            continue


    # return matched paper

# add_doc("../Data/docs/actual_docs/attentionpaper.pdf")
read_docs()
