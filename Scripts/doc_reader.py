import os
import re
import json
import pdfplumber
import numpy as np
import spacy
from openai import OpenAI
from dotenv import load_dotenv
from tkinter import Tk,filedialog
from main_flow import extract_key_words

root = Tk()
root.withdraw()

nlp = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])
nlp.enable_pipe('senter')

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

def load_npz(path):
    return np.load(path, allow_pickle=True)

def get_summary(query):
    sys_prompt = """You are a document summarization assistant.

TASK:
You will receive multiple document chunks, each labeled as [CHUNK N].
Summarize EACH chunk SEPARATELY. Do NOT merge chunks together.
Return ONLY a single valid JSON array with exactly as many objects as there are chunks.

OUTPUT SCHEMA (must match exactly):
[
  {
    "chunk_id": 1,
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
7. "summary" must contain 2-4 sentences per chunk.
8. Preserve important facts, names, dates, numbers, and technical terms.
9. "key_words" must contain 3-7 keywords taken from that specific chunk.
10. Do not invent information not present in the document.
11. Do NOT merge, combine, or skip any chunk.
12. Every [CHUNK N] label must have exactly one corresponding object in the array.
13. The array length must equal the number of chunks provided.

EXAMPLE INPUT:
[CHUNK 1]
Apple announced a new AI model in June 2025.

[CHUNK 2]
The model was trained on a large dataset and outperforms previous benchmarks.

EXAMPLE OUTPUT:
[
  {
    "chunk_id": 1,
    "summary": "Apple announced a new AI model in June 2025. The model will be integrated into several products.",
    "key_words": ["Apple", "AI model", "June 2025"]
  },
  {
    "chunk_id": 2,
    "summary": "The model was trained on a large dataset and achieves strong benchmark performance.",
    "key_words": ["training", "dataset", "benchmarks"]
  }
]
    """

    m = [{"role":"system","content":f"{sys_prompt}"},{"role":"user","content":f"Doc Chunks: {query}"}]
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

def make_batch(chunks):
    len_batch = 3

    batches = []

    index = 0
    for i in range(len(chunks)):
        if i == index:
            index += len_batch
            batch_text = ""
            for k in range(len_batch):
                if i + k < len(chunks):
                    batch_text += f"Chunk {k}: {chunks[i + k]} \n"

            batches.append(batch_text)

    return batches

def make_summaries(batches):
    summaries = []
    for batch in batches:
        s = get_summary(batch)
        summaries.append(s)

    s = [i for summary in summaries for i in summary]

    return s

def insert_doc():
    filepath = filedialog.askopenfilename()

    if not filepath:
        print("No file selected")
    else:
        with_summary = False
        ask_with_summary = ask_user("Do You Want To Save Summary of doc OR actual doc (Press 0 for Actual 1 For Summary) (Note Summaries can be Wrong Or might not cover details): ")

        if ask_with_summary == "0":
            with_Summary = False
        else:
            with_Summary = True

        add_doc(filepath,with_summary)

def make_chunks(t):
    paras = t.split("\n")
    chunks = []
    max_limit = 2000
    min_limit = 1700
    cur_chunk = ""
    n = 2

    for para in paras:
        text = (cur_chunk if cur_chunk else "") + para

        if len(text) < min_limit:
            cur_chunk += " " + para
            continue

        if len(text) <= max_limit:
            chunks.append(text)
            cur_chunk = ""
        else:
            splits = nlp(text)
            sentencez = [s.text.strip() for s in splits.sents]

            count = 0
            first_half = []
            second_half = []

            for i,sent in enumerate(sentencez):
                if len(sent) + count >= max_limit:
                    first_half.append(sent)
                    second_half = first_half[-n:] + sentencez[i+1:]
                    break
                else:
                    first_half.append(sent)
                    count += len(sent)

            chunks.append(" ".join(first_half))
            cur_chunk = " ".join(second_half)

    if cur_chunk:
        chunks.extend(make_chunks(cur_chunk))

    return chunks

def add_doc(doc_path,with_summary=False):
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
    batches = make_batch(chunks)
    key_words = []
    embeddings = []
    summaries = []
    if with_summary:
        summaries = make_summaries(batches)

        for summary in summaries:
            e = get_embedding(summary["summary"])
            embeddings.append(e)

            if not summary["key_words"]:
                key_words.append([])
            else:
                key_words.append(summary["key_words"])
    else:
        for chunk in chunks:
            key_words.extend(extract_key_words(chunk))
            e = get_embedding(chunk)
            embeddings.append(e)


    np_keywords = np.array(key_words,dtype=object)
    print(np_keywords)
    np_embeddings = np.array(embeddings,dtype=np.float32)

    file_name_npz = f"../Data/docs/npz_files/{file_name}.npz"
    mean_embedding = np.mean(np_embeddings,axis=0)
    np.savez(file_name_npz,key_words=np_keywords,embeddings=np_embeddings,mean_embedding=mean_embedding)

    json_dict = {"file_name":file_name ,"npz_path":file_name_npz ,"doc_data":"\n ".join([i["summary"] for i in summaries]) if with_summary else "\n ".join([i for i in chunks])}

    write_json(json_dict)

def select_docs():
    selected_docs = []

    while True:
        # load docs
        docs = load_json()
        showed = False
        to_ask = "Enter The Number Mentioned Beside Docs OF the doc you wanna attach: (or press n to add new doc) or press q to quit doc process: "
        options = []

        if docs:
            if len(selected_docs) > max_attachments:
                print("Max Attachments Reached :(")
                break

            for i,doc in enumerate(docs):
                if doc in selected_docs: continue
                showed = True
                options.append(i+1)
                print(f"{i+1}: {doc["file_name"]} \n")
        else:
            print("No documents found")

        if not showed:
            print("No documents found")
            to_ask = "Press n to add new doc: Press q to quit: "

        # ask user
        user_choice = ask_user(to_ask)

        if user_choice == "n":
            insert_doc()
            continue

        if user_choice == "q":
            return selected_docs

        if user_choice and user_choice.isdigit() or user_choice not in options:
            user_choice = int(user_choice)

            if user_choice > len(docs) :
                print("Invalid Input: ")
                continue

            selected_docs.append(docs[user_choice-1])

            print("Doc Added")

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
            print("Invalid Input: ")
            continue

    return selected_docs

def load_docs():
    selected_docs = select_docs()

    if not selected_docs:
        return []

    loaded_docs = []

    for doc in selected_docs:
        npz_file = load_npz(doc["npz_path"])
        loaded_docs.append({"file_name":doc["file_name"],"doc_data":doc["doc_data"],"key_words":npz_file["key_words"],"embeddings":npz_file["embeddings"],"mean_embedding":npz_file["mean_embedding"]})

    return loaded_docs

def unload_docs(loaded_docs):
    while True:
        if not loaded_docs:
            return []

        for i,doc in enumerate(loaded_docs):
            print(f"{i+1}: {doc['file_name']} \n")

        u_i = ask_user("Enter The Number of file You wanna remove or press q to quit: ")

        if u_i == "q":
            return loaded_docs

        if u_i and u_i.isdigit():
            if int(u_i) > len(loaded_docs) or int(u_i) <= 0:
                print("Invalid Input: ")
                continue

            loaded_docs.pop(int(u_i) - 1)

def compare_msg(msg,loaded_docs,k):
    embedded_msg = np.array(get_embedding(msg),dtype=np.float32).flatten()
    inner_threshold = 0.45
    key_words = extract_key_words(msg)
    key_words = [i[0] for i in key_words]
    print(key_words)

    data_kept = []

    for i,doc in enumerate(loaded_docs):
        chunk_embeddings = doc["embeddings"]
        chunk_keys = doc["key_words"]
        chunk_data = doc["doc_data"].split("\n")

        similarity = np.dot(chunk_embeddings,embedded_msg)/(np.linalg.norm(chunk_embeddings,axis=1) * np.linalg.norm(embedded_msg))
        topic_score = [sum(1 for x in key_words if x in chunk_key)/len(chunk_key) if chunk_key else 0 for chunk_key in chunk_keys]
        topic_score = np.array(topic_score,dtype=np.float32)

        combined = similarity + topic_score
        combined_indices = np.argsort(combined)[::-1][:min(k,len(combined))]
        print(combined)

        passing_indices = [i for i in combined_indices if combined[i] >= inner_threshold]
        print(passing_indices)

        for i in passing_indices:
            data_kept.append(chunk_data[i])

    return data_kept