import re
import json
import fitz
import loader
import numpy as np
from loader import console,get_embedding,get_response
from tkinter import Tk,filedialog
from graph_search import compare_embed,make_graph

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

    raw = get_response(query,sys_prompt)

    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            console.print(f"[dim]Parsing Error: {e}[/dim]")
            return []
    else:
        return []

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def read_doc(path):
    try:
        doc = fitz.open(path)

        text = ""

        for page in doc:
            text += page.get_text()

        return clean_text(text)
    except FileNotFoundError:
        return ""

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
    root = Tk()
    root.withdraw()

    filepath = filedialog.askopenfilename()

    if not filepath:
        console.print("[dim]No file selected[/dim]")
    else:
        ask_with_summary = ask_user("Do You Want To Save Summary of doc OR actual doc (Press 0 for Actual 1 For Summary) (Note Summaries can be Wrong Or might not cover details): ")

        if ask_with_summary == "0":
            with_summary = False
        else:
            with_summary = True

        add_doc(filepath,with_summary)

def make_chunks(t):
    max_limit = 1500
    min_limit = 1200
    overlap = 2

    sentences = loader.make_sentences(t)
    chunks = []
    curr_chunk = []
    curr_chunk_text = ""

    for i,sent in enumerate(sentences):
        added = curr_chunk_text + ". " + sent

        if len(added) < min_limit:
            curr_chunk_text += f". {sent}"
            curr_chunk.append(sent)

        elif len(added) < max_limit:
            chunks.append(added)
            curr_chunk.append(sent)
            curr_chunk = curr_chunk[-overlap:]
            curr_chunk_text = ". ".join(curr_chunk)

        else:
            if i == len(sentences) - 1:
                if len(sent) > 500:
                    sent = sent[:500]

                curr_chunk_text += ". " + sent
                chunks.append(curr_chunk_text)
                curr_chunk = []
                curr_chunk_text = ""
            else:
                chunks.append(curr_chunk_text)
                curr_chunk = curr_chunk[-overlap:]
                curr_chunk_text = ". ".join(curr_chunk)
                curr_chunk.append(sent)
                curr_chunk_text += f". {sent}"

    if curr_chunk_text:
        chunks.append(curr_chunk_text)

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
    else:
        console.print("[red]Unsupported file type[/red]")
        return

    with console.status("[dim]Making Chunks And Batches.. [/dim]", spinner="dots"):
        chunks = make_chunks(text)
        batches = make_batch(chunks)

    key_words = []
    embeddings = []
    summaries = []

    if with_summary:
        summaries = make_summaries(batches)

        for summary in summaries:
            e = np.array(get_embedding([summary["summary"]])[0],dtype=np.float32)
            embeddings.append(e)

            if not summary["key_words"]:
                key_words.append([])
            else:
                key_words.append(summary["key_words"])
    else:
        for chunk in chunks:
            kw = loader.extract_keywords(chunk)
            key_words.extend(kw)
            e = np.array(get_embedding([chunk + f"Keywords: {", ".join(kw)}"])[0],dtype=np.float32)
            embeddings.append(e)

    np_embeddings = np.array(embeddings,dtype=np.float32)

    json_dict = {"file_name":file_name,"doc_data":summaries if with_summary else chunks}

    make_graph(np_embeddings,file_name,True)

    write_json(json_dict)

def select_docs():
    selected_docs = []

    while True:
        docs = load_json()
        showed = False
        to_ask = "Enter The Number Mentioned Beside Docs OF the doc you wanna attach: (or press n to add new doc) or press q to quit doc process: "
        options = []

        if docs:
            if len(selected_docs) > max_attachments:
                console.print("[dim]Max Attachments Reached :([/dim]")
                break

            for i,doc in enumerate(docs):
                if doc in selected_docs: continue
                showed = True
                options.append(i+1)
                console.print(f"[dim]{i + 1}: {doc['file_name']}[/dim]")
        else:
            console.print("[dim]No documents found[/dim]")

        if not showed and docs:
            console.print("[dim]No Stored Documents Are Availabe To Add[/dim]")
            to_ask = "Press n to add new doc: Press q to quit: "

        user_choice = ask_user(to_ask)

        if user_choice == "n":
            insert_doc()
            continue

        if user_choice == "q":
            return selected_docs

        if user_choice and user_choice.isdigit() and int(user_choice) in options:
            user_choice = int(user_choice)

            if user_choice > len(docs) or user_choice <= 0:
                console.print("[dim]Invalid Input[/dim]")
                continue

            selected_docs.append(docs[user_choice-1])

            console.print("[dim]Doc Added[/dim]")

            if len(selected_docs) <= max_attachments:
                want_another_doc = ask_user("Want To Add another document? (0 for NO or 1 for YES) ")
                if want_another_doc == "0":
                    break
                elif want_another_doc == "1":
                    continue
                else:
                    console.print("[dim]Invalid Input[/dim]")
                    continue
        else:
            console.print("[dim]Invalid Input[/dim]")
            continue

    return selected_docs

def load_docs():
    selected_docs = select_docs()

    if not selected_docs:
        return []

    loaded_docs = selected_docs.copy()

    return loaded_docs

def unload_docs(loaded_docs):
    while True:
        if not loaded_docs:
            return []

        for i,doc in enumerate(loaded_docs):
            console.print(f"[dim]{i + 1}: {doc['file_name']}[/dim]")

        user_input = ask_user("Enter The Number of file You wanna remove or press q to quit: ")

        if user_input == "q":
            return loaded_docs

        if user_input and user_input.isdigit():
            if int(user_input) > len(loaded_docs) or int(user_input) <= 0:
                console.print("[dim]Invalid Input[/dim]")
                continue

            loaded_docs.pop(int(user_input) - 1)

def compare_msg_doc(msg, loaded_docs):
    key_words = loader.extract_keywords(msg)
    embedded_msg = np.array(get_embedding([msg + f"KeyWords: {", ".join(key_words)}"])[0],dtype=np.float32).flatten()

    similar_chunks = []

    with console.status("[dim]Comparing Query...[/dim]",spinner="dots"):
        for doc in loaded_docs:
            ids = compare_embed(query_embed=embedded_msg,name=doc["file_name"],depth=5)
            chunks = doc["doc_data"]
            similar_chunks.append(f"File Name: {doc['file_name']} \nData Retrieved-->")
            temp = [chunks[k] for k in ids]
            similar_chunks.extend(temp)

    return "\n".join(similar_chunks)