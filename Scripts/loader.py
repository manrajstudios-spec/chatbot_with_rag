import os
import json
import spacy
import tiktoken
import numpy as np
from openai import OpenAI
from keybert import KeyBERT
from dotenv import load_dotenv
from rich.console import Console
# from transformers import AutoTokenizer,AutoModelForSequenceClassification

enc = tiktoken.get_encoding("cl100k_base")
console = Console()
load_dotenv()
ky = KeyBERT()
api_key = os.getenv("groq_api")

nlp = spacy.load('en_core_web_sm')
nlp.enable_pipe('senter')

groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

embedding_model = "embeddinggemma:300m"
main_model = "openai/gpt-oss-120b"

# tokenizer = AutoTokenizer.from_pretrained("/home/manraj_studios/PycharmProjects/Yuzu-Ai-Companion/Model/intent_tokenizer")
# model = AutoModelForSequenceClassification.from_pretrained("/home/manraj_studios/PycharmProjects/Yuzu-Ai-Companion/Model/intent_clf")

def count_tokens(text):
    return len(enc.encode(text))

def make_sentences(text):
    sents = nlp(text).sents
    return [s.text.strip() for s in sents]

def extract_keywords(text):
    keys = ky.extract_keywords(text,stop_words="english",use_mmr=True)
    return [key[0] for key in keys]

def get_embedding(text_list):
    text_len = count_tokens(" ".join(text_list))

    response = ollama_client.embeddings.create(model=embedding_model,input=text_list)

    embeddings = [item.embedding for item in response.data]

    return embeddings

def get_response(query,sys_prompt,parse=True):
    msg = [{"role":"system","content":sys_prompt},{"role":"user","content":query}]

    response = groq_client.chat.completions.create(model=main_model,messages=msg)

    if parse:
        return json.loads(response.choices[0].message.content.strip())
    else:
        return response.choices[0].message.content.strip()
