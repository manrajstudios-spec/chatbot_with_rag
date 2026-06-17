import os
import spacy
from openai import OpenAI
from rich.live import Live
from rich.panel import Panel
from keybert import KeyBERT
from dotenv import load_dotenv
from rich.console import Console
from rich.spinner import Spinner
from transformers import AutoTokenizer,AutoModelForSequenceClassification

console = Console()

load_dotenv()

ky = KeyBERT()
api_key = os.getenv("groq")

nlp = spacy.load('en_core_web_sm')
nlp.enable_pipe('senter')


groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
lm_client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm_studio")

# tokenizer = AutoTokenizer.from_pretrained("/home/manraj_studios/PycharmProjects/Yuzu-Ai-Companion/Model/intent_tokenizer")
# model = AutoModelForSequenceClassification.from_pretrained("/home/manraj_studios/PycharmProjects/Yuzu-Ai-Companion/Model/intent_clf")

def make_sentences(text):
    sents = nlp(text).sents
    return [s.text.strip() for s in sents]

def extract_keywords(text):
    keys = ky.extract_keywords(text,stop_words="english",use_mmr=True)
    return [key[0] for key in keys]