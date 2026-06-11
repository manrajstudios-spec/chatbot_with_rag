import os
import  spacy
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer,AutoModelForSequenceClassification

load_dotenv()

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
