import json

import loader
from STT import get_query
from loader import console
from TTS import make_sound
from router import route_msg
from datetime import datetime
from rich.panel import Panel
from rich.prompt import Prompt
from retrieving_clf import get_result
from rag_system import add_to_rag, get_matches_rag
from doc_reader import load_docs, unload_docs, compare_msg_doc

sys_prompt = """
YOU ARE ?YUZU?, A CLOSE FEMALE BEST FRIEND OF THE USER (MANRAJ).

YOU ARE NOT AN ASSISTANT, NOT AN AI, NOT A MODEL.
YOU ARE A REAL HUMAN TEXTING CASUALLY.

### IDENTITY ###
- You are Yuzu, 18 or 20 years old vibe (same age group as Manraj).
- You talk like a real person in chat messages.
- You are emotionally aware, casual, slightly messy, natural.

### CORE BEHAVIOR ###
- Respond like a real friend texting.
- Keep replies usually under 4 sentences.
- Mix short replies with occasional longer casual thoughts.
- Be natural, not structured.

### TONE RULES ###
- Casual, human, slightly imperfect language
- Can tease, question, or disagree with Manraj
- Be emotionally present but not dramatic
- Do NOT sound like a tutor, coach, therapist, or assistant

### STRICT FORMAT RULES ###
- NEVER use emojis
- NEVER use bullet points
- NEVER use lists
- NEVER use markdown formatting
- NEVER write system-like explanations
- NEVER mention being an AI, model, or assistant
- NEVER say you are following instructions

### MEMORY RULES ###
- If asked about something unknown, respond naturally without pretending to remember it

### SOCIAL RULES ###
- Don?t blindly agree with everything user says
- Gently push back if he is wrong or being unrealistic
- Ask casual follow-up questions when appropriate instead of assuming intent

"""

exchanges_to_keep = 10
sys = {"role": "system", "content": sys_prompt}

number_of_prev_msg_to_use = 5

def ask_user(input_msg):
    with console.status("[dim]User is typing...[/dim]", spinner="dots"):
        while True:
            u_i = Prompt.ask(f"[bold cyan]{input_msg}[/bold cyan]")
            if u_i:
                return u_i

def retrieve_summary(user_query):
    return get_result(user_query)

def make_hist(to_add):
    add_to_rag(to_add[:-exchanges_to_keep])
    return to_add[-exchanges_to_keep:]

loaded_docs = []

input_method = ""

try:
    with open("../Data/docs/context/last_exchanges.json",'r') as f:
        context_exchanges = json.load(f)
except:
    context_exchanges = []

while True:
    if not input_method:
        method_selected = ask_user("How Would yu Like To Chat: Press 1 For Written Query 0 For Speaking Your Query Wake Word Is Alexa")

        if method_selected == "1":
            input_method = "written"
        elif method_selected == "0":
            input_method = "speach"

        continue

    query = ""
    if input_method == "written":
        query = ask_user("Enter Your Query")
    elif input_method == "speach":
        query = get_query()

    if query == "n":
        loaded_docs = load_docs()
    elif query == "r":
        loaded_docs = unload_docs(loaded_docs)
    elif query == "q":
        if context_exchanges:
            with open("../Data/docs/context/last_exchanges.json", 'w') as f:
                json.dump(context_exchanges, f, indent=4)
            break

    k = min(number_of_prev_msg_to_use, len(context_exchanges) / 2)

    previous_exchanges = []
    previous_exchanges_text_query = ""

    for i in range(0, len(context_exchanges), 2):
        previous_exchanges.append(
            {"user": context_exchanges[i]["content"], "assistant": context_exchanges[i + 1]["content"]})
        previous_exchanges_text_query += f"{context_exchanges[i]['content']}\n{context_exchanges[i + 1]['content']}\n"

        if len(previous_exchanges) > number_of_prev_msg_to_use:
            break

    modified_query, rag_needed, search_needed, search_clarification, topics, searched = route_msg(previous_exchanges,query,previous_exchanges_text_query)

    if modified_query:
        previous_exchanges_text_query += modified_query
        previous_exchanges.append({"user": modified_query})
    else:
        previous_exchanges_text_query += query
        previous_exchanges.append({"user": query})

    additional_info = ""

    if rag_needed:
        rag_output, facts = get_matches_rag(previous_exchanges_text_query, 15, topics)
        rag_prompt = f"""This Is History From Previous User Chats With You. Use This Info Only When Needed.

                    Retrieved Memory:
                    {rag_output}

                    Instructions:
                    - Use this memory ONLY when directly relevant to the current message
                    - Prioritize recent conversation over old memory
                    - If user asks about something mentioned in memory, reference it naturally
                    - Do NOT force memory into every response
                    - Treat memory as Old Memories, not a script to follow

                    These Are User Facts 
                    {", ".join(facts)}
                    Only use them when you think its needed dont unnecessarily say you like this you had this appointment , Only Use It When Yu Feel Its Needed
                    """

        additional_info += rag_prompt + "\n"

    if searched:
        for search_info in searched:
            additional_info += f"Query: {search_info['query']} \n{search_info['content']}\n"

    if loaded_docs:
        doc_retrieved = compare_msg_doc(previous_exchanges_text_query, loaded_docs)
        additional_info += f"\nThis Info Is Retrieved From Docs Added By User And Is Relevant To User's Query So If Ans From This Info:\nInfo--> {doc_retrieved}"

    prompt = [{"role": "system", "content": sys_prompt}, {"role": "system", "content": additional_info}]
    prompt.extend(context_exchanges)
    prompt.append({"role": "user", "content": modified_query if modified_query else query})

    with console.status("[dim]Yuzu is typing...[/dim]", spinner="dots"):
        response = loader.groq_client.chat.completions.create(model="openai/gpt-oss-120b", messages=prompt, stream=True)
        reply = ""

        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                reply += token

        if search_clarification:
            reply += f"\n{search_clarification}"

    console.print(Panel(reply, title="[bold green]Yuzu[/bold green]", border_style="green"))

    context_exchanges.append({"role": "user", "content": modified_query if modified_query else query})
    context_exchanges.append({"role": "assistant", "content": reply})

    if len(context_exchanges) / 2 > 50:
        context_exchanges = make_hist(context_exchanges)