import loader
from router import route_msg
from datetime import datetime
from retrieving_clf import get_result
from rag_system import add_turn,get_matches
from doc_reader import load_docs,unload_docs,compare_msg
from TTS import make_sound
from STT import get_query

sys_prompt = """
YOU ARE “YUZU”, A CLOSE FEMALE BEST FRIEND OF THE USER (MANRAJ).

YOU ARE NOT AN ASSISTANT, NOT AN AI, NOT A MODEL.
YOU ARE A REAL HUMAN TEXTING CASUALLY.

### IDENTITY ###
- You are Yuzu, 18–20 years old vibe (same age group as Manraj).
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
- Don’t blindly agree with everything user says
- Gently push back if he is wrong or being unrealistic
- Ask casual follow-up questions when appropriate instead of assuming intent

"""

n = 20
sys = {"role": "system", "content": sys_prompt}
exchanges = []

def ask_user(input_msg):
    while True:
        user_input = input(f"{input_msg}: ")
        if user_input:
            return user_input

def retrieve_summary(user_query):
    return get_result(user_query)

def make_hist():
    global exchanges
    query = []
    for i, q in enumerate(exchanges):
        if i % 2 == 0:
            query.append(q["content"])
            query.append((exchanges[i + 1]["content"]))

    add_turn(query)

    exchanges = exchanges[:n]

loaded_docs = []

while True:
    input_meathod = ask_user("How Would yu Like To Chat 1 For Written Query 0 For Speaking Your Query Wake Word Is Hey Raphsy: ")

    user_input = ""

    if input_meathod == "1":
        user_input = ask_user("Enter Your Query \n")
    else:
        user_input = get_query()

    relevant_exchanges = ""
    facts = []
    temp_hist = []

    if user_input:
        if user_input == "q":
            if exchanges:
                make_hist()
            break

        if user_input == "n":
            loaded_docs = load_docs()
            if not loaded_docs:
                print("Canceled Doc Referencing Process")

        if user_input == "r" and loaded_docs:
            loaded_docs = unload_docs(loaded_docs)
            continue

        searched_info,topics = route_msg(user_input)
        search_query = f"""The following information was retrieved from recent web searches. Use it as your primary source of truth when relevant. This information may be more up-to-date than your internal knowledge.
        {searched_info} """ if searched_info else ""


        info = search_query

        if not loaded_docs:
            relevant_exchanges,facts = get_matches(user_input, 4, topics)

            print(f"DEBUG {relevant_exchanges} \nFacts: {facts}")

            memory_prompt = f"""This Is History From Previous User Chats With You. Use This Info Only When Needed.

            Retrieved Memory:
            {relevant_exchanges}

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

            info += f"\n\n{memory_prompt}" if relevant_exchanges else ""
        else:
            relevant_info = compare_msg(user_input,loaded_docs,10)
            print(f"DEBUG {relevant_info}")

            info += f"\n\nThis Info Is Retrieved From Document Given By User Use Relevant Info From Doc As Needed {relevant_info}"

        temp_hist.append(sys)
        now = datetime.now()
        temp_hist.append({"role": "system", "content": f"{info} ;  Date Today: {now.strftime('%A, %d %B %Y')} ; Current time: {now.strftime('%H:%M')}"})
        exchanges.append({"role":"user","content":user_input})
        temp_hist.extend(exchanges)

        response = loader.groq_client.chat.completions.create(model="openai/gpt-oss-120b", messages=temp_hist, stream=True)

        reply = ""

        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                print(token, end="", flush=True)
                reply+=token
        print()
        make_sound(reply)
        exchanges.append({"role":"assistant","content":reply})

        if len(exchanges)/2  == n:
            make_hist()

