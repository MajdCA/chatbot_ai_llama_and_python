import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
import difflib
import json
from pathlib import Path

HISTORY_FILE = Path("conversation_history.json")
PREPARED_FILE = "prepared_answers.json"
################################################### load prompts #######################################################
def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

SYSTEM = read("prompts/system.md")
GLOSSARY = read("prompts/glossary.md")
EXAMPLES = read("prompts/examples.md")




model = OllamaLLM(model="llama3.1")
template="""
Answer the question below shortly and fast .

Here's the conversation history:  {context}

Question: {question}

Answer: 

 
 
"""


#prompt=ChatPromptTemplate.from_template(template)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"{SYSTEM}\n\nDomain glossary:\n{GLOSSARY}\n\nExamples:\n{EXAMPLES}"
    ),
    MessagesPlaceholder(variable_name="context"),
    ("user", "{question}")
])

chain= prompt | model

####################################################### functions #######################################################

def handle_conversation():
    # Load curated history
    clean_history = load_clean_history()
    # Load prepared answers
    prepared_answers = load_prepared_answers()

    # Convert JSON into LangChain format
    context = []
    for turn in clean_history[-10:]:  # only keep last 10 exchanges
        context.append({"role": "user", "content": turn["user"]})
        context.append({"role": "assistant", "content": turn["assistant"]})

    print("Starting conversation with GeoAtlas Assistant...")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Ending conversation.")
            break
            
        # Check for prepared answers
        prepared_answer = find_prepared_answer(user_input, prepared_answers)
        if prepared_answer:
            print(f"Assistant (prepared): {prepared_answer}")
            context += f"\nUser: {user_input}\nAssistant: {prepared_answer}\n"
            continue
        # Generate response via LLM if no prepared answer found
        start = time.time()
        response = chain.invoke({"context": context, "question": user_input})
        print(f"Assistant: {response}")

        # Save turn to JSON file
        save_turn(user_input, str(response))

        # Update runtime context
        context.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": str(response)}
        ])

        print(f"(Response time: {time.time() - start:.2f}s)")



def save_turn(user_input, assistant_reply):  # Save each turn of the conversation to a JSON file to then fine-tune the model manually ( i will delete the wrong answers from the file manually)
    data = []
    if HISTORY_FILE.exists():
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    data.append({"user": user_input, "assistant": assistant_reply})
    HISTORY_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_clean_history(): # Load history and handle empty or corrupted files
    if HISTORY_FILE.exists():
        content = HISTORY_FILE.read_text(encoding="utf-8").strip()
        if not content:
            return []
        return json.loads(content)
    return []

def load_prepared_answers():
    try:
        with open(PREPARED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    
def find_prepared_answer(user_input: str, prepared_answers, threshold=0.7):
    questions = [f["question"] for f in prepared_answers]
    closest = difflib.get_close_matches(user_input, questions, n=1, cutoff=threshold)
    if closest:
        for f in prepared_answers:
            if f["question"] == closest[0]:
                return f["answer"]
    return None








#response = chain.invoke( {"context":"" , "question":"What is the capital of France?"})

####################################################### main #######################################################

if __name__ == "__main__":

    handle_conversation()


