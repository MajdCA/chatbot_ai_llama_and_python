import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.1")
template="""
Answer the question below .

Here's the conversation history:  {context}

Question: {question}

Answer: 

 
 
"""


prompt=ChatPromptTemplate.from_template(template)

chain= prompt | model


def handle_conversation():
    context = ""
    print("Starting conversation...")
    print("Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending conversation.")
            break
        start_time = time.time()
        response = chain.invoke({"context": context, "question": user_input})
        print(f"Assistant: {response}")
        
        end_time = time.time()

        response_time = end_time - start_time

        print(response)
        print(f"Response time: {response_time:.4f} seconds")


        context += f"\nUser: {user_input}\nAssistant: {response}"








#response = chain.invoke( {"context":"" , "question":"What is the capital of France?"})

if __name__ == "__main__":
    handle_conversation()


