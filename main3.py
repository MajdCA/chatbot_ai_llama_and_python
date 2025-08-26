from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_community.embeddings import HuggingFaceEmbeddings
# ✅ NEW
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
import json
import time
import os
os.environ["USER_AGENT"] = "geo-assistant/1.0"

# List of URLs to load documents from
urls = [
    "https://www.aseltd.eu/en/",
    "https://www.aseltd.eu/en/about/team/",
    "https://www.aseltd.eu/en/about/privacy-policy/",
    "https://www.aseltd.eu/en/products/cir-array/",
    "https://www.aseltd.eu/en/products/rad-array/",
    "https://www.aseltd.eu/en/products/preconv-array/",
    "https://www.aseltd.eu/en/products/tilt-meter/",
    "https://www.aseltd.eu/en/products/vertical-array/",
    "https://www.aseltd.eu/en/products/vertical-array-structure/",
    "https://www.aseltd.eu/en/products/horizontal-array/",
    "https://www.aseltd.eu/en/products/therm-array/",
    "https://www.aseltd.eu/en/products/dfence/",
    "https://www.aseltd.eu/en/products/musa/",
    "https://www.aseltd.eu/en/products/tlp-one/",
    "https://www.aseltd.eu/en/products/g201/",
    "https://www.aseltd.eu/en/products/g301/",
    "https://www.aseltd.eu/en/products/g802/",
    "https://www.aseltd.eu/en/products/gmux/",
    "https://www.aseltd.eu/en/products/dmux/",
    "https://www.aseltd.eu/en/products/mind/",
    "https://www.aseltd.eu/en/products/geo-atlas/",
    "https://www.aseltd.eu/en/products/ase-mind/",
    "https://www.aseltd.eu/en/projects/",
    "https://www.aseltd.eu/en/about/publications/",
    "https://www.aseltd.eu/en/contacts/contact/",
    "https://www.aseltd.eu/en/contacts/webinar/",
    "https://www.aseltd.eu/en/projects/cancano/",
    "https://www.aseltd.eu/en/sectors/landslides/",
    "https://www.aseltd.eu/en/sectors/underground-excavations/",
    "https://www.aseltd.eu/en/sectors/structures/",
    "https://www.aseltd.eu/en/projects/agna/",
    "https://www.aseltd.eu/en/projects/bologna/",
    "https://www.aseltd.eu/en/projects/floronzo-tunnel/",
    "https://www.aseltd.eu/en/projects/padova/",
    "https://www.aseltd.eu/en/projects/peschiera/",
    "https://www.aseltd.eu/en/projects/turow/",
    "https://www.aseltd.eu/en/projects/sanbenedetto/",
    "https://www.aseltd.eu/en/projects/tav-napoli-bari/",
    "https://www.aseltd.eu/en/projects/carozzo/",
    "https://www.aseltd.eu/en/projects/maranello/",
    "https://www.aseltd.eu/en/projects/ss652-fondovalle-sangro/",
    "https://www.aseltd.eu/it/",
    "https://www.aseltd.eu/it/about/team/",
    "https://www.aseltd.eu/it/about/privacy-policy/",
    "https://www.aseltd.eu/it/products/cir-array/",
    "https://www.aseltd.eu/it/products/rad-array/",
    "https://www.aseltd.eu/it/products/preconv-array/",
    "https://www.aseltd.eu/it/products/tilt-meter/",
    "https://www.aseltd.eu/it/products/vertical-array/",
    "https://www.aseltd.eu/it/products/vertical-array-structure/",
    "https://www.aseltd.eu/it/products/horizontal-array/",
    "https://www.aseltd.eu/it/products/therm-array/",
    "https://www.aseltd.eu/it/products/dfence/",
    "https://www.aseltd.eu/it/products/musa/",
    "https://www.aseltd.eu/it/products/tlp-one/",
    "https://www.aseltd.eu/it/products/g201/",
    "https://www.aseltd.eu/it/products/g301/",
    "https://www.aseltd.eu/it/products/g802/",
    "https://www.aseltd.eu/it/products/gmux/",
    "https://www.aseltd.eu/it/products/dmux/",
    "https://www.aseltd.eu/it/products/mind/",
    "https://www.aseltd.eu/it/products/geo-atlas/",
    "https://www.aseltd.eu/it/products/ase-mind/",
    "https://www.aseltd.eu/it/projects/",
    "https://www.aseltd.eu/it/about/publications/",
    "https://www.aseltd.eu/it/contacts/contact/",
    "https://www.aseltd.eu/it/contacts/webinar/",
    "https://www.aseltd.eu/it/projects/cancano/",
    "https://www.aseltd.eu/it/sectors/landslides/",
    "https://www.aseltd.eu/it/sectors/underground-excavations/",
    "https://www.aseltd.eu/it/sectors/structures/",
    "https://www.aseltd.eu/it/projects/agna/",
    "https://www.aseltd.eu/it/projects/bologna/",
    "https://www.aseltd.eu/it/projects/floronzo-tunnel/",
    "https://www.aseltd.eu/it/projects/padova/",
    "https://www.aseltd.eu/it/projects/peschiera/",
    "https://www.aseltd.eu/it/projects/turow/",
    "https://www.aseltd.eu/it/projects/sanbenedetto/",
    "https://www.aseltd.eu/it/projects/tav-napoli-bari/",
    "https://www.aseltd.eu/it/projects/carozzo/",
    "https://www.aseltd.eu/it/projects/maranello/",
    "https://www.aseltd.eu/it/projects/ss652-fondovalle-sangro/",
    "https://www.aseltd.eu/pl/",
    "https://www.aseltd.eu/pl/about/team/",
    "https://www.aseltd.eu/pl/about/privacy-policy/",
    "https://www.aseltd.eu/pl/products/cir-array/",
    "https://www.aseltd.eu/pl/products/rad-array/",
    "https://www.aseltd.eu/pl/products/preconv-array/",
    "https://www.aseltd.eu/pl/products/tilt-meter/",
    "https://www.aseltd.eu/pl/products/vertical-array/",
    "https://www.aseltd.eu/pl/products/vertical-array-structure/",
    "https://www.aseltd.eu/pl/products/horizontal-array/",
    "https://www.aseltd.eu/pl/products/therm-array/",
    "https://www.aseltd.eu/pl/products/dfence/",
    "https://www.aseltd.eu/pl/products/musa/",
    "https://www.aseltd.eu/pl/products/tlp-one/",
    "https://www.aseltd.eu/pl/products/g201/",
    "https://www.aseltd.eu/pl/products/g301/",
    "https://www.aseltd.eu/pl/products/g802/",
    "https://www.aseltd.eu/pl/products/gmux/",
    "https://www.aseltd.eu/pl/products/dmux/",
    "https://www.aseltd.eu/pl/products/mind/",
    "https://www.aseltd.eu/pl/products/geo-atlas/",
    "https://www.aseltd.eu/pl/products/ase-mind/",
    "https://www.aseltd.eu/pl/projects/",
    "https://www.aseltd.eu/pl/about/publications/",
    "https://www.aseltd.eu/pl/contacts/contact/",
    "https://www.aseltd.eu/pl/contacts/webinar/",
    "https://www.aseltd.eu/pl/projects/cancano/",
    "https://www.aseltd.eu/pl/sectors/landslides/",
    "https://www.aseltd.eu/pl/sectors/underground-excavations/",
    "https://www.aseltd.eu/pl/sectors/structures/",
    "https://www.aseltd.eu/pl/projects/agna/",
    "https://www.aseltd.eu/pl/projects/bologna/",
    "https://www.aseltd.eu/pl/projects/floronzo-tunnel/",
    "https://www.aseltd.eu/pl/projects/padova/",
    "https://www.aseltd.eu/pl/projects/peschiera/",
    "https://www.aseltd.eu/pl/projects/turow/",
    "https://www.aseltd.eu/pl/projects/sanbenedetto/",
    "https://www.aseltd.eu/pl/projects/tav-napoli-bari/",
    "https://www.aseltd.eu/pl/projects/carozzo/",
    "https://www.aseltd.eu/pl/projects/maranello/",
    "https://www.aseltd.eu/pl/projects/ss652-fondovalle-sangro/",
]

# Load documents from the URLs
print("Loading documents from URLs...")
start_time = time.time()
docss = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docss for item in sublist]
end_time = time.time()
print(f"Loaded {len(docs_list)} documents in {end_time - start_time:.2f} seconds.")

# Initialize a text splitter with specified chunk size and overlap
print("Splitting documents into chunks...")
start_time = time.time()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
print("Splitting documents...")
# Split the documents into chunks
docs= text_splitter.split_documents(docs_list)
print(f"Split into {len(docs)} chunks.")
end_time = time.time()
print(f"Document splitting completed in {end_time - start_time:.2f} seconds.")

# Load embedding model
print("Loading embedding model and creating vector store...")
start_time = time.time()
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs
)
end_time = time.time()
print(f"Embedding model loaded in {end_time - start_time:.2f} seconds.")
print("Creating FAISS vector store...")
start_time = time.time()
# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)
end_time = time.time()
print(f"FAISS vector store created in {end_time - start_time:.2f} seconds.")
# Save and reload the vector store
start_time = time.time()
print("Saving FAISS vector store locally...")
vectorstore.save_local("faiss_index_")
persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)
print("FAISS vector store saved and reloaded.")
end_time = time.time()
print(f"FAISS vector store save and reload completed in {end_time - start_time:.2f} seconds.")
# Create a retriever
print("Creating retriever from vector store...")
start_time = time.time()
retriever = persisted_vectorstore.as_retriever(search_kwargs={"k": 1})
end_time = time.time()
print(f"Retriever created in {end_time - start_time:.2f} seconds.")
#from langchain_community.llms import Ollama

# Initialize the LLaMA model
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0,
    max_tokens=200
)

# Test with a sample prompt
print("Testing LLaMA model with a sample prompt...")
start_time = time.time()
response = llm.invoke("what is your name ?")
print(response)
end_time = time.time()
print(f"LLaMA model responded in {end_time - start_time:.2f} seconds.")

# Create RetrievalQA
print("Creating RetrievalQA chain...")
start_time = time.time()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
end_time = time.time()
print(f"RetrievalQA chain created in {end_time - start_time:.2f} seconds.")
# Interactive query loop
print("You can now ask questions about the content from the provided URLs.")
while True:
    query = input("Type your query (or type 'Exit' to quit): \n")
    if query.lower() == "exit":
        break
    print("Retrieving relevant documents and generating answer...")
    start_time = time.time()
    
    retrieved_docs = retriever.get_relevant_documents(query)
    end_time = time.time()
    print(f"Retrieved {len(retrieved_docs)} documents in {end_time - start_time:.2f} seconds.")
    # Prepare data to save
    data_to_save = []
    for i, doc in enumerate(retrieved_docs):
        data_to_save.append({
            "index": i+1,
            "source": doc.metadata.get("source", "no source"),
            "content": doc.page_content[:1000]  # first 1000 chars
        })

    # Save to JSON file (append mode)
    with open("retrieved_docs.json", "a", encoding="utf-8") as f:
        json.dump({"query": query, "retrieved_docs": data_to_save}, f, ensure_ascii=False, indent=2)
        f.write("\n")  # separate entries by new line
    start_time = time.time()
    print("Generating answer...")
    result = qa.run(query)
    end_time = time.time()
    print(f"Answer generated in {end_time - start_time:.2f} seconds.")
    print(result)