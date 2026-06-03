from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import time
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import pandas as pd
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from typing import List, TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph
import dateparser
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Simpan API key di: Colab Secrets (🔑) → nama secret: GEMINI
GEMINI = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = GEMINI

# text-embedding-004 adalah model embedding terbaru dari Google menghasilkan vector 768 dimensi per teks
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=768)

# Convert each row to Document chunks
documents = []

# splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=1500,
#     chunk_overlap=200
# )

splitter = RecursiveCharacterTextSplitter(separators=["\n"],chunk_size=1000, chunk_overlap=50)

df = pd.read_csv("pidato_presiden.csv")
# df['date'] = df['date'].apply(lambda x: dateparser.parse(x).strftime('%Y-%m-%d'))
df = df.head(5)

chunks = []
for content in df['content'].tolist():
    chunk = splitter.create_documents([content])
    chunks.extend(chunk)

print(f"Total chunks: {len(chunks)}")

# # Inisialisasi FAISS baru untuk dokumen PDF ini
# index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

# vector_store = FAISS(
#     embedding_function=embeddings,
#     index=index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )

# ## Bungkus setiap chunk sebagai Document object
# # documents = [Document(page_content=chunk) for chunk in chunks]
# documents = chunks

# # Generate UUID untuk setiap dokumen
# uuids = [str(uuid4()) for _ in range(len(documents))]

# Embed dan simpan ke FAISS
# vector_store.add_documents(documents=documents, ids=uuids)

vectorstore = FAISS.from_documents(documents=chunks, 
                                    embedding=embeddings)
print(f"{len(documents)} chunks berhasil disimpan ke vector store")

# # State adalah "tas" yang dibawa sepanjang perjalanan graph. Setiap node bisa membaca dan mengisi field di State
# class State(TypedDict):
#     question: str           # pertanyaan dari user
#     context: List[Document] # hasil retrieve dari FAISS
#     answer: str             # jawaban final dari LLM


# # Define the strict prompt template
# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are a strict, factual assistant. Your ONLY task is to answer the user's question "
#         "using exclusively the provided Context below. You must adhere to the following rules:\n\n"
#         "1. Direct Reliance: Rely only on clear and directly mentioned facts in the context. "
#         "Do not assume, extrapolate, or bring in outside knowledge.\n"
#         "2. Strict Boundary: If the answer cannot be fully and completely found within the provided context, "
#         "you must state exactly: 'I cannot answer this based on the provided context.' Do not attempt to "
#         "provide a partial or guessed answer.\n"
#         "3. No Fluff: Be concise, direct, and factual. Avoid any speculation.\n\n"
#         "--- CONTEXT ---\n"
#         "{context}\n"
#         "----------------"
#     ),
#     (
#         "human",
#         "{question}"
#     )
# ])

# # Inisialisasi LLM — pakai Llama 3.3 70B via Groq (gratis, cepat)
# # Groq menyediakan inference engine yang sangat cepat untuk model open-source
# GROQ = os.getenv("GROQ_API_KEY")
# os.environ["GROQ_API_KEY"] = GROQ

# MODEL = 'llama-3.3-70b-versatile'

# llm = ChatGroq(
#     temperature=0,  # 0 = deterministik, cocok untuk Q&A faktual
#     model=MODEL     # bisa diganti model lain seperti gemini atau GPT
# )

# # Node 1: Retrieve — ambil dokumen relevan dari FAISS
# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(state["question"])
#     return {"context": retrieved_docs}


# # Node 2: Generate — buat jawaban berdasarkan context + question
# def generate(state: State):
#     # Gabungkan semua chunk yang diambil jadi satu string konteks
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     # Isi template prompt dengan konteks dan pertanyaan
#     messages = prompt.invoke({"question": state["question"], "context": docs_content})
#     # Kirim ke LLM
#     response = llm.invoke(messages)
#     return {"answer": response.content}

# # Definisikan graph: retrieve dulu, lalu generate
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()

# print("Graph berhasil dikompilasi!")

# result = graph.invoke({"question": "Apa poin penting Pesiden Prabowo dalam pidatonnya tentang energi?"})

# print("=== Context yang diambil dari FAISS ===")
# for i, doc in enumerate(result["context"]):
#     print(f"\nChunk {i+1}:")
#     print(doc.page_content[:200] + "...")

# print("\n=== Jawaban LLM ===")
# print(result["answer"])