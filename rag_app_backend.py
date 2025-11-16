from dotenv import load_dotenv
load_dotenv()
import os
import json
import requests
from uuid import uuid4

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_dim = len(embeddings.embed_query("hello"))

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

index = faiss.IndexFlatL2(embedding_dim)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

dummy_docs = [
    Document(page_content="LangChain is a framework for building LLM applications.", metadata={"source": "wiki"}),
    Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={"source": "rag"}),
    Document(page_content="FAISS is a high-performance vector search engine.", metadata={"source": "vector"}),
    Document(page_content="TinyLlama is a 1.1B model fine-tuned for text generation.", metadata={"source": "model"}),
]

vector_store.add_documents(dummy_docs)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

RAG_PROMPT = """
You are a helpful AI assistant.

Use ONLY the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

def call_finetuned_llm(prompt: str):
    payload = {"inputs": prompt}
    resp = requests.post(API_URL, json=payload)

    try:
        data = resp.json()
        return data.get("result", "No response found.")
    except Exception as e:
        return f"Error decoding LLM response: {str(e)}"
    
def generate_answer(question: str):

    # Retrieve
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Build prompt
    final_prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    # LLM call
    answer = call_finetuned_llm(final_prompt)

    return {
        "question": question,
        "context": context,
        "answer": answer
    }
