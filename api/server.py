from fastapi import FastAPI
from pydantic import BaseModel

from embeddings.embedder import Embedder
from retriever.faiss_index import FaissIndex
from generator.llm import generate_answer

app = FastAPI()

embedder = Embedder()

# Example dummy data (later we load real docs)
documents = [
    "Many SaaS startups fail after Series A due to poor unit economics.",
    "A common startup failure reason is high customer acquisition cost.",
    "Startups often collapse when they scale before achieving product-market fit."
]

# Build embeddings
embeddings = embedder.embed(documents)

# Create FAISS index
index = FaissIndex(len(embeddings[0]))
index.add(embeddings, documents)


class QueryRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(query: QueryRequest):

    query_embedding = embedder.embed([query.question])[0]

    retrieved_chunks = index.search(query_embedding)

    context = "\n".join(retrieved_chunks)

    answer = generate_answer(query.question, context)

    return {
        "question": query.question,
        "answer": answer,
        "sources": retrieved_chunks
    }