from fastapi import FastAPI
from pydantic import BaseModel

from embeddings.embedder import Embedder
from retriever.faiss_index import FaissIndex
from generator.llm import generate_answer
from retriever.bm25_index import BM25Index
from data.ingest import ingest_pdf
from retriever.reranker import Reranker

app = FastAPI()

pdf_chunks = ingest_pdf("data/documents/startup_report.pdf")
print(len(pdf_chunks),"holaa")
embedder = Embedder()
reranker=Reranker()
documents = pdf_chunks
embeddings = embedder.embed(documents)

index = FaissIndex(len(embeddings[0]))
index.add(embeddings, documents)

bm25_index = BM25Index(documents)


class QueryRequest(BaseModel):
    question: str


def hybrid_search(question, query_embedding):

    vector_results = index.search(query_embedding, k=3)
    keyword_results = bm25_index.search(question, k=3)

    combined = []

    for doc in vector_results:
        if doc not in combined:
            combined.append(doc)

    for doc in keyword_results:
        if doc not in combined:
            combined.append(doc)

    return combined[:5]


@app.post("/ask")
def ask_question(query: QueryRequest):

    query_embedding = embedder.embed([query.question])[0]

    # retrieved_chunks = hybrid_search(query.question, query_embedding)
    
    retrieved_chunks = hybrid_search(query.question, query_embedding)

    reranked_chunks = reranker.rerank(
            query.question,
            retrieved_chunks,
            top_k=3
        )

    context = "\n".join(reranked_chunks)
    
    answer = generate_answer(query.question, context)

    return {
        "question": query.question,
        "answer": answer,
        "sources": reranked_chunks
    }