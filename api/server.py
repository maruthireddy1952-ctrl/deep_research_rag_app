from fastapi import FastAPI
from pydantic import BaseModel

from embeddings.embedder import Embedder
from retriever.faiss_index import FaissIndex
from generator.llm import generate_answer
from retriever.bm25_index import BM25Index
from data.ingest import ingest_pdf
from retriever.reranker import Reranker
from evaluation.retrieval_evaluator import evaluate_retrieval
from retriever.query_rewriter import rewrite_query
from evaluation.confidence import compute_confidence
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
    attempts = 0

    try:
        original_query = query.question

        query_embedding = embedder.embed([original_query])[0]

        retrieved = hybrid_search(original_query, query_embedding)

        reranked_chunks, scores = reranker.rerank(original_query, retrieved)

        good = evaluate_retrieval(original_query, reranked_chunks)

        if not good:
            attempts += 1
            new_query = rewrite_query(original_query)

            new_embedding = embedder.embed([new_query])[0]

            retrieved = hybrid_search(new_query, new_embedding)

            reranked_chunks, scores = reranker.rerank(new_query, retrieved)

        context = "\n".join(reranked_chunks)

        answer = generate_answer(original_query, context)

        confidence = compute_confidence(scores, attempts)

        return {
            "question": original_query,
            "answer": answer,
            "sources": reranked_chunks,
            "confidence": confidence,
            "attempts": attempts
        }

    except Exception as e:
        return {
            "error": str(e),
            "question": getattr(query, "question", None)
        }