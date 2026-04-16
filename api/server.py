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

    

    original_query = query.question

    query_embedding = embedder.embed([original_query])[0]

    retrieved_chunks = hybrid_search(original_query, query_embedding)

    reranked_chunks = reranker.rerank(original_query, retrieved_chunks)

    good = evaluate_retrieval(original_query, reranked_chunks)

    retrieval_attempts = 1
    rewritten_query = None

    if not good:

        rewritten_query = rewrite_query(original_query)

        new_embedding = embedder.embed([rewritten_query])[0]

        retrieved_chunks = hybrid_search(rewritten_query, new_embedding)

        reranked_chunks = reranker.rerank(rewritten_query, retrieved_chunks)

        retrieval_attempts += 1

    context = "\n".join(reranked_chunks)

    answer = generate_answer(original_query, context)

    return {
        "question": original_query,
        "answer": answer,
        "sources": reranked_chunks,
        "retrieval_attempts": retrieval_attempts,
        "query_rewritten": rewritten_query,
        "retrieval_success": good
    }