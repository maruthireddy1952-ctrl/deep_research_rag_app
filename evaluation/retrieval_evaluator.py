import requests


def evaluate_retrieval(question, chunks):

    context = "\n".join(chunks)

    prompt = f"""
You are a strict retrieval evaluator.

Question:
{question}

Retrieved documents:
{context}

Rules:
- If the documents clearly contain information needed to answer the question → YES
- If the documents are only loosely related → NO
- If the documents do NOT contain enough information → NO

Answer with ONLY one word:
YES or NO.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json()["response"].strip().lower()

    return "yes" in answer