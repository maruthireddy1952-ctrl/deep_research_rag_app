import requests

def generate_answer(question, context):

    prompt = f"""
Answer the question using the context.

Context:
{context}

Question:
{question}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]