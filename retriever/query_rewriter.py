import requests


def rewrite_query(question):

    prompt = f"""
Rewrite the user query to improve document retrieval.

Original question:
{question}

Return only the rewritten query.
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": prompt,
                "stream": False
            }
        )

        data = response.json()

        print("Rewrite model output:", data)

        rewritten = data.get("response")

        if rewritten:
            return rewritten.strip()

        return question

    except Exception as e:

        print("Query rewrite failed:", e)

        return question