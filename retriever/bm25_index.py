from rank_bm25 import BM25Okapi


class BM25Index:

    def __init__(self, documents):

        self.documents = documents

        tokenized_docs = [doc.split() for doc in documents]

        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query, k=5):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for idx in ranked[:k]:
            results.append(self.documents[idx])

        return results