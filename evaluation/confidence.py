def compute_confidence(rerank_scores, attempts):

    if not rerank_scores:
        return 0.0

    avg_score = sum(rerank_scores) / len(rerank_scores)

    # normalize roughly
    normalized = min(max(avg_score, 0), 1)

    penalty = 1 / attempts

    confidence = normalized * penalty

    return round(confidence, 2)