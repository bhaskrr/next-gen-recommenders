# Precision@K=∣Recommended ∩ Relevant∣/K
def precision_at_k(recommended, relevant, k):
    """
    recommended: list of recommended items IDs (ranked)
    relevant: set of relevant item IDs
    k: cutoff

    Returns: Precision@K
    """

    if k == 0:
        return 0.0

    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return hits / k


# Recall@K=∣Recommended ∩ Relevant∣/∣Relevant∣
def recall_at_k(recommended, relevant, k):
    """
    recommended: list of recommended item IDs (ranked)
    relevant: set of relevant item IDs
    k: cutoff

    Returns: Recall@K
    """

    if k == 0:
        return 0.0

    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return hits / len(relevant)
