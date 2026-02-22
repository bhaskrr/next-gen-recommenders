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

# A hit occurs when an item recommended to a user in their Top-list is
# actually consumed, clicked, or highly rated by that user in the test set.
# In our case, if any of the recommended items in top k are relevant to the user, we will consider it a hit
# with a hit rate of 100% and otherwise 0%
def hit_rate_at_k(recommended, relevant, k):
    """
    Returns 1 if at least one relevant item appears in top-k
    """
    
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return 1.0 if len(hits) > 0 else 0.0