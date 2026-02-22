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

# In a recommendation system, it’s not just about what you recommend, but the order in which you recommend it.
# Average Precision measures two things simultaneously:
# Precision: How many of the recommended items were actually relevant?
# Order (Ranking): Were the relevant items placed at the top of the list?
def average_precision_at_k(recommended, relevant, k):
    """
    Returns average precision@K
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    
    if not relevant_set:
        return 0.0

    score = 0.0
    hits = 0

    for i, item in enumerate(recommended_k):
        if item in relevant_set:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(relevant_set), k)

# The Reciprocal Rank (RR) for a single query is simply the inverse of the rank of the first relevant item.
def rr_at_k(recommended, relevant, k):
    """
    Mean Reciprocal Rank@K
    """
    recommended_k = recommended[:k]

    for i, item in enumerate(recommended_k):
        if item in relevant:
            return 1.0 / (i + 1)

    return 0.0

