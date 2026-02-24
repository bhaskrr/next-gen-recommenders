import numpy as np


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


# DCG calculates the cumulative gain (relevance score) of items, but discounts it if the item appears lower in the list.
# This reflects that users are more likely to interact with top-ranked items.


def dcg_at_k(recommended_scores, k):
    """
    Computes the Discounted Cumulative Gain (DCG) at a specific rank.

    DCG measures the effectiveness of a ranking system by summing the
    relevance scores of items, penalizing (discounting) those that appear
    lower in the list. In security mining, this helps evaluate how well
    a model ranks critical threats versus low-priority noise.

    The formula used is:
    $$DCG_{k} = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

    Args:
        recommended_scores (array-like): A list or array of relevance scores
            (e.g., [3, 2, 0, 1]). Higher values represent higher relevance.
        k (int): The number of top-ranked items to consider in the calculation.

    Returns:
        float: The calculated DCG value. Returns 0.0 if the input is empty
            or k is 0.

    Example:
        >>> scores = [3, 2, 3, 0, 1, 2]
        >>> dcg_at_k(scores, 3)
        12.3927...
    """
    recommended_scores = np.asfarray(recommended_scores)[:k]
    if recommended_scores.size:
        return np.sum(
            (2**recommended_scores - 1)
            / np.log2(np.arange(2, recommended_scores.size + 2))
        )
    return 0.0


def ndcg_at_k(recommended, relevant, k):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG) at rank k.

    NDCG normalizes the DCG score by the Ideal DCG (IDCG), which is the maximum
    possible DCG achievable if the most relevant items were perfectly ranked
    at the top. This provides a score between 0.0 and 1.0, making it possible
    to compare model performance across different security datasets.

    The formula used is:
    $$NDCG_{k} = \frac{DCG_{k}}{IDCG_{k}}$$

    Args:
        recommended (array-like): The relevance scores assigned by the model
            to the top-k items (e.g., [3, 2, 0, 1]).
        relevant (array-like): The ground-truth relevance scores for all
            available items, used to calculate the ideal ranking.
        k (int): The number of top-ranked items to evaluate.

    Returns:
        float: The normalized DCG value (0.0 to 1.0). Returns 0.0 if there
            are no relevant items or k is 0.

    Example:
        >>> model_scores = [3, 2, 0, 1]
        >>> ground_truth = [3, 3, 2, 2, 1]
        >>> ndcg_at_k(model_scores, ground_truth, 3)
        0.875...
    """
    # Calculate the number of relevant items to consider for the ideal ranking
    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    # IDCG assumes the highest relevance scores are at the top (1/log2(i+2))
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

    # Calculate the actual DCG using the model's recommended ranking
    actual_dcg = dcg_at_k(recommended, k)

    return actual_dcg / ideal_dcg
