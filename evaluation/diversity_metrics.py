import numpy as np

# Intra-list diversity (ILD) metrics measure the variety of items in a single, top-N recommendation list
# to prevent redundant, monotonous suggestions


def intra_list_diversity(recommended, item_similarity_matrix, k):
    """
    recommended: ranked list of item IDs
    item_similarity_matrix: dict {(i,j): similarity_score}
    k: cutoff

    Returns: ILD score for a single user
    """
    recommended_k = recommended[:k]

    if len(recommended_k) <= 1:
        return 0.0

    total_dissimilarity = 0.0
    count = 0

    for i in range(len(recommended_k)):
        for j in range(i + 1, len(recommended_k)):
            item_i = recommended_k[i]
            item_j = recommended_k[j]

            sim = item_similarity_matrix.get(
                (item_i, item_j), item_similarity_matrix.get((item_j, item_i), 0)
            )

            total_dissimilarity += 1 - sim
            count += 1

    return total_dissimilarity / count if count > 0 else 0.0


# Popularity bias in recommender systems is the tendency to disproportionately recommend a small number of highly popular items,
# neglecting the vast, less-popular "long-tail" items. This stems from models learning from skewed historical data,
# often resulting in lower-quality, less diverse recommendations that create a reinforcing feedback loop for popular content.


def popularity_bias_at_k(predictions, item_popularity, k):
    """
    Returns average popularity of recommended items
    """
    popularity_scores = []

    for recommended in predictions.values():
        recommended_k = recommended[:k]

        for item in recommended_k:
            pop = item_popularity.get(item, 0)
            popularity_scores.append(pop)

    return np.mean(popularity_scores)
