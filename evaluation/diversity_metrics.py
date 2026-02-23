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
