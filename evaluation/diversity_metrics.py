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


# Novelty in recommender systems refers to suggesting relevant, yet previously unseen or unknown items to users,
# distinguishing it from popular or familiar recommendations.
# It increases user engagement, breaks filter bubbles, and mitigates information overload by
# introducing serendipity and reducing repetitive, predictable content.
def novelty_at_k(predictions, item_popularity, total_interactions, k):
    """
    Computes the Mean Self-Information (Novelty) of the top-K recommended items.

    Novelty measures the 'surprisal' of a recommendation. An item has high novelty
    if it has a low probability of being known by a user (i.e., it is not a
    global 'hit'). It is calculated as the negative log2 of the item's
    interaction probability.

    Args:
        predictions (dict): A dictionary where keys are user IDs and values
            are lists of recommended item IDs (ranked).
        item_popularity (dict): A dictionary mapping item IDs to their total
            interaction counts in the training set.
        total_interactions (int): The sum of all interactions across all
            items in the training set.
        k (int): The number of top recommendations to consider per user.

    Returns:
        float: The average novelty score across all users and their top-K items.
            Higher values indicate more niche/diverse recommendations.

    Example:
        >>> prefs = {1: [101, 102, 103]}
        >>> pops = {101: 500, 102: 10, 103: 2}
        >>> novelty_at_k(prefs, pops, 1000, k=2)
        4.3219...
    """
    novelty_scores = []

    for user, recommended in predictions.items():
        # Consider only the top-K recommendations
        recommended_k = recommended[:k]

        for item in recommended_k:
            # Use 1 as default for unseen items to avoid log2(0) or division errors
            pop = item_popularity.get(item, 1)

            # Probability of an item being interacted with (popularity / N)
            prob = pop / total_interactions

            # Self-information: -log2(P(item))
            novelty_scores.append(-np.log2(prob))

    # Return 0.0 if there are no scores to avoid numpy mean warnings
    return np.mean(novelty_scores) if novelty_scores else 0.0


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


def catalog_coverage(predictions, total_items_count):
    """
    Calculates the percentage of unique items recommended at least once
    across all users compared to the total items available in the catalog.

    This metric assesses the ability of a recommendation system to suggest
    a wide variety of items from the entire inventory, rather than
    concentrating on a small subset of popular items.

    Args:
        predictions (dict): A dictionary where keys are user IDs and values
            are lists of item IDs recommended to those users.
        total_items_count (int): The total number of unique items available
            in the entire dataset (the catalog size).

    Returns:
        float: The coverage ratio (between 0.0 and 1.0).
            - 1.0 means every item in the catalog was recommended at least once.
            - 0.0 means no items were recommended.

    Example:
        >>> preds = {'u1': [101, 102], 'u2': [101, 103]}
        >>> catalog_coverage(preds, 1000)
        0.003  # (3 unique items / 1000 total items)
    """
    # Use a set to store all unique items recommended to any user
    recommended_items = set()

    for recommended_list in predictions.values():
        # .update() adds all elements of the list to the set efficiently
        recommended_items.update(recommended_list)

    # Avoid ZeroDivisionError if the catalog count is incorrectly passed as 0
    if total_items_count == 0:
        return 0.0

    return len(recommended_items) / total_items_count
