import pandas as pd

class PopularityRecommender:
    def __init__(self):
        self.item_popularity = None
        self.ranked_items = None
    
    def fit(self, train_df: pd.DataFrame):
        """Compute item popularity from train data."""
        
        item_counts = train_df.groupby("movieId").size()
        
        # Sort items by descending popularity
        self.item_popularity = item_counts.sort_values(ascending=False)
        self.ranked_items = list(self.item_popularity.index)

    def recommend(self, user_id, user_train_items, k= 10):
        """
        Recommend top-k unseen popular items for a user.
        
        user_train_items: dict {user_id: set(items)}
        """
        seen_items = user_train_items.get(user_id, set())
        
        recommendations = []
        
        for item in self.ranked_items:
            if item not in seen_items:
                recommendations.append(item)
            if len(recommendations) >= k:
                break
                
        return recommendations
