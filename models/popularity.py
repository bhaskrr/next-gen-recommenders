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