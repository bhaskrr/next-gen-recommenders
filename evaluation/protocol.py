import pandas as pd


class EvaluationProtocol:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        ground_truth,
        user_train_items,
        k=10,
    ):
        self.train_df = train_df
        self.test_df = test_df
        self.ground_truth = ground_truth
        self.user_train_items = user_train_items
        self.k = k
