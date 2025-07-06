from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd

# Define a custom cross-validator for our crypto ml dataset
class TimeSeriesSplitByToken(BaseCrossValidator):
    
    def __init__(self, n_splits=3, date_series=None, token_series=None):
        self.n_splits = n_splits
        self.date_series = date_series
        self.token_series = token_series

    def split(self, X, y=None, groups=None):
        unique_tokens = self.token_series.unique().tolist()
        self.date_series = pd.to_datetime(self.date_series)

        # Determine global date ranges for splits across all tokens
        global_dates = sorted(self.date_series.unique().tolist())
        split_size = len(global_dates) // (self.n_splits + 1)

        # Generate train and validation indices for each split
        for i in range(1, self.n_splits + 1):
            train_start_date = global_dates[0]
            train_end_date = global_dates[i * split_size]
            val_start_date = train_end_date
            val_end_date = global_dates[min(len(global_dates) - 1, (i + 1) * split_size)]

            train_indices = []
            val_indices = []

            # Collect indices for each token
            for token in unique_tokens:
                token_mask = self.token_series == token

                train_mask = (
                    token_mask &
                    (self.date_series >= train_start_date) &
                    (self.date_series < train_end_date)
                )

                val_mask = (
                    token_mask &
                    (self.date_series >= val_start_date) &
                    (self.date_series < val_end_date)
                )

                # Get row positions (not indices) for each token
                train = np.where(train_mask)[0].tolist()
                val = np.where(val_mask)[0].tolist()

                if len(train) > 0 and len(val) > 0:
                    train_indices.extend(train)
                    val_indices.extend(val)

            # Yield the indices as numpy arrays
            yield np.array(train_indices), np.array(val_indices)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class GlobalTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=5, time_col='time_period_end'):
        self.n_splits = n_splits
        self.time_col = time_col

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if self.time_col not in X.columns:
            raise ValueError(f"Missing time_col '{self.time_col}' in X")

        # Ensure datetime
        X = X.copy()
        X[self.time_col] = pd.to_datetime(X[self.time_col])
        X = X.sort_values(self.time_col)

        # Get sorted unique timestamps
        unique_times = X[self.time_col].sort_values().unique()
        n_timestamps = len(unique_times)

        if self.n_splits >= n_timestamps:
            raise ValueError("n_splits must be less than number of unique timestamps")

        # Calculate fold sizes
        fold_size = n_timestamps // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end_time = unique_times[fold_size * (i + 1) - 1]
            test_start_time = unique_times[fold_size * (i + 1)]
            test_end_time = unique_times[fold_size * (i + 2) - 1]

            train_idx = X[X[self.time_col] <= train_end_time].index
            test_idx = X[(X[self.time_col] > test_start_time) & (X[self.time_col] <= test_end_time)].index

            yield train_idx, test_idx