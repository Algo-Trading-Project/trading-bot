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
        global_dates = self.date_series.sort_values().unique()
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

                # Append the row positions (not indices) to the split lists
                train_indices.extend(np.where(train_mask)[0].tolist())
                val_indices.extend(np.where(val_mask)[0].tolist())

            # Yield the indices as numpy arrays
            yield np.array(train_indices), np.array(val_indices)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits