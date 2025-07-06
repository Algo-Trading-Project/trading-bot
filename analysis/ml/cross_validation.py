from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd


class ChronoKFold(BaseCrossValidator):
    """
    Time-aware K-fold splitter that yields non-overlapping, consecutive
    test blocks so that every sample appears **exactly once** in a test fold.
    Train folds contain *all* data strictly earlier than the test block.

    Parameters
    ----------
    n_splits : int, default=5
        Number of consecutive folds.
    """

    def __init__(self, n_splits: int = 5):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits

    # ------------------------------------------------------------------ #
    #  Required API
    # ------------------------------------------------------------------ #
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Parameters
        ----------
        X : pd.DataFrame or array-like with DatetimeIndex
        groups : 1-d array-like of datetimes (optional)

        Yields
        ------
        train_idx, test_idx : np.ndarray
            The row positions for the train / test set of that fold.
        """
        # ------------------------------------------------------------------
        # 1. Extract the time axis we will use for ordering
        # ------------------------------------------------------------------
        if groups is not None:                       # preferred by scikit-learn
            times = pd.to_datetime(np.asarray(groups))
            if times.shape[0] != len(X):
                raise ValueError("groups length does not match X")
        else:
            if not isinstance(X, (pd.DataFrame, pd.Series)):
                raise ValueError(
                    "`X` must be a pandas object if `groups` is not given"
                )
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError(
                    "`X.index` must be DatetimeIndex (or pass `groups`)"
                )
            times = X.index.to_numpy()

        # ------------------------------------------------------------------
        # 2.  Sort indices by time  (stable sort keeps original order inside ties)
        # ------------------------------------------------------------------
        order = np.argsort(times, kind="stable")
        ordered_idx = np.arange(len(times))[order]      # numeric positions
        n_samples = len(times)

        # ------------------------------------------------------------------
        # 3.  Build fold boundaries
        # ------------------------------------------------------------------
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1  # distribute remainder

        start = 0
        boundaries = []
        for size in fold_sizes:
            stop = start + size
            boundaries.append((start, stop))          # slice on ordered array
            start = stop

        # ------------------------------------------------------------------
        # 4.  Yield
        # ------------------------------------------------------------------
        for k, (test_start, test_stop) in enumerate(boundaries):
            test_slice = ordered_idx[test_start:test_stop]
            train_slice = ordered_idx[:test_start]     # all earlier samples

            if train_slice.size == 0:                  # first fold has no train
                # scikit-learn estimators expect at least 1 train sample.
                # skip the first fold altogether
                continue

            yield train_slice, test_slice