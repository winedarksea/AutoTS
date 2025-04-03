import numpy as np
import pandas as pd


class hierarchial(object):
    """Create hierarchial series, then reconcile.

    Currently only performs one-level groupings.
    Args:
        grouping_method (str): method to create groups. 'User' requires hier_id input of groupings.
        n_groups (int): number of groups, if above is not 'User'
        reconciliation (str): None, or 'mean' method to combine top and bottom forecasts.
        grouping_ids (dict): dict of series_id: group_id to use if grouping is 'User'
    """

    def __init__(
        self,
        grouping_method: str = 'tile',
        n_groups: int = 5,
        reconciliation: str = 'mean',
        grouping_ids: dict = None,
    ):
        self.grouping_method = str(grouping_method).lower()
        self.n_groups = n_groups
        self.reconciliation = reconciliation
        self.grouping_ids = grouping_ids

        if self.grouping_method == 'user':
            if grouping_ids is None:
                raise ValueError("grouping_ids must be provided.")

    def fit(self, df):
        """Construct and save object info."""
        # construct grouping_ids if not given
        if self.grouping_method != 'user':
            num_hier = df.shape[1] / self.n_groups
            if self.grouping_method == 'dbscan':
                X = df.mean().values.reshape(-1, 1)
                from sklearn.cluster import DBSCAN

                clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
                grouping_ids = clustering.labels_
            elif self.grouping_method == 'tile':
                grouping_ids = np.tile(
                    np.arange(self.n_groups), int(np.ceil(num_hier))
                )[: df.shape[1]]
            elif self.grouping_method == 'alternating':
                grouping_ids = np.repeat(
                    np.arange(self.n_groups), int(np.ceil(num_hier))
                )[: df.shape[1]]
            elif self.grouping_method == 'kmeans':
                from sklearn.cluster import KMeans

                X = df.mean().values.reshape(-1, 1)
                kmeans = KMeans(n_clusters=self.n_groups, random_state=0).fit(X)
                grouping_ids = kmeans.labels_
            grouping_ids = grouping_ids.astype(str).astype(np.object)
            # z is a deliberate typo to make such an id very rare in source
            grouping_ids = grouping_ids + '_hierarchy_levelz'
            grouping_ids = dict(zip(df.columns.tolist(), grouping_ids))
            self.grouping_ids = grouping_ids
        else:
            # fix missing or extra ids
            grouping_ids = {}
            for x in df.columns:
                if x not in self.grouping_ids.keys():
                    grouping_ids[x] = 'hierarchy_levelz'
                else:
                    grouping_ids[x] = self.grouping_ids[x]
            self.grouping_ids = grouping_ids.copy()

        self.top_ids = set(grouping_ids.values())
        self.bottom_ids = grouping_ids.keys()

        hier = df.abs().groupby(grouping_ids, axis=1).sum()
        self.hier = hier

        if self.reconciliation == 'mean':
            level_sums = pd.DataFrame(hier.sum(axis=0))
            individal_sums = pd.DataFrame(df.abs().sum(axis=0))
            divisors = pd.DataFrame.from_dict(grouping_ids, orient='index')
            divisors.columns = ['group']
            divisors = divisors.merge(level_sums, left_on='group', right_index=True)
            divisors = divisors.merge(individal_sums, left_index=True, right_index=True)
            divisors.columns = ['group', 'divisor', 'value']
            divisors['fraction'] = divisors['value'] / divisors['divisor']
            self.divisors = divisors

        return self

    def transform(self, df):
        """Apply hierarchy to existing data with bottom levels only."""
        try:
            return pd.concat([df, self.hier], axis=1)
        except Exception as e:
            raise ValueError(f"{e} .fit() has not been called.")

    def reconcile(self, df):
        """Apply to forecasted data containing bottom and top levels."""
        if self.reconciliation is None:
            return df[self.bottom_ids]
        elif self.reconciliation == 'mean':
            fore = df
            fracs = pd.DataFrame(
                np.repeat(
                    self.divisors['fraction'].values.reshape(1, -1),
                    fore.shape[0],
                    axis=0,
                )
            )
            fracs.index = fore.index
            fracs.columns = pd.MultiIndex.from_frame(
                self.divisors.reset_index()[['index', 'group']]
            )

            top_level = fore[self.top_ids]
            bottom_up = (
                fore[self.bottom_ids].abs().groupby(self.grouping_ids, axis=1).sum()
            )

            diff = (top_level - bottom_up) / 2

            # gotta love that 'level' option on multiple for easy broadcasting
            test = fracs.multiply(diff, level='group')
            test.columns = self.divisors.index

            result = fore[self.bottom_ids] + test
            return result
        else:
            print("Complete and utter failure.")
            return df


"""
grouping_ids = {
    'CSUSHPISA': 'A',
    'EMVOVERALLEMV': 'A',
    'EXCAUS': 'exchange rates',
    'EXCHUS': 'exchange rates',
    'EXUSEU': 'exchange rates',
    'GS10': 'B',
    'MCOILWTICO': 'C',
    'T10YIEM': 'C',
    'USEPUINDXM': 'C'
    }
test = hierarchial(n_groups=3, grouping_method='dbscan',
                   grouping_ids=None, reconciliation='mean').fit(df)
test_df = test.transform(df)
test.reconcile(test_df)
"""
# how to assign groups
# how to assign blame/reconcile
# multiple group levels
# one additional overall-level


def mint_reconcile(S: np.ndarray, y_all: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    MinT reconciliation formula:
       y_all^r = S (S' W^-1 S)^-1 S' W^-1 y_all

    Parameters
    ----------
    S : np.ndarray, shape (L, M)
        The hierarchy (aggregator) matrix. L = number of hierarchical levels
        (top + middle + bottom), M = number of bottom-level series.
    y_all : np.ndarray, shape (T, L)
        Forecasts at all levels for T time points (the second dimension L
        must match S.shape[0]).
    W : np.ndarray, shape (L, L)
        The (regularized) covariance (or weighting) matrix for the hierarchical levels.

    Returns
    -------
    y_all_reconciled : np.ndarray, shape (T, L)
        Reconciled forecasts for all L levels (top, middle, bottom).
    """
    # Invert W
    W_inv = np.linalg.inv(W)
    # (S' W^-1 S)^-1
    S_T = S.T
    M = S_T @ W_inv @ S   # shape (M, M)
    M_inv = np.linalg.inv(M)

    # Reconcile (vectorized for all T)
    y_all_T = y_all.T  # (L, T)
    step_a = S_T @ W_inv @ y_all_T  # (M, T)
    step_b = M_inv @ step_a         # (M, T)
    step_c = S @ step_b             # (L, T)
    y_all_reconciled = step_c.T     # (T, L)
    return y_all_reconciled


def erm_reconcile(S: np.ndarray, y_all: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    ERM (Error or Empirical Risk Minimization) Reconciliation:
      Solve Weighted LS:  min_{y_bottom}  ||y_all - S y_bottom||_W^2
      subject to hierarchical constraints.

    The closed-form solution for y_bottom^r:
       y_bottom^r = (S' W S)^{-1} S' W y_all
    => y_all^r = S y_bottom^r = S (S' W S)^{-1} S' W y_all

    Parameters
    ----------
    S : np.ndarray, shape (L, M)
        Hierarchy matrix. L = # total levels, M = # bottom series.
    y_all : np.ndarray, shape (T, L)
        Forecasts for T time points, dimension L.
    W : np.ndarray, shape (L, L)
        Weight (covariance) matrix for the Weighted LS objective.

    Returns
    -------
    y_all_reconciled : np.ndarray, shape (T, L)
        Reconciled forecasts for all L levels.
    """
    S_T = S.T
    A = S_T @ W @ S  # shape (M, M)
    A_inv = np.linalg.inv(A)

    # P = S (S' W S)^{-1} S' W
    P = S @ A_inv @ S_T @ W  # shape (L, L)

    # (T, L) @ (L, L) => (T, L)
    y_reconciled = y_all @ P.T
    return y_reconciled


def ledoit_wolf_covariance(X: np.ndarray, assume_centered: bool = False) -> np.ndarray:
    """
    Computes the Ledoit-Wolf shrunk covariance matrix of X.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The data matrix. Each row is an observation, each column is a variable.
    assume_centered : bool
        If True, X is assumed to already be centered.

    Returns
    -------
    lw_cov : np.ndarray, shape (n_features, n_features)
        The Ledoit-Wolf shrunk covariance matrix estimate.

    Notes
    -----
    - This shrinks the sample covariance toward the identity matrix.
    - The shrinkage intensity gamma is determined from data per Ledoit & Wolf (2004).
    """
    n_samples, n_features = X.shape
    if not assume_centered:
        X = X - X.mean(axis=0, keepdims=True)

    # Empirical covariance
    emp_cov = (X.T @ X) / (n_samples - 1)

    # mu = average of diag(emp_cov)
    mu = np.trace(emp_cov) / n_features

    # Sum-of-squared differences for beta
    den = (n_samples - 1.0) ** 2
    beta = 0.0
    # Minimal for-loop to accumulate squared difference
    for i in range(n_samples):
        row = X[i, :]
        diff = np.outer(row, row) - emp_cov
        beta += (diff * diff).sum()
    beta /= den

    # gamma = beta / sum((emp_cov - mu*I)^2)
    diff = emp_cov - mu * np.eye(n_features)
    gamma = beta / (diff * diff).sum()
    gamma = max(0.0, min(1.0, gamma))  # clip

    shrunk_cov = (1.0 - gamma) * emp_cov + gamma * mu * np.eye(n_features)
    return shrunk_cov
