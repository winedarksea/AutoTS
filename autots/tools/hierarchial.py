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

        hier = df.abs().T.groupby(grouping_ids).sum()
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
                fore[self.bottom_ids].abs().T.groupby(self.grouping_ids).sum()
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
