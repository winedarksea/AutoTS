# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:27:46 2022

@author: Colin
"""
import datetime
import numpy as np
import pandas as pd
import os
from itertools import groupby
from operator import itemgetter


def consecutive_groups(iterable, ordering=lambda x: x):
    """Yield groups of consecutive items using :func:`itertools.groupby`.

    From more_itertools package, see description there for details (circa mid 2022)
    """
    for k, g in groupby(enumerate(iterable), key=lambda x: x[0] - ordering(x[1])):
        yield map(itemgetter(1), g)


class NonparametricThreshold:
    def __init__(
        self,
        data,
        warmup_pts: int = 1,
        p=0.1,
        error_buffer=1,
        z_init=2.5,
        z_limit=12.0,
        z_step=0.5,
        max_contamination=0.25,
        mean_weight: float = 10,
        sd_weight: float = 10,
        anomaly_count_weight: float = 1,
        inverse: bool = False,
    ):
        """
        Data and outlier calculations for a 1D numpy array.
        Includes finding thresholds, pruning, and scoring anomalous sequences
        for errors and inverted errors (flipped around mean) - significant drops
        in values can also be anomalous.
        Modified from telemanom and https://arxiv.org/pdf/1802.04431.pdf

        making raising mean/sd weight and z_init smaller will encourage more outliers
        making making count_weight larger and z_max larger will encourage fewer outliers
        more pruning leads to fewer outliers
        truly massive outliers (orders of magnitude above others) will make it hard for smaller anomalies to be detected

        Args:
            data (np.array): wide style data
            warmup_pts (int): amount of initial data to discard
            p (float): percent of data to prune - pruning removes 'near' anomalies, use to focus on most anomalous
            error_buffer (int): number of values surrounding an error that are brought into the sequence (promotes grouping on nearby sequences)
            z_init (float): std dev starting point from mean for search
            z_limit (float): max st dev away to include in search space
            z_step (float): size of zstep, smaller values are slower but may yield slightly better optimums
            max_contaimation (float): max % of data to allow as a considered an outlier
            mean_weight (int): usually in range [1, 1000] - favor decrease in mean
            sd_weight (int): usually in range [1, 1000] - favor decrease in std deviation
            anomaly_count_weight (int): usally in range [0, 20] - favor fewer anomaly detections

        Attributes:
            i_anom (arr): indices of anomalies in window
            i_anom_inv (arr): indices of anomalies in window of inverted
                telemetry values
            E_seq (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window
            E_seq_inv (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window of inverted telemetry
                values
            non_anom_max (float): highest smoothed error value below epsilon
            non_anom_max_inv (float): highest smoothed error value below
                epsilon_inv
            config (obj): see Args
            anom_scores (arr): score indicating relative severity of each
                anomaly sequence in E_seq within a window
            window_num (int): see Args
            sd_lim (int): default number of standard deviations to use for
                threshold if no winner or too many anomalous ranges when scoring
                candidate thresholds
            sd_threshold (float): number of standard deviations for calculation
                of best anomaly threshold
            sd_threshold_inv (float): same as above for inverted channel values
            e_s (arr): exponentially-smoothed prediction errors in window
            e_s_inv (arr): inverted e_s
            sd_e_s (float): standard deviation of e_s
            mean_e_s (float): mean of e_s
            epsilon (float): threshold for e_s above which an error is
                considered anomalous
            epsilon_inv (float): threshold for inverted e_s above which an error
                is considered anomalous
            y_test (arr): Actual telemetry values for window
            sd_values (float): st dev of y_test
            perc_high (float): the 95th percentile of y_test values
            perc_low (float): the 5th percentile of y_test values
            inter_range (float): the range between perc_high - perc_low
            num_to_ignore (int): number of values to ignore initially when
                looking for anomalies
        """

        self.i_anom = np.array([])
        self.E_seq = np.array([])
        self.non_anom_max = -1000000
        self.i_anom_inv = np.array([])
        self.E_seq_inv = np.array([])
        self.non_anom_max_inv = -1000000

        self.warmup_pts = warmup_pts
        self.p = p
        self.z_init = z_init
        self.z_step = z_step
        self.max_contamination = max_contamination
        self.error_buffer = error_buffer
        self.mean_weight = mean_weight
        self.sd_weight = sd_weight
        self.anomaly_count_weight = anomaly_count_weight
        self.anom_scores = []

        self.sd_lim = z_limit
        self.sd_threshold = self.sd_lim
        self.sd_threshold_inv = self.sd_lim

        self.e_s = data

        self.mean_e_s = np.mean(self.e_s)
        self.sd_e_s = np.std(self.e_s)
        self.e_s_inv = np.array([self.mean_e_s + (self.mean_e_s - e) for e in self.e_s])

        self.epsilon = self.mean_e_s + self.sd_lim * self.sd_e_s
        self.epsilon_inv = self.mean_e_s + self.sd_lim * self.sd_e_s

        # ignore initial error values until enough history for processing
        self.num_to_ignore = self.warmup_pts
        # if y_test is small, ignore fewer
        if len(data) < 2500:
            self.num_to_ignore = self.warmup_pts
        if len(data) < 1800:
            self.num_to_ignore = 0

    def find_epsilon(self, inverse=False):
        """
        Find the anomaly threshold that maximizes function representing
        tradeoff between:
            a) number of anomalies and anomalous ranges
            b) the reduction in mean and st dev if anomalous points are removed
            from errors
        (see https://arxiv.org/pdf/1802.04431.pdf)

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """
        e_s = self.e_s if not inverse else self.e_s_inv

        max_score = -10000000

        for z in np.arange(self.z_init, self.sd_lim, self.z_step):
            epsilon = self.mean_e_s + (self.sd_e_s * z)

            pruned_e_s = e_s[e_s < epsilon]

            i_anom = np.argwhere(e_s >= epsilon).reshape(
                -1,
            )
            i_anom = i_anom[i_anom >= self.num_to_ignore]

            if len(i_anom) > 0:
                i_anom = np.sort(np.unique(i_anom))

                """
                # diverging from original code, I am only using buffer for grouping (not reported anoms)
                buffer = np.arange(1, self.error_buffer)
                i_anom_buffered = np.sort(
                    np.concatenate(
                        (
                            i_anom,
                            np.array([i + buffer for i in i_anom]).flatten(),
                            np.array([i - buffer for i in i_anom]).flatten(),
                        )
                    )
                )
                # remove faulty indices generated by adding buffer
                i_anom_buffered = i_anom_buffered[(i_anom_buffered < len(e_s)) & (i_anom_buffered >= 0)]

                # if it is first window, ignore initial errors (need some history)
                i_anom_buffered = i_anom_buffered[i_anom_buffered >= self.num_to_ignore]
                # sort for grouping and easier viewing
                i_anom_buffered = np.sort(np.unique(i_anom_buffered))

                # group anomalous indices into continuous sequences
                groups = [list(group) for group in consecutive_groups(i_anom_buffered)]
                E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
                """

                # generate a score for change in mean and st.dev
                # made this more greedy than original
                mean_perc_decrease = (
                    self.mean_e_s - np.mean(pruned_e_s)
                ) / self.mean_e_s
                sd_perc_decrease = (self.sd_e_s - np.std(pruned_e_s)) / self.sd_e_s
                score = (
                    (1 + mean_perc_decrease) ** self.mean_weight
                    + (1 + sd_perc_decrease) ** self.sd_weight
                ) / (
                    len(i_anom) ** self.anomaly_count_weight  # + len(E_seq) ** 2
                )
                """
                score = (mean_perc_decrease + sd_perc_decrease) / (
                    len(i_anom) + len(E_seq) ** 2
                )
                """
                # print(f"iter {z} with shape {i_anom.shape} with score: {score} and m%d {mean_perc_decrease}")

                # set epsilon if score is highest seen
                if (
                    score >= max_score
                    and len(i_anom) < (len(e_s) * self.max_contamination)
                    # and len(E_seq) <= 5
                ):
                    max_score = score
                    if not inverse:
                        self.sd_threshold = z
                        self.epsilon = self.mean_e_s + z * self.sd_e_s
                        self.i_anom = i_anom
                        # self.E_seq = E_seq
                    else:
                        self.sd_threshold_inv = z
                        self.epsilon_inv = self.mean_e_s + z * self.sd_e_s
                        self.i_anom_inv = i_anom
                        # self.E_seq_inv = E_seq

    def compare_to_epsilon(self, inverse=False):
        """
        Compare smoothed error values to epsilon (error threshold) and group
        consecutive errors together into sequences.

        Args:
            errors_all (obj): Errors class object containing list of all
            previously identified anomalies in test set
        """

        e_s = self.e_s if not inverse else self.e_s_inv
        epsilon = self.epsilon if not inverse else self.epsilon_inv

        i_anom = np.argwhere((e_s >= epsilon)).reshape(
            -1,
        )
        i_anom = np.sort(np.unique(i_anom))
        # if it is first window, ignore initial errors (need some history)
        i_anom = i_anom[i_anom >= self.num_to_ignore]

        if len(i_anom) == 0:
            return

        """
        buffer = np.arange(1, self.error_buffer + 1)
        # diverging from original code, I am only using buffer for grouping (not reported anoms)
        i_anom_buffered = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        # remove faulty indices generated by adding buffer
        i_anom_buffered = i_anom_buffered[(i_anom_buffered < len(e_s)) & (i_anom_buffered >= 0)]

        # if it is first window, ignore initial errors (need some history)
        i_anom = i_anom[i_anom >= self.num_to_ignore]
        i_anom_buffered = i_anom_buffered[i_anom_buffered >= self.num_to_ignore]
        i_anom_buffered = np.sort(np.unique(i_anom_buffered))

        # group anomalous indices into continuous sequences
        groups = [list(group) for group in consecutive_groups(i_anom_buffered)]
        E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
        """

        if inverse:
            self.i_anom_inv = i_anom
            # self.E_seq_inv = E_seq
        else:
            self.i_anom = i_anom
            # self.E_seq = E_seq

    def prune_anoms(self, inverse=False):
        """
        Remove anomalies that don't meet minimum separation from the next
        closest anomaly or error value

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """
        if self.p is None or self.p <= 0:
            return
        # E_seq = self.E_seq if not inverse else self.E_seq_inv
        e_s = self.e_s if not inverse else self.e_s_inv
        i_anom = self.i_anom if not inverse else self.i_anom_inv
        if i_anom.size == 0:
            return
        else:
            non_anom_max = np.max(self.e_s[~i_anom])
        # non_anom_max = self.non_anom_max if not inverse else self.non_anom_max_inv

        # if len(E_seq) == 0:
        #     return

        # E_seq_max = np.array([max(e_s[e[0]: e[1] + 1]) for e in E_seq])
        # adjusted to i_anom, with the grouping done by E_seq not used
        E_seq_max = e_s[i_anom]
        E_seq_max_sorted = np.sort(E_seq_max)[::-1]
        E_seq_max_sorted = np.append(E_seq_max_sorted, [non_anom_max])

        i_to_remove = np.array([])
        # "If the minimum decrease p is not met by d(i) *and* for all subsequent errors
        # those smoothed error sequences are reclassified as nominal"
        for i in range(0, len(E_seq_max_sorted) - 1):
            if (E_seq_max_sorted[i] - E_seq_max_sorted[i + 1]) / E_seq_max_sorted[
                i
            ] < self.p:
                i_to_remove = np.append(
                    i_to_remove, np.argwhere(E_seq_max == E_seq_max_sorted[i])
                )
            else:
                # this else handles the "all subsequent part" by resetting
                i_to_remove = np.array([])
        i_to_remove[::-1].sort()

        if len(i_to_remove) > 0:
            # print(f"pruning {len(i_to_remove)} anomalies")
            # print(i_to_remove)
            i_anom = np.delete(i_anom, i_to_remove.astype(int), axis=0)

        # this was previously updating i_anom and i_anom_inv instead of E_seq
        if len(i_anom) == 0 and inverse:
            self.i_anom_inv = np.array([])
            return
        elif len(i_anom) == 0 and not inverse:
            self.i_anom = np.array([])
            return

        # indices_to_keep = np.concatenate([range(e_seq[0], e_seq[-1] + 1) for e_seq in E_seq])

        if not inverse:
            # mask = np.isin(self.i_anom, indices_to_keep)
            self.i_anom = i_anom
        else:
            # mask_inv = np.isin(self.i_anom_inv, indices_to_keep)
            self.i_anom_inv = i_anom

    def score_anomalies(self):
        """
        Calculate anomaly scores based on max distance from epsilon
        for each anomalous sequence.
        """
        return abs(self.e_s - self.epsilon) / (self.mean_e_s + self.sd_e_s)


def nonparametric(series, method_params):
    mod = NonparametricThreshold(series.to_numpy().flatten(), **method_params)
    mod.find_epsilon()
    mod.prune_anoms()
    i_anom = mod.i_anom
    if method_params.get('inverse', False):
        mod.find_epsilon(inverse=True)
        mod.prune_anoms(inverse=True)
        i_anom = np.unique(np.concatenate([i_anom, mod.i_anom_inv]))
    scores = pd.DataFrame(
        mod.score_anomalies(),
        index=series.index,
        columns=['anomaly_score'],
    )
    # print(i_anom)
    res = pd.DataFrame(
        np.where(series.index.isin(series.index[i_anom.astype(int)]), -1, 1),
        index=series.index,
        columns=["anomaly"],
    )
    return res, scores


"""
df = pd.read_csv(
    "holidays.csv", index_col=0, parse_dates=[0],
)

from autots.evaluator.auto_model import back_forecast

backcast = back_forecast(
    df,
    model_name="LastValueNaive",
    model_param_dict={},
    model_transform_dict={"fillna": "rolling_mean", "transformations": {"1": "DifferencedTransformer"}, "transformation_params": {"1": {}}},
    n_splits="auto",
    forecast_length=4,
    frequency="infer",
)

config = {
    'l_s': 1,  # 250
    'batch_size': 70,  # 70
    'p': 0.10,  # 0.14
    'error_buffer': 1,  # 100
    "window_size": 10,  # 30
    "smoothing_perc": 0.05,  # 0.03, None
}

errors = Errors(y_test=backcast.forecast.to_numpy()[:, 10:11], y_hat=df.to_numpy()[:, 10:11], config=config)
errors.process_batches(backcast.forecast.to_numpy())
result_row = {
    'n_predicted_anoms': len(errors.E_seq),
    # 'normalized_pred_error': errors.normalized,
    'anom_scores': errors.anom_scores
}
result_row['anomaly_sequences'] = errors.E_seq

print('{} anomalies found'.format(result_row['n_predicted_anoms']))
# print('anomaly sequences start/end indices: {}'.format(result_row['anomaly_sequences']))
# print('anomaly scores: {}\n'.format(result_row['anom_scores']))

indices = []
for x in result_row['anomaly_sequences']:
    indices.extend(list(range(x[0], x[1] + 1)))
df.index[indices]


from autots.tools.transform import DatepartRegressionTransformer

model = DatepartRegressionTransformer(
    datepart_method="simple_3",
    regression_model={
        "model": "ElasticNet",  # ElasticNet
        "model_params": {},  # {"max_depth": None, "min_samples_split": 0.05}s
    },
)
df2 = abs(model.fit_transform(df))

# for univariate case, flatten after taking log or normalizing
# remove in advance values with extremely high score
# runtime, number of holidays, prediction forecast gain

for i in range(df.shape[1]):
    data = df2.to_numpy()[:, i].flatten()
    mod = NonparametricThreshold(
        data, p=None, error_buffer=20,
        warmup_pts=1,
        z_init=1.75, z_limit=10.0,
        z_step=0.25,
        max_contamination=0.25,
        mean_weight=25,
        sd_weight=25,
        anomaly_count_weight=1,
    )
    mod.find_epsilon()
    print(mod.epsilon)
    num_anoms = len(mod.i_anom)
    print(num_anoms)
    print(f"sd_threshold {mod.sd_threshold}")
    # mod.compare_to_epsilon()
    mod.prune_anoms()
    num_anoms = len(mod.i_anom)
    mod.score_anomalies()
    print(f"post pruning: {len(mod.i_anom)}")
    if data.ndim > 1:
        dates = np.tile(df.index, df.shape[1])[mod.i_anom]
    else:
        dates = df.index[mod.i_anom]
    print(dates)
    print(pd.qcut(data, 10, labels=False)[mod.i_anom])
    col_name = df.columns[i]
    print(col_name)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    df2[df.columns[i]].plot(ax=ax, title=col_name)
    if num_anoms > 1:
        ax.scatter(dates, data[mod.i_anom], c="red")
    plt.show()

    mod.find_epsilon(inverse=True)
    print(f"Inverse anomalies: {len(mod.i_anom_inv)}")
    mod.prune_anoms(inverse=True)
"""
