# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified here to be less messy, but still has problems

import random
import datetime
import numpy as np
import pandas as pd
from autots.models.base import ModelObject, PredictionObject
from autots.tools.probabilistic import Point_to_Probability

try:
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow import keras
    from pandas.tseries.holiday import EasterMonday
    from pandas.tseries.holiday import GoodFriday
    from pandas.tseries.holiday import Holiday
    from pandas.tseries.holiday import SU
    from pandas.tseries.holiday import TH
    from pandas.tseries.holiday import USColumbusDay
    from pandas.tseries.holiday import USLaborDay
    from pandas.tseries.holiday import USMartinLutherKingJr
    from pandas.tseries.holiday import USMemorialDay
    from pandas.tseries.holiday import USPresidentsDay
    from pandas.tseries.holiday import USThanksgivingDay
    from pandas.tseries.offsets import DateOffset
    from pandas.tseries.offsets import Day
    from pandas.tseries.offsets import Easter
    from tqdm import tqdm

    full_import = True
except Exception:
    full_import = False


def _distance_to_holiday(holiday):
    """Return distance to given holiday."""

    def _distance_to_day(index, MAX_WINDOW=200):
        # This is 183 to cover half a year (in both directions), also for leap years
        # + 17 as Eastern can be between March, 22 - April, 25
        holiday_date = holiday.dates(
            index - pd.Timedelta(days=MAX_WINDOW),
            index + pd.Timedelta(days=MAX_WINDOW),
        )
        assert (
            len(holiday_date) != 0  # pylint: disable=g-explicit-length-test
        ), f"No closest holiday for the date index {index} found."
        # It sometimes returns two dates if it is exactly half a year after the
        # holiday. In this case, the smaller distance (182 days) is returned.
        return (index - holiday_date[0]).days

    return _distance_to_day


def get_HOLIDAYS():
    EasterSunday = Holiday("Easter Sunday", month=1, day=1, offset=[Easter(), Day(0)])
    NewYearsDay = Holiday("New Years Day", month=1, day=1)
    SuperBowl = Holiday("Superbowl", month=2, day=1, offset=DateOffset(weekday=SU(1)))
    MothersDay = Holiday(
        "Mothers Day", month=5, day=1, offset=DateOffset(weekday=SU(2))
    )
    IndependenceDay = Holiday("Independence Day", month=7, day=4)
    ChristmasEve = Holiday("Christmas", month=12, day=24)
    ChristmasDay = Holiday("Christmas", month=12, day=25)
    NewYearsEve = Holiday("New Years Eve", month=12, day=31)
    BlackFriday = Holiday(
        "Black Friday",
        month=11,
        day=1,
        offset=[pd.DateOffset(weekday=TH(4)), Day(1)],
    )
    CyberMonday = Holiday(
        "Cyber Monday",
        month=11,
        day=1,
        offset=[pd.DateOffset(weekday=TH(4)), Day(4)],
    )

    HOLIDAYS = [
        EasterMonday,
        GoodFriday,
        USColumbusDay,
        USLaborDay,
        USMartinLutherKingJr,
        USMemorialDay,
        USPresidentsDay,
        USThanksgivingDay,
        EasterSunday,
        NewYearsDay,
        SuperBowl,
        MothersDay,
        IndependenceDay,
        ChristmasEve,
        ChristmasDay,
        NewYearsEve,
        BlackFriday,
        CyberMonday,
    ]
    return HOLIDAYS


class TimeCovariates(object):
    """Extract all time covariates except for holidays."""

    def __init__(
        self,
        datetimes,
        normalized=True,
        holiday=False,
    ):
        """Init function.

        Args:
          datetimes: pandas DatetimeIndex (lowest granularity supported is min)
          normalized: whether to normalize features or not
          holiday: fetch holiday features or not

        Returns:
          None
        """
        self.normalized = normalized
        self.dti = datetimes
        self.holiday = holiday

    def _minute_of_hour(self):
        minutes = np.array(self.dti.minute, dtype=np.float32)
        if self.normalized:
            minutes = minutes / 59.0 - 0.5
        return minutes

    def _hour_of_day(self):
        hours = np.array(self.dti.hour, dtype=np.float32)
        if self.normalized:
            hours = hours / 23.0 - 0.5
        return hours

    def _day_of_week(self):
        day_week = np.array(self.dti.dayofweek, dtype=np.float32)
        if self.normalized:
            day_week = day_week / 6.0 - 0.5
        return day_week

    def _day_of_month(self):
        day_month = np.array(self.dti.day, dtype=np.float32)
        if self.normalized:
            day_month = day_month / 30.0 - 0.5
        return day_month

    def _day_of_year(self):
        day_year = np.array(self.dti.dayofyear, dtype=np.float32)
        if self.normalized:
            day_year = day_year / 364.0 - 0.5
        return day_year

    def _month_of_year(self):
        month_year = np.array(self.dti.month, dtype=np.float32)
        if self.normalized:
            month_year = month_year / 11.0 - 0.5
        return month_year

    def _week_of_year(self):
        week_year = np.array(self.dti.strftime("%U").astype(int), dtype=np.float32)
        if self.normalized:
            week_year = week_year / 51.0 - 0.5
        return week_year

    def _get_holidays(self):
        dti_series = self.dti.to_series()
        HOLIDAYS = get_HOLIDAYS()
        hol_variates = np.vstack(
            [dti_series.apply(_distance_to_holiday(h)).values for h in tqdm(HOLIDAYS)]
        )
        # hol_variates is (num_holiday, num_time_steps), the normalization should be
        # performed in the num_time_steps dimension.
        return StandardScaler().fit_transform(hol_variates.T).T

    def get_covariates(self):
        """Get all time covariates."""
        moh = self._minute_of_hour().reshape(1, -1)
        hod = self._hour_of_day().reshape(1, -1)
        dom = self._day_of_month().reshape(1, -1)
        dow = self._day_of_week().reshape(1, -1)
        doy = self._day_of_year().reshape(1, -1)
        moy = self._month_of_year().reshape(1, -1)
        woy = self._week_of_year().reshape(1, -1)

        all_covs = [
            moh,
            hod,
            dom,
            dow,
            doy,
            moy,
            woy,
        ]
        columns = ["moh", "hod", "dom", "dow", "doy", "moy", "woy"]
        if self.holiday:
            hol_covs = self._get_holidays()
            HOLIDAYS = get_HOLIDAYS()
            all_covs.append(hol_covs)
            columns += [f"hol_{i}" for i in range(len(HOLIDAYS))]

        return pd.DataFrame(
            data=np.vstack(all_covs).transpose(),
            columns=columns,
            index=self.dti,
        )


class TimeSeriesdata(object):
    """Data loader class."""

    def __init__(
        self,
        df,
        num_cov_cols,
        cat_cov_cols,
        ts_cols,
        train_range,
        val_range,
        test_range,
        hist_len,
        pred_len,
        batch_size,
        freq="D",
        normalize=True,
        epoch_len=None,
        holiday=False,
        permute=True,
    ):
        """Initialize objects.

        Args:
          df: wide style dataframe
          num_cov_cols: list of numerical global covariates
          cat_cov_cols: list of categorical global covariates
          ts_cols: columns corresponding to ts
          train_range: tuple of train ranges
          val_range: tuple of validation ranges
          test_range: tuple of test ranges
          hist_len: historical context
          pred_len: prediction length
          batch_size: batch size (number of ts in a batch)
          freq: freq of original data
          normalize: std. normalize data or not
          epoch_len: num iters in an epoch
          holiday: use holiday features or not
          permute: permute ts in train batches or not

        Returns:
          None
        """
        self.data_df = df
        if not num_cov_cols:
            self.data_df["ncol"] = np.zeros(self.data_df.shape[0])
            num_cov_cols = ["ncol"]
        if not cat_cov_cols:
            self.data_df["ccol"] = np.zeros(self.data_df.shape[0])
            cat_cov_cols = ["ccol"]
        self.data_df.fillna(0, inplace=True)
        # self.data_df.set_index(pd.DatetimeIndex(self.data_df[datetime_col]), inplace=True)
        self.num_cov_cols = num_cov_cols
        self.cat_cov_cols = cat_cov_cols
        self.ts_cols = ts_cols
        self.train_range = train_range
        self.val_range = val_range
        self.test_range = test_range
        data_df_idx = self.data_df.index
        date_index = data_df_idx.union(
            pd.date_range(
                data_df_idx[-1] + pd.Timedelta(1, freq=freq),
                periods=pred_len + 1,
                freq=freq,
            )
        )
        self.time_df = TimeCovariates(date_index, holiday=holiday).get_covariates()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.freq = freq
        self.normalize = normalize
        self.data_mat = self.data_df[self.ts_cols].to_numpy().transpose()
        self.data_mat = self.data_mat[:, 0 : self.test_range[1]]
        self.time_mat = self.time_df.to_numpy().transpose()
        self.num_feat_mat = self.data_df[num_cov_cols].to_numpy().transpose()
        self.cat_feat_mat, self.cat_sizes = self._get_cat_cols(cat_cov_cols)
        self.normalize = normalize
        if normalize:
            self._normalize_data()
        self.epoch_len = epoch_len
        self.permute = permute

    def _get_cat_cols(self, cat_cov_cols):
        """Get categorical columns."""
        cat_vars = []
        cat_sizes = []
        for col in cat_cov_cols:
            dct = {x: i for i, x in enumerate(self.data_df[col].unique())}
            cat_sizes.append(len(dct))
            mapped = (
                self.data_df[col].map(lambda x: dct[x]).to_numpy().transpose()
            )  # pylint: disable=cell-var-from-loop
            cat_vars.append(mapped)
        return np.vstack(cat_vars), cat_sizes

    def _normalize_data(self):
        self.scaler = StandardScaler()
        train_mat = self.data_mat[:, self.train_range[0] : self.train_range[1]]
        self.scaler = self.scaler.fit(train_mat.transpose())
        self.data_mat = self.scaler.transform(self.data_mat.transpose()).transpose()

    def train_gen(self):
        """Generator for training data."""
        num_ts = len(self.ts_cols)
        perm = np.arange(
            self.train_range[0] + self.hist_len,
            self.train_range[1] - self.pred_len,
        )
        perm = np.random.permutation(perm)
        hist_len = self.hist_len
        if not self.epoch_len:
            epoch_len = len(perm)
        else:
            epoch_len = self.epoch_len
        for idx in perm[0:epoch_len]:
            for _ in range(num_ts // self.batch_size + 1):
                if self.permute:
                    tsidx = np.random.choice(
                        num_ts, size=self.batch_size, replace=False
                    )
                else:
                    tsidx = np.arange(num_ts)
                dtimes = np.arange(idx - hist_len, idx + self.pred_len)
                (
                    bts_train,
                    bts_pred,
                    bfeats_train,
                    bfeats_pred,
                    bcf_train,
                    bcf_pred,
                ) = self._get_features_and_ts(dtimes, tsidx, hist_len)

                all_data = [
                    bts_train,
                    bfeats_train,
                    bcf_train,
                    bts_pred,
                    bfeats_pred,
                    bcf_pred,
                    tsidx,
                ]
                yield tuple(all_data)

    def test_val_gen(self, mode="val"):
        """Generator for validation/test data."""
        if mode == "val":
            start = self.val_range[0]
            end = self.val_range[1] - self.pred_len + 1
        elif mode == "test":
            start = self.test_range[0]
            end = self.test_range[1] - self.pred_len + 1
        elif mode == "predict":
            start = self.test_range[0]
            end = self.test_range[1] + 1
        else:
            raise NotImplementedError("Eval mode not implemented")
        num_ts = len(self.ts_cols)
        hist_len = self.hist_len
        perm = np.arange(start, end)
        if self.epoch_len:
            epoch_len = self.epoch_len
        else:
            epoch_len = len(perm)
        for idx in perm[0:epoch_len]:
            for batch_idx in range(0, num_ts, self.batch_size):
                tsidx = np.arange(batch_idx, min(batch_idx + self.batch_size, num_ts))
                dtimes = np.arange(idx - hist_len, idx + self.pred_len)
                (
                    bts_train,
                    bts_pred,
                    bfeats_train,
                    bfeats_pred,
                    bcf_train,
                    bcf_pred,
                ) = self._get_features_and_ts(dtimes, tsidx, hist_len)
                all_data = [
                    bts_train,
                    bfeats_train,
                    bcf_train,
                    bts_pred,
                    bfeats_pred,
                    bcf_pred,
                    tsidx,
                ]
                yield tuple(all_data)

    def _get_features_and_ts(self, dtimes, tsidx, hist_len=None):
        """Get features and ts in specified windows."""
        if hist_len is None:
            hist_len = self.hist_len
        data_times = dtimes[dtimes < self.data_mat.shape[1]]
        bdata = self.data_mat[:, data_times]
        bts = bdata[tsidx, :]
        bnf = self.num_feat_mat[:, data_times]
        bcf = self.cat_feat_mat[:, data_times]
        btf = self.time_mat[:, dtimes]
        if bnf.shape[1] < btf.shape[1]:
            rem_len = btf.shape[1] - bnf.shape[1]
            rem_rep = np.repeat(bnf[:, [-1]], repeats=rem_len)
            rem_rep_cat = np.repeat(bcf[:, [-1]], repeats=rem_len)
            bnf = np.hstack([bnf, rem_rep.reshape(bnf.shape[0], -1)])
            bcf = np.hstack([bcf, rem_rep_cat.reshape(bcf.shape[0], -1)])
        bfeats = np.vstack([btf, bnf])
        bts_train = bts[:, 0:hist_len]
        bts_pred = bts[:, hist_len:]
        bfeats_train = bfeats[:, 0:hist_len]
        bfeats_pred = bfeats[:, hist_len:]
        bcf_train = bcf[:, 0:hist_len]
        bcf_pred = bcf[:, hist_len:]
        return bts_train, bts_pred, bfeats_train, bfeats_pred, bcf_train, bcf_pred

    def tf_dataset(self, mode="train"):
        """Tensorflow Dataset."""
        if mode == "train":
            gen_fn = self.train_gen
        else:
            gen_fn = lambda: self.test_val_gen(mode)
        output_types = tuple(
            [tf.float32] * 2 + [tf.int32] + [tf.float32] * 2 + [tf.int32] * 2
        )
        dataset = tf.data.Dataset.from_generator(gen_fn, output_types)
        # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


EPS = 1e-7


if full_import:

    class MLPResidual(keras.layers.Layer):
        """Simple one hidden state residual network."""

        def __init__(self, hidden_dim, output_dim, layer_norm=False, dropout_rate=0.0):
            super(MLPResidual, self).__init__()
            self.lin_a = tf.keras.layers.Dense(
                hidden_dim,
                activation="relu",
            )
            self.lin_b = tf.keras.layers.Dense(
                output_dim,
                activation=None,
            )
            self.lin_res = tf.keras.layers.Dense(
                output_dim,
                activation=None,
            )
            if layer_norm:
                self.lnorm = tf.keras.layers.LayerNormalization()
            self.layer_norm = layer_norm
            self.dropout = tf.keras.layers.Dropout(dropout_rate)

        def call(self, inputs):
            """Call method."""
            h_state = self.lin_a(inputs)
            out = self.lin_b(h_state)
            out = self.dropout(out)
            res = self.lin_res(inputs)
            if self.layer_norm:
                return self.lnorm(out + res)
            return out + res

    def _make_dnn_residual(hidden_dims, layer_norm=False, dropout_rate=0.0):
        """Multi-layer DNN residual model."""
        if len(hidden_dims) < 2:
            return keras.layers.Dense(
                hidden_dims[-1],
                activation=None,
            )
        layers = []
        for i, hdim in enumerate(hidden_dims[:-1]):
            layers.append(
                MLPResidual(
                    hdim,
                    hidden_dims[i + 1],
                    layer_norm=layer_norm,
                    dropout_rate=dropout_rate,
                )
            )
        return keras.Sequential(layers)

    class TideModel(keras.Model):
        """Main class for multi-scale DNN model."""

        def __init__(
            self,
            model_config,
            pred_len,
            cat_sizes,
            num_ts,
            transform=False,
            cat_emb_size=4,
            layer_norm=False,
            dropout_rate=0.0,
        ):
            """Tide model.

            Args:
              model_config: configurations specific to the model.
              pred_len: prediction horizon length.
              cat_sizes: number of categories in each categorical covariate.
              num_ts: number of time-series in the dataset
              transform: apply reversible transform or not.
              cat_emb_size: embedding size of categorical variables.
              layer_norm: use layer norm or not.
              dropout_rate: level of dropout.
            """
            super().__init__()
            self.model_config = model_config
            self.transform = transform
            if self.transform:
                self.affine_weight = self.add_weight(
                    name="affine_weight",
                    shape=(num_ts,),
                    initializer="ones",
                    trainable=True,
                )

                self.affine_bias = self.add_weight(
                    name="affine_bias",
                    shape=(num_ts,),
                    initializer="zeros",
                    trainable=True,
                )
            self.pred_len = pred_len
            self.encoder = _make_dnn_residual(
                model_config.get("hidden_dims"),
                layer_norm=layer_norm,
                dropout_rate=dropout_rate,
            )
            self.decoder = _make_dnn_residual(
                model_config.get("hidden_dims")[:-1]
                + [
                    model_config.get("decoder_output_dim") * self.pred_len,
                ],
                layer_norm=layer_norm,
                dropout_rate=dropout_rate,
            )
            self.linear = tf.keras.layers.Dense(
                self.pred_len,
                activation=None,
            )
            self.time_encoder = _make_dnn_residual(
                model_config.get("time_encoder_dims"),
                layer_norm=layer_norm,
                dropout_rate=dropout_rate,
            )
            self.final_decoder = MLPResidual(
                hidden_dim=model_config.get("final_decoder_hidden"),
                output_dim=1,
                layer_norm=layer_norm,
                dropout_rate=dropout_rate,
            )
            self.cat_embs = []
            for cat_size in cat_sizes:
                self.cat_embs.append(
                    tf.keras.layers.Embedding(
                        input_dim=cat_size, output_dim=cat_emb_size
                    )
                )
            self.ts_embs = tf.keras.layers.Embedding(input_dim=num_ts, output_dim=16)
            self.train_loss = keras.losses.MeanSquaredError()

        @tf.function
        def _assemble_feats(self, feats, cfeats):
            """assemble all features."""
            all_feats = [feats]
            for i, emb in enumerate(self.cat_embs):
                all_feats.append(tf.transpose(emb(cfeats[i, :])))
            return tf.concat(all_feats, axis=0)

        @tf.function
        def call(self, inputs):
            """Call function that takes in a batch of training data and features."""
            past_data = inputs[0]
            future_features = inputs[1]
            bsize = past_data[0].shape[0]
            tsidx = inputs[2]
            past_feats = self._assemble_feats(past_data[1], past_data[2])
            future_feats = self._assemble_feats(future_features[0], future_features[1])
            past_ts = past_data[0]
            if self.transform:
                affine_weight = tf.gather(self.affine_weight, tsidx)
                affine_bias = tf.gather(self.affine_bias, tsidx)
                batch_mean = tf.math.reduce_mean(past_ts, axis=1)
                batch_std = tf.math.reduce_std(past_ts, axis=1)
                batch_std = tf.where(
                    tf.math.equal(batch_std, 0.0), tf.ones_like(batch_std), batch_std
                )
                past_ts = (past_ts - batch_mean[:, None]) / batch_std[:, None]
                past_ts = affine_weight[:, None] * past_ts + affine_bias[:, None]
            encoded_past_feats = tf.transpose(
                self.time_encoder(tf.transpose(past_feats))
            )
            encoded_future_feats = tf.transpose(
                self.time_encoder(tf.transpose(future_feats))
            )
            enc_past = tf.repeat(tf.expand_dims(encoded_past_feats, axis=0), bsize, 0)
            enc_past = tf.reshape(enc_past, [bsize, -1])
            enc_fut = tf.repeat(
                tf.expand_dims(encoded_future_feats, axis=0), bsize, 0
            )  # batch x fdim x H
            enc_future = tf.reshape(enc_fut, [bsize, -1])
            residual_out = self.linear(past_ts)
            ts_embs = self.ts_embs(tsidx)
            encoder_input = tf.concat([past_ts, enc_past, enc_future, ts_embs], axis=1)
            encoding = self.encoder(encoder_input)
            decoder_out = self.decoder(encoding)
            decoder_out = tf.reshape(
                decoder_out, [bsize, -1, self.pred_len]
            )  # batch x d x H
            final_in = tf.concat([decoder_out, enc_fut], axis=1)
            out = self.final_decoder(tf.transpose(final_in, (0, 2, 1)))  # B x H x 1
            out = tf.squeeze(out, axis=-1)
            out += residual_out
            if self.transform:
                out = (out - affine_bias[:, None]) / (affine_weight[:, None] + EPS)
                out = out * batch_std[:, None] + batch_mean[:, None]
            return out

        @tf.function
        def train_step(self, past_data, future_features, ytrue, tsidx, optimizer):
            """One step of training."""
            with tf.GradientTape() as tape:
                all_preds = self((past_data, future_features, tsidx), training=True)
                loss = self.train_loss(ytrue, all_preds)

            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return loss

        def get_all_eval_data(self, data, mode, num_split=1):
            y_preds = []
            y_trues = []
            all_test_loss = 0
            all_test_num = 0
            idxs = np.arange(0, self.pred_len, self.pred_len // num_split).tolist() + [
                self.pred_len
            ]
            for i in range(len(idxs) - 1):
                indices = (idxs[i], idxs[i + 1])
                (
                    all_y_true,
                    all_y_pred,
                    test_loss,
                    test_num,
                ) = self.get_eval_data_for_split(data, mode, indices)
                y_preds.append(all_y_pred)
                y_trues.append(all_y_true)
                all_test_loss += test_loss
                all_test_num += test_num
            return np.hstack(y_preds), np.hstack(y_trues), all_test_loss / all_test_num

        def get_eval_data_for_split(self, data, mode, indices):
            iterator = data.tf_dataset(mode=mode)

            all_y_true = None
            all_y_pred = None

            def set_or_concat(a, b):
                if a is None:
                    return b
                return tf.concat((a, b), axis=1)

            all_test_loss = 0
            all_test_num = 0
            ts_count = 0
            ypreds = []
            ytrues = []
            for all_data in tqdm(iterator):
                past_data = all_data[:3]
                future_features = all_data[4:6]
                y_true = all_data[3]
                tsidx = all_data[-1]
                all_preds = self((past_data, future_features, tsidx), training=False)
                y_pred = all_preds
                y_pred = y_pred[:, 0 : y_true.shape[1]]
                id1 = indices[0]
                id2 = min(indices[1], y_true.shape[1])
                y_pred = y_pred[:, id1:id2]
                y_true = y_true[:, id1:id2]
                loss = self.train_loss(y_true, y_pred)
                all_test_loss += loss
                all_test_num += 1
                ts_count += y_true.shape[0]
                ypreds.append(y_pred)
                ytrues.append(y_true)
                if ts_count >= len(data.ts_cols):
                    ts_count = 0
                    ypreds = tf.concat(ypreds, axis=0)
                    ytrues = tf.concat(ytrues, axis=0)
                    all_y_true = set_or_concat(all_y_true, ytrues)
                    all_y_pred = set_or_concat(all_y_pred, ypreds)
                    ypreds = []
                    ytrues = []
            return (
                all_y_true.numpy(),
                all_y_pred.numpy(),
                all_test_loss.numpy(),
                all_test_num,
            )

        def evaluate(self, data, mode, num_split=1):
            all_y_pred, all_y_true, test_loss = self.get_all_eval_data(
                data, mode, num_split
            )

            result_dict = {}
            for metric in METRICS:
                eval_fn = METRICS[metric]
                result_dict[metric] = np.float64(eval_fn(all_y_pred, all_y_true))

            return (
                result_dict,
                (all_y_pred, all_y_true),
                test_loss,
            )


def mape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true).flatten()
    abs_val = np.abs(y_true).flatten()
    idx = np.where(abs_val > EPS)
    mpe = np.mean(abs_diff[idx] / abs_val[idx])
    return mpe


def mae_loss(y_pred, y_true):
    return np.abs(y_pred - y_true).mean()


def wape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true)
    abs_val = np.abs(y_true)
    wpe = np.sum(abs_diff) / (np.sum(abs_val) + EPS)
    return wpe


def smape(y_pred, y_true):
    abs_diff = np.abs(y_pred - y_true)
    abs_mean = (np.abs(y_true) + np.abs(y_pred)) / 2
    smpe = np.mean(abs_diff / (abs_mean + EPS))
    return smpe


def rmse(y_pred, y_true):
    return np.sqrt(np.square(y_pred - y_true).mean())


def nrmse(y_pred, y_true):
    mse = np.square(y_pred - y_true)
    return np.sqrt(mse.mean()) / np.abs(y_true).mean()


METRICS = {
    "mape": mape,
    "wape": wape,
    "smape": smape,
    "nrmse": nrmse,
    "rmse": rmse,
    "mae": mae_loss,
}


class TiDE(ModelObject):
    """Google Research MLP based forecasting approach."""

    def __init__(
        self,
        name: str = "UnivariateRegression",
        random_seed=42,
        frequency="D",
        learning_rate=0.0009999,
        transform=False,
        layer_norm=False,
        holiday=True,
        dropout_rate=0.3,
        batch_size=512,
        hidden_size=256,
        num_layers=1,
        hist_len=720,
        decoder_output_dim=16,
        final_decoder_hidden=64,
        num_split=4,
        min_num_epochs=0,
        train_epochs=100,
        patience=10,
        epoch_len=None,
        permute=True,
        normalize=True,
        gpu_index=0,
        n_jobs: int = "auto",
        forecast_length: int = 14,
        prediction_interval: float = 0.9,
        verbose: int = 0,
    ):
        ModelObject.__init__(
            self,
            name="TiDE",
            frequency=frequency,
            prediction_interval=prediction_interval,
            regression_type=None,
            holiday_country=None,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.forecast_length = forecast_length
        self.learning_rate = learning_rate
        self.transform = transform
        self.layer_norm = layer_norm
        self.holiday = holiday
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hist_len = hist_len
        self.decoder_output_dim = decoder_output_dim
        self.final_decoder_hidden = final_decoder_hidden
        self.num_split = num_split
        self.min_num_epochs = min_num_epochs
        self.train_epochs = train_epochs
        self.patience = patience
        self.epoch_len = epoch_len
        self.permute = permute
        self.normalize = normalize
        self.gpu_index = gpu_index

    def fit(self, df=None, num_cov_cols=None, cat_cov_cols=None, future_regressor=None):
        """Training TS code."""
        df = self.basic_profile(df)
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)
        keras.backend.clear_session()
        data_df = df.copy()

        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if False:
                tf.config.experimental.set_visible_devices(gpus[self.gpu_index], "GPU")
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        print(e)
        except Exception as e:
            print("TF GPU init error:" + repr(e))

        # some weird way of passing validation indexes which is silly for prod
        full_len_idx = data_df.shape[0]
        # will be tied to num_splits
        boundaries = [
            full_len_idx - self.forecast_length * 3,
            full_len_idx - self.forecast_length * 2,
            full_len_idx - self.forecast_length,
        ]

        if num_cov_cols is None and cat_cov_cols is None:
            ts_cols = data_df.columns
        else:
            num_cov_cols_temp = (
                [] if not isinstance(num_cov_cols, list) else num_cov_cols
            )
            cat_cov_cols_temp = (
                [] if not isinstance(cat_cov_cols, list) else cat_cov_cols
            )
            ts_cols = [
                col
                for col in data_df.columns
                if col not in set(num_cov_cols_temp + cat_cov_cols_temp)
            ]
        dtl = TimeSeriesdata(
            df=data_df,
            num_cov_cols=num_cov_cols,
            cat_cov_cols=None,
            ts_cols=np.array(ts_cols),
            train_range=[0, boundaries[0]],
            val_range=[boundaries[0], boundaries[1]],
            test_range=[boundaries[1], boundaries[2]],
            hist_len=self.hist_len,
            pred_len=self.forecast_length,
            batch_size=min(self.batch_size, len(ts_cols)),
            freq=self.frequency,
            normalize=self.normalize,
            epoch_len=self.epoch_len,
            holiday=self.holiday,
            permute=self.permute,
        )

        # Create model
        model_config = {
            "model_type": "dnn",
            "hidden_dims": [self.hidden_size] * self.num_layers,
            "time_encoder_dims": [64, 4],
            "decoder_output_dim": self.decoder_output_dim,
            "final_decoder_hidden": self.final_decoder_hidden,
            "batch_size": dtl.batch_size,
        }
        self.model = TideModel(
            model_config=model_config,
            pred_len=self.forecast_length,
            num_ts=len(ts_cols),
            cat_sizes=dtl.cat_sizes,
            transform=self.transform,
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
        )

        step = tf.Variable(0)
        # LR scheduling
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=30 * dtl.train_range[1],
        )

        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1e3)

        best_loss = np.inf
        pat = 0
        # best_check_path = None
        while step.numpy() < self.train_epochs + 1:
            ep = step.numpy()
            # sys.stdout.flush()

            iterator = dtl.tf_dataset(mode="train")
            for i, batch in enumerate(iterator):
                # print(f"i is {i}")
                past_data = batch[:3]
                future_features = batch[4:6]
                tsidx = batch[-1]
                loss = self.model.train_step(  # noqa
                    past_data, future_features, batch[3], tsidx, optimizer
                )
                # print(f"training epoch is {ep} with loss: {loss}")

            step.assign_add(1)
            # Test metrics
            val_metrics, val_res, val_loss = self.model.evaluate(
                dtl, "val", num_split=self.num_split
            )
            test_metrics, test_res, test_loss = self.model.evaluate(
                dtl, "test", num_split=self.num_split
            )
            tracked_loss = val_metrics["rmse"]
            if tracked_loss < best_loss and ep > self.min_num_epochs:
                best_loss = tracked_loss
                pat = 0
            else:
                pat += 1
                # early stopping
                if pat > self.patience:
                    break

            if self.verbose > 0:
                print(
                    f"epoch {ep} with mape {val_metrics['mape']} and rmse {val_metrics['rmse']}"
                )
        self.past_data = past_data
        self.future_features = future_features
        self.tsidx = tsidx
        self.dtl = dtl
        self.fit_runtime = datetime.datetime.now() - self.startTime

        return self

    def predict(
        self,
        forecast_length="self",
        future_regressor=None,
        mode="test",
        just_point_forecast=False,
    ):
        if forecast_length != "self":
            if forecast_length != self.forecast_length:
                raise ValueError("forecast_length must be set in TiDE init")
        predictStartTime = datetime.datetime.now()
        iterator = self.dtl.tf_dataset(mode=mode)
        *_, batch = iterator
        past_data = batch[:3]
        future_features = batch[4:6]
        tsidx = batch[-1]
        all_preds = self.model((past_data, future_features, tsidx), training=False)
        if self.normalize:
            arr = self.dtl.scaler.inverse_transform(all_preds.numpy().T)
        else:
            arr = all_preds.numpy().T
        fut_idx = pd.date_range(
            self.train_last_date, periods=self.forecast_length + 1, freq=self.frequency
        )[1:]
        forecast = pd.DataFrame(arr, columns=self.column_names, index=fut_idx)
        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.dtl.data_df[self.column_names],
                forecast,
                prediction_interval=self.prediction_interval,
                method="inferred_normal",
            )

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=self.forecast_length,
                forecast_index=forecast.index,
                forecast_columns=self.column_names,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )
            return prediction
        return forecast

    def get_new_params(self, method: str = "random"):
        """Return dict of new parameters for parameter tuning."""
        return {
            # borrowed mostly from the defaults used for the paper results
            "learning_rate": random.choices(
                [0.00006558, 0.000999999, 0.000030127, 0.01, 0.000001],
                [0.3, 0.3, 0.3, 0.05, 0.05],
            )[0],
            "transform": random.choices([True, False], [0.3, 0.7])[0],
            "layer_norm": random.choices([True, False], [0.2, 0.8])[0],
            "holiday": random.choices([True, False], [0.3, 0.7])[0],
            "dropout_rate": random.choices([0.0, 0.3, 0.5, 0.7], [0.3, 0.3, 0.3, 0.05])[
                0
            ],
            "batch_size": random.choices([1024, 512, 257, 32], [0.05, 0.4, 0.1, 0.1])[
                0
            ],
            "hidden_size": random.choices([1024, 512, 256, 64], [0.1, 0.2, 0.4, 0.1])[
                0
            ],
            "num_layers": random.choices([1, 2, 3], [0.85, 0.1, 0.02])[0],
            "hist_len": random.choices([720, 10, 364, 21], [0.1, 0.1, 0.1, 0.1])[0],
            "decoder_output_dim": random.choices([16, 8, 4, 32], [0.4, 0.4, 0.4, 0.1])[
                0
            ],
            "final_decoder_hidden": random.choices(
                [128, 64, 32, 16], [0.1, 0.8, 0.1, 0.1]
            )[0],
            "num_split": random.choices([6, 4, 1, 2], [0.1, 0.8, 0.2, 0.2])[0],
            "min_num_epochs": random.choices([5, 10, 20], [0.85, 0.1, 0.1])[0],
            "train_epochs": random.choices([20, 40, 100, 500], [0.2, 0.4, 0.2, 0.05])[
                0
            ],
            # "patience": random.choices([5000, 1000, 50000], [0.85, 0.05, 0.1])[0],
            "epoch_len": random.choices(
                [1, 4, 20, 40, None], [0.3, 0.2, 0.1, 0.1, 0.3]
            )[0],
            "permute": random.choices([True, False], [0.3, 0.7])[0],
            "normalize": random.choices([True, False], [0.5, 0.5])[0],
        }

    def get_params(self):
        """Return dict of current parameters."""
        return {
            "learning_rate": self.learning_rate,
            "transform": self.transform,
            "layer_norm": self.layer_norm,
            "holiday": self.holiday,
            "dropout_rate": self.dropout_rate,
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "hist_len": self.hist_len,
            "decoder_output_dim": self.decoder_output_dim,
            "final_decoder_hidden": self.final_decoder_hidden,
            "num_split": self.num_split,
            "min_num_epochs": self.min_num_epochs,
            "train_epochs": self.train_epochs,
            # "patience": self.patience,
            "epoch_len": self.epoch_len,
            "permute": self.permute,
            "normalize": self.normalize,
        }


"""
import matplotlib.pyplot as plt
from autots import load_daily

df = load_daily(long=False)
params = TiDE().get_new_params()
print(params)
mod = TiDE(forecast_length=60, **params)
mod = mod.fit(df=df)
forecast = mod.predict()
col = 'wiki_all'
combined = pd.concat([df.iloc[-100:], forecast.forecast], axis=0)
for col in df.columns:
    ax = combined[col].plot(title=col)
    ax.vlines(
        x=forecast.forecast.index[0],
        ls='-.',
        lw=1,
        colors='darkred',
        ymin=combined[col].min(),
        ymax=combined[col].max(),
    )
    plt.savefig(f"TiDE_{col}", dpi=300)
    plt.show()
"""
