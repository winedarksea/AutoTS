# -*- coding: utf-8 -*-
"""
Extracted from auto_ts.py, the functions to create validation segments.

Warning, these are used in AMFM, possibly other places. Avoid modification of function structures, if possible.

Created on Mon Jan 16 11:36:01 2023

@author: Colin
"""
import numpy as np
from autots.tools.transform import GeneralTransformer
from autots.tools.window_functions import retrieve_closest_indices
from autots.tools.seasonal import seasonal_window_match


def extract_seasonal_val_periods(validation_method):
    val_list = [x for x in str(validation_method) if x.isdigit()]
    seasonal_val_periods = int(''.join(val_list))
    return seasonal_val_periods


def validate_num_validations(
    validation_method="backwards",
    num_validations=2,
    df_wide_numeric=None,
    forecast_length=None,
    min_allowed_train_percent=0.5,
    verbose=0,
):
    """Check how many validations are possible given the length of the data. Beyond initial eval split which is always assumed."""
    if 'seasonal' in validation_method and validation_method != "seasonal":
        seasonal_val_periods = extract_seasonal_val_periods(validation_method)
        temp = df_wide_numeric.shape[0] + forecast_length
        max_possible = temp / seasonal_val_periods
    else:
        max_possible = (df_wide_numeric.shape[0]) / forecast_length
    # now adjusted for minimum % amount of training data required
    if (max_possible - np.floor(max_possible)) > min_allowed_train_percent:
        max_possible = int(max_possible)
    else:
        max_possible = int(max_possible) - 1
    # set auto and max validations
    if num_validations == "auto":
        num_validations = 3 if max_possible >= 4 else max_possible
    elif num_validations == "max":
        num_validations = 50 if max_possible > 51 else max_possible - 1
    # this still has the initial test segment as a validation segment, so -1
    elif max_possible < (num_validations + 1):
        num_validations = max_possible - 1
        if verbose >= 0:
            print(
                "Too many training validations for length of data provided, decreasing num_validations to {}".format(
                    num_validations
                )
            )
    else:
        num_validations = abs(int(num_validations))
    if num_validations <= 0:
        num_validations = 0
    return int(num_validations)


def generate_validation_indices(
    validation_method,
    forecast_length,
    num_validations,
    df_wide_numeric,
    validation_params={},
    preclean=None,
    verbose=0,
):
    """generate validation indices (equals num_validations + 1 as includes initial eval).

    Args:
        validation_method (str): 'backwards', 'even', 'similarity', 'seasonal', 'seasonal 364', etc.
        forecast_length (int): number of steps ahead for forecast
        num_validations (int): number of additional vals after first eval sample
        df_wide_numeric (pd.DataFrame): pandas DataFrame with a dt index and columns as time series
        preclean (dict): transformer dict, used for similarity cleaning
        verbose (int): verbosity
    """
    bval_list = ['backwards', 'back', 'backward']
    base_val_list = bval_list + ['even', 'Even']
    # num_validations = int(num_validations)

    # generate similarity matching indices (so it can fail now, not after all the generations)
    if validation_method == "similarity":
        sim_df = df_wide_numeric.copy()
        if preclean is None:
            params = {
                "fillna": "median",  # mean or median one of few consistent things
                "transformations": {"0": "MaxAbsScaler"},
                "transformation_params": {
                    "0": {},
                },
            }
            trans = GeneralTransformer(
                forecast_length=forecast_length, verbose=verbose, **params
            )
            sim_df = trans.fit_transform(sim_df)

        created_idx = retrieve_closest_indices(
            sim_df,
            num_indices=num_validations + 1,
            forecast_length=forecast_length,
            include_last=True,
            verbose=verbose,
            **validation_params,
            # **self.similarity_validation_params,
        )
        validation_indexes = [
            df_wide_numeric.index[df_wide_numeric.index <= indx[-1]]
            for indx in created_idx
        ]
        del sim_df
    elif validation_method == "seasonal":
        test, _ = seasonal_window_match(
            DTindex=df_wide_numeric.index,
            k=num_validations + 1,
            forecast_length=forecast_length,
            **validation_params,
            # **self.seasonal_validation_params,
        )
        validation_indexes = [df_wide_numeric.index[0 : x[-1]] for x in test.T]
    elif validation_method in base_val_list or (
        'seasonal' in validation_method and validation_method != "seasonal"
    ):
        validation_indexes = [df_wide_numeric.index]
    else:
        raise ValueError(
            f"Validation Method `{validation_method}` not recognized try 'backwards'"
        )

    if validation_method in bval_list:
        idx = df_wide_numeric.index
        shp0 = df_wide_numeric.shape[0]
        for y in range(num_validations):
            # gradually remove the end
            current_slice = idx[0 : shp0 - (y + 1) * forecast_length]
            validation_indexes.append(current_slice)
    elif validation_method in ['even', 'Even']:
        idx = df_wide_numeric.index
        # /num_validations biases it towards the last segment
        for y in range(num_validations):
            validation_size = len(idx) - forecast_length
            validation_size = validation_size / (num_validations + 1)
            validation_size = int(np.floor(validation_size))
            current_slice = idx[0 : validation_size * (y + 1) + forecast_length]
            validation_indexes.append(current_slice)
    elif 'seasonal' in validation_method and validation_method != "seasonal":
        idx = df_wide_numeric.index
        shp0 = df_wide_numeric.shape[0]
        seasonal_val_periods = extract_seasonal_val_periods(validation_method)
        for y in range(num_validations):
            val_per = (y + 1) * seasonal_val_periods
            if seasonal_val_periods < forecast_length:
                pass
            else:
                val_per = val_per - forecast_length
            val_per = shp0 - val_per
            current_slice = idx[0:val_per]
            validation_indexes.append(current_slice)
    return validation_indexes
