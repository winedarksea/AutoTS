#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 08:21:47 2025

@author: colincatlin
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from autots.evaluator.validation import (
    extract_seasonal_val_periods,
    validate_num_validations,
    generate_validation_indices,
)

# Create a dummy DataFrame to use in tests.
def create_dummy_df(n_rows=10, n_cols=1):
    # Using a simple integer index here for clarity.
    data = np.random.randn(n_rows, n_cols)
    return pd.DataFrame(data, index=pd.date_range("2020-01-01", freq='D', periods=n_rows))


class TestValidationSegments(unittest.TestCase):

    def test_extract_seasonal_val_periods(self):
        # A string with digits should return the integer value
        self.assertEqual(extract_seasonal_val_periods("seasonal 364"), 364)
        self.assertEqual(extract_seasonal_val_periods("seasonal6"), 6)
        self.assertEqual(extract_seasonal_val_periods("s4eas3onal"), 43)
        # If no digits are found, int('') will raise a ValueError.
        with self.assertRaises(ValueError):
            extract_seasonal_val_periods("seasonal")  # no digits

    def test_validate_num_validations_backwards(self):
        # Using a non‐seasonal method, e.g. "backwards"
        df = create_dummy_df(n_rows=10)
        forecast_length = 2

        # For non‐seasonal methods, max_possible = df_rows / forecast_length.
        # Here 10/2 = 5.0; since 5.0 - 5 = 0 (not > min_allowed_train_percent),
        # max_possible becomes int(5) - 1 = 4.
        # When num_validations=2: since 4 >= (2+1), the input is accepted.
        result = validate_num_validations(
            validation_method="backwards",
            num_validations=2,
            df_wide_numeric=df,
            forecast_length=forecast_length,
            min_allowed_train_percent=0.5,
            verbose=0,
        )
        self.assertEqual(result, 2)

        # If too many validations are requested, it is reduced.
        result = validate_num_validations(
            validation_method="backwards",
            num_validations=5,
            df_wide_numeric=df,
            forecast_length=forecast_length,
            min_allowed_train_percent=0.5,
            verbose=0,
        )
        # Here 4 < (5+1)=6, so the function should reduce num_validations to max_possible - 1 = 4 - 1 = 3.
        self.assertEqual(result, 3)

    def test_validate_num_validations_seasonal(self):
        # For seasonal validations with extra period specified.
        df = create_dummy_df(n_rows=10)
        forecast_length = 2
        # For validation_method like "seasonal 4": the seasonal period is 4.
        # temp = df_rows + forecast_length = 10+2 = 12, so max_possible = 12/4 = 3.0.
        # With no fractional part > min_allowed_train_percent, max_possible becomes 3 - 1 = 2.
        # Now if we ask for 2 validations then 2 < (2+1)=3 so it is reduced to 2-1 = 1.
        result = validate_num_validations(
            validation_method="seasonal 4",
            num_validations=2,
            df_wide_numeric=df,
            forecast_length=forecast_length,
            min_allowed_train_percent=0.5,
            verbose=0,
        )
        self.assertEqual(result, 1)

        # When using "auto" or "max", check the returned values.
        result_auto = validate_num_validations(
            validation_method="seasonal 4",
            num_validations="auto",
            df_wide_numeric=df,
            forecast_length=forecast_length,
            min_allowed_train_percent=0.5,
            verbose=0,
        )
        # With max_possible = 2, the auto branch returns max_possible (i.e. 2).
        self.assertEqual(result_auto, 2)

        result_max = validate_num_validations(
            validation_method="seasonal 4",
            num_validations="max",
            df_wide_numeric=df,
            forecast_length=forecast_length,
            min_allowed_train_percent=0.5,
            verbose=0,
        )
        # In the "max" branch, returns max_possible - 1 = 2 - 1 = 1.
        self.assertEqual(result_max, 1)

    @patch("autots.tools.window_functions.retrieve_closest_indices")
    @patch("autots.tools.seasonal.GeneralTransformer")
    def test_generate_validation_indices_similarity(self, mock_transformer, mock_retrieve):
        # For similarity method, we patch GeneralTransformer and retrieve_closest_indices.
        df = create_dummy_df(n_rows=10)
        forecast_length = 1
        num_validations = 1
        validation_method = "similarity"
        validation_params = {}

        # Set up a dummy transformer that simply returns the input dataframe.
        # this is a llm suggestion. GeneralTransformer with empty params would do the same just fine
        transformer_instance = MagicMock()
        transformer_instance.fit_transform.side_effect = lambda df: df
        mock_transformer.return_value = transformer_instance

        # Set up a dummy retrieve_closest_indices:
        # Return a list of two numpy arrays (since num_validations+1 == 2).
        dummy_created_idx = [np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3, 4, 5])]
        mock_retrieve.return_value = dummy_created_idx

        result = generate_validation_indices(
            validation_method=validation_method,
            forecast_length=forecast_length,
            num_validations=num_validations,
            df_wide_numeric=df,
            validation_params=validation_params,
            verbose=0,
        )

        # We expect a list of length 2.
        self.assertEqual(len(result), 2)
        # For the first created index, the last value is 3 so we expect df.index[df.index <= 3].
        expected_first = df.index[df.index <= 3]
        expected_second = df.index[df.index <= 5]
        # Convert indices to lists for comparison.
        self.assertEqual(list(result[0]), list(expected_first))
        self.assertEqual(list(result[1]), list(expected_second))

    @patch("autots.seasonal_window_match")
    def test_generate_validation_indices_seasonal(self, mock_seasonal_match):
        # it might be worth rewriting this without the @patch

        # For seasonal method without extra period (i.e. exactly "seasonal")
        df = create_dummy_df(n_rows=10)
        forecast_length = 2
        num_validations = 2
        validation_method = "seasonal"
        validation_params = {}

        # Create a dummy return value for seasonal_window_match.
        # seasonal_window_match returns (test, _) where test is a 2D array.
        # We need to have test.T produce columns whose last element (x[-1]) gives a slicing index.
        # For example, let test = [[4, 5, 6],
        #                           [4, 5, 6]]
        # Then test.T gives columns [ [4,4], [5,5], [6,6] ]
        # and x[-1] will be 4, 5, and 6 respectively.
        dummy_test = np.array([[4, 5, 6],
                               [4, 5, 6]])
        mock_seasonal_match.return_value = (dummy_test, None)

        result = generate_validation_indices(
            validation_method=validation_method,
            forecast_length=forecast_length,
            num_validations=num_validations,
            df_wide_numeric=df,
            validation_params=validation_params,
            verbose=0,
        )

        # Expect three slices corresponding to x[-1] values of 4, 5, and 6.
        self.assertEqual(len(result), 3)
        expected_slices = [df.index[0:4], df.index[0:5], df.index[0:6]]
        for res, exp in zip(result, expected_slices):
            self.assertEqual(list(res), list(exp))

    def test_generate_validation_indices_backwards(self):
        # Test the backwards method (e.g., "backwards" or its variants)
        df = create_dummy_df(n_rows=10)
        forecast_length = 2
        num_validations = 2
        validation_method = "backwards"

        result = generate_validation_indices(
            validation_method=validation_method,
            forecast_length=forecast_length,
            num_validations=num_validations,
            df_wide_numeric=df,
            verbose=0,
        )
        # For "backwards", the code first sets validation_indexes = [df.index]
        # and then for each validation appends idx[0: n_rows - (y+1)*forecast_length].
        # With n_rows=10 and forecast_length=2:
        #  y=0 -> slice = df.index[0:10-2] = df.index[0:8]
        #  y=1 -> slice = df.index[0:10-4] = df.index[0:6]
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result[0]), list(df.index))
        self.assertEqual(list(result[1]), list(df.index[0:8]))
        self.assertEqual(list(result[2]), list(df.index[0:6]))

    def test_generate_validation_indices_even(self):
        # Test the even method
        df = create_dummy_df(n_rows=10)
        forecast_length = 2
        num_validations = 2
        validation_method = "even"

        result = generate_validation_indices(
            validation_method=validation_method,
            forecast_length=forecast_length,
            num_validations=num_validations,
            df_wide_numeric=df,
            verbose=0,
        )
        # For "even", the initial index is [df.index],
        # and then for y in range(num_validations):
        #   validation_size = floor((n_rows - forecast_length)/(num_validations+1))
        # Here: (10-2)/(3) = 8/3 -> floor(2.66) = 2.
        # Then:
        #   y=0: slice = df.index[0: 2*1 + 2] = df.index[0:4]
        #   y=1: slice = df.index[0: 2*2 + 2] = df.index[0:6]
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result[0]), list(df.index))
        self.assertEqual(list(result[1]), list(df.index[0:4]))
        self.assertEqual(list(result[2]), list(df.index[0:6]))

    def test_generate_validation_indices_seasonal_with_period(self):
        # Test the seasonal method when extra digits are provided (e.g., "seasonal 4")
        df = create_dummy_df(n_rows=10)
        forecast_length = 2
        num_validations = 2
        validation_method = "seasonal 4"

        result = generate_validation_indices(
            validation_method=validation_method,
            forecast_length=forecast_length,
            num_validations=num_validations,
            df_wide_numeric=df,
            verbose=0,
        )
        # Because "seasonal 4" is caught by the base-val branch,
        # the initial validation_indexes = [df.index]
        # then in the seasonal branch:
        #   seasonal_val_periods = 4.
        # For y=0: val_per = (1*4) - forecast_length = 4 - 2 = 2; then slice index = df.index[0: (10-2)] = df.index[0:8]
        # For y=1: val_per = (2*4) - forecast_length = 8 - 2 = 6; then slice index = df.index[0: (10-6)] = df.index[0:4]
        expected = [
            df.index,
            df.index[0:8],
            df.index[0:4],
        ]
        self.assertEqual(len(result), 3)
        for res, exp in zip(result, expected):
            self.assertEqual(list(res), list(exp))

    def test_generate_validation_indices_mixed_length(self):
        # Test the mixed_length validation method.
        df = create_dummy_df(n_rows=10)
        forecast_length = 2
        num_validations = 2
        validation_method = "mixed_length"

        result = generate_validation_indices(
            validation_method=validation_method,
            forecast_length=forecast_length,
            num_validations=num_validations,
            df_wide_numeric=df,
            verbose=0,
        )
        # For mixed_length, the loop iterates num_validations+1 times (i.e. 3 times):
        # count == 0: cut = int(10/2)=5, so tuple: (df.index[0:5], df.index[5:])
        # count == 1: cut = 10 - int(10/3)= 10 - 3 =7, so tuple: (df.index[0:7], df.index[7:])
        # count == 2: cut = 10 - (3*2)= 10 - 6 =4,
        #          so tuple: (df.index[0:4], df.index[4: 4+2] = df.index[4:6])
        self.assertEqual(len(result), 3)
        expected = [
            (df.index[0:5], df.index[5:]),
            (df.index[0:7], df.index[7:]),
            (df.index[0:4], df.index[4:6]),
        ]
        for (res1, res2), (exp1, exp2) in zip(result, expected):
            self.assertEqual(list(res1), list(exp1))
            self.assertEqual(list(res2), list(exp2))

    def test_generate_validation_indices_unknown(self):
        # If an unknown validation_method is provided, the function should raise a ValueError.
        df = create_dummy_df(n_rows=100)
        forecast_length = 2
        num_validations = 2
        validation_method = "unknown_method"

        with self.assertRaises(ValueError):
            generate_validation_indices(
                validation_method=validation_method,
                forecast_length=forecast_length,
                num_validations=num_validations,
                df_wide_numeric=df,
                verbose=0,
            )

    def test_validate_num_validations_short_data(self):
        # Test that if the data is too short to allow any validations, 0 is returned.
        df = create_dummy_df(n_rows=1)
        forecast_length = 2  # longer than data
        result = validate_num_validations(
            validation_method="backwards",
            num_validations=2,
            df_wide_numeric=df,
            forecast_length=forecast_length,
            verbose=0,
        )
        self.assertEqual(result, 0)

    def test_validate_num_validations_negative_input(self):
        # If a negative number is provided, abs() is taken and the result is adjusted if needed.
        df = create_dummy_df(n_rows=100)
        forecast_length = 2
        # Passing -1 will be converted to 1.
        result = validate_num_validations(
            validation_method="backwards",
            num_validations=-1,
            df_wide_numeric=df,
            forecast_length=forecast_length,
            verbose=0,
        )
        self.assertEqual(result, 1)


# if __name__ == "__main__":
#     unittest.main()
