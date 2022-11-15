# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:26:27 2022

@author: Colin
"""
import unittest
import numpy as np
import pandas as pd
from autots.tools.impute import FillNA


class TestImpute(unittest.TestCase):

    def test_impute(self):
        print("Starting test_impute")
        df_nan = pd.DataFrame({
            'a': [5, 10, 15, np.nan, 10],
            'b': [5, 50, 15, np.nan, 10],
        })
        filled = FillNA(df_nan, method='ffill', window=10)
        self.assertTrue((filled.values.flatten() == np.array([5, 5, 10, 50, 15, 15, 15, 15, 10, 10])).all())

        filled = FillNA(df_nan, method='mean')
        self.assertTrue((filled.values.flatten() == np.array([5, 5, 10, 50, 15, 15, 10, 20, 10, 10])).all())

        filled = FillNA(df_nan, method='median')
        self.assertTrue((filled.values.flatten() == np.array([5, 5, 10, 50, 15, 15, 10, 12.5, 10, 10])).all())

        df_nan = pd.DataFrame({
            'a': [5, 10, 15, np.nan, 10],
            'b': [5, 50, 15, np.nan, 10],
        })
        filled = FillNA(df_nan, method='fake_date')
        self.assertTrue((filled.values.flatten() == np.array([5, 5, 5, 5, 10, 50, 15, 15, 10, 10])).all())

        df_nan = pd.DataFrame({
            'a': [5, 10, 15, np.nan, 10],
            'b': [5, 50, 15, np.nan, 10],
        })
        filled = FillNA(df_nan, method='fake_date_slice')
        self.assertTrue((filled.values.flatten() == np.array([5, 5, 10, 50, 15, 15, 10, 10])).all())
