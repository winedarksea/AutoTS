from os.path import dirname, join
import numpy as np
import pandas as pd

def load_toy_daily():
    """
    4 series of sample daily data from late 2019
    Testing some basic missing and categorical features.
    """
    module_path = dirname(__file__)
    data_file_name = join(module_path, 'data', 'toy_daily.csv')
    
    df_long = pd.read_csv(data_file_name)
    df_long['date'] = pd.to_datetime(df_long['date'], infer_datetime_format = True)

    return df_long