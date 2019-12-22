import numpy as np
import pandas as pd

class ModelObject(object):
    """
    Models should all have methods:
        .fit(df) (taking a DataFrame with DatetimeIndex and n columns of n timeseries)
        .predict(forecast_length = int)
    
    Args:
        name (str): Model Name
    """
    def __init__(self, name: str = "Uniniated Model Name"):
        self.name = name
    
    def __repr__(self):
        return self.name
    
def ModelMonster(model: str, parameters: dict):
    """Directs strings and parameters to appropriate model objects.
    
    Args:
        model (str): Name of Model Function
        parameters (dict): Dictionary of parameters to pass through to model
    """