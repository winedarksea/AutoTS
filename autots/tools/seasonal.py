# -*- coding: utf-8 -*-
"""
seasonal

@author: Colin
"""
import random

def seasonal_int(include_one: bool = False):
    """Generate a random integer of typical seasonalities."""
    prob_dict = {
        'random_int': 0.1,
        1: 0.05,
        2: 0.05,
        4: 0.05,
        7: 0.15,
        10: 0.01,
        12: 0.1,
        24: 0.1,
        28: 0.1,
        60: 0.1,
        96: 0.04,
        168: 0.01,
        364: 0.1,
        1440: 0.01,
        420: 0.01,
        52: 0.01,
        84: 0.01,
    }
    lag = random.choices(
        list(prob_dict.keys()),
        list(prob_dict.values()),
        k=1,
    )[0]
    if not include_one and str(lag) == '1':
        lag = 'random_int'
    if lag == 'random_int':
        lag = random.randint(2, 100)
    return int(lag)
