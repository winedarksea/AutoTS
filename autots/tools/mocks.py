"""Fake/local imports for when libraries aren't available, to prevent the whole package from failing to load."""
import math
import numpy as np
import pandas as pd

class Module:  # fake version of torch.nn.Module
    def __init__(self, *args, **kwargs):
        # Store args for debugging if needed
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        # Do nothing, return input or None
        return args[0] if args else None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        return

    def train(self, mode=True):
        return self

    def eval(self):
        return self


class Dataset:  # fake version of torch.utils.data.Dataset
    def __init__(self, *args, **kwargs):
        # Store args for debugging if needed
        self.args = args
        self.kwargs = kwargs

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None


class DataLoader:  # fake version of torch.utils.data.DataLoader
    def __init__(self, dataset, *args, **kwargs):
        # Store dataset and other args for debugging if needed
        self.dataset = dataset
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0


class TensorDataset:  # fake version of torch.utils.data.TensorDataset
    def __init__(self, *tensors):
        # Store tensors for debugging if needed
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0]) if self.tensors else 0

    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)

tqdm = lambda x, **kwargs: x  # fake version of tqdm
curve_fit = lambda x: "scipy import failed"
butter = lambda x: "scipy import failed"
sosfiltfilt = lambda x: "scipy import failed"
savgol_filter = lambda x: "scipy import failed"
fftconvolve = lambda x: "scipy import failed"

class norm(object):  # fake version of scipy.stats.norm
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def ppf(self, q):
        """Simple normal quantile approximation using Abramowitz & Stegun rational approximation."""
        # Claude came up with this, so let's hope the LLM got it right.
        if q <= 0 or q >= 1:
            raise ValueError("q must be in (0, 1)")
        
        # Use symmetry: if q > 0.5, compute for 1-q and negate
        if q > 0.5:
            return -self.ppf(1 - q)
        
        # Rational approximation for q in (0, 0.5]
        t = math.sqrt(-2 * math.log(q))
        z = t - (2.515517 + 0.802853*t + 0.010328*t*t) / (1 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t)
        return z * self.scale + self.loc

    def cdf(self, x):
        """Cumulative distribution function using error function."""
        return 0.5 * (1 + math.erf((x - self.loc) / (self.scale * math.sqrt(2))))

class MinMaxScaler:  # local version of sklearn.preprocessing.MinMaxScaler
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_, self.scale_ = None, None
    
    def fit(self, X):
        X = np.array(X)
        self.min_ = X.min(axis=0)
        self.scale_ = X.max(axis=0) - self.min_
        self.scale_[self.scale_ == 0] = 1
        return self
    
    def transform(self, X):
        
        X = np.array(X)
        return (X - self.min_) / self.scale_ * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class StandardScaler:  # local version of sklearn.preprocessing.StandardScaler
    """
    StandardScaler that works with both numpy arrays and pandas DataFrames.
    Compatible with sklearn.preprocessing.StandardScaler interface.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.skip_columns = None
        self._is_dataframe = False
    
    def fit(self, X):
        """Compute the mean and standard deviation for each feature."""
        self._is_dataframe = isinstance(X, pd.DataFrame)
        
        if self._is_dataframe:
            self.mean_ = X.mean()
            self.scale_ = X.std(ddof=0).replace(0, 1)  # Use population standard deviation
            # Identify columns to skip (constant or zero std)
            self.skip_columns = (self.scale_ == 1)  # 0 replaced with 1, exact 1 unlikely in real data
        else:
            X = np.array(X)
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.scale_[self.scale_ == 0] = 1
            self.skip_columns = None
        return self
    
    def transform(self, X):
        """Scale the dataset using the stored mean and standard deviation."""
        if self._is_dataframe and isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            X_scaled = (X_copy - self.mean_) / self.scale_
            # Restore original values for columns that should not be scaled
            if self.skip_columns is not None:
                X_scaled.loc[:, self.skip_columns] = X_copy.loc[:, self.skip_columns]
            return X_scaled
        else:
            return (np.array(X) - self.mean_) / self.scale_
    
    def inverse_transform(self, X):
        """Revert the scaled data back to the original scale."""
        if self._is_dataframe and isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            X_original = (X_copy * self.scale_) + self.mean_
            # Restore original values for columns that were not scaled
            if self.skip_columns is not None:
                X_original.loc[:, self.skip_columns] = X_copy.loc[:, self.skip_columns]
            return X_original
        else:
            return (np.array(X) * self.scale_) + self.mean_
    
    def fit_transform(self, X):
        """Fit the scaler and transform the dataset."""
        return self.fit(X).transform(X)
