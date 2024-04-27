import numpy as np
import pandas as pd


def create_gaussian_wavelet(p, frequency=3, sigma=1.0):
    """
    Create a Gaussian-modulated cosine wavelet with specified frequency and sigma.
    
    Parameters:
    - p (float): The period or length to generate the wavelet.
    - frequency (int): Frequency of the cosine wave.
    - sigma (float): Standard deviation for the Gaussian envelope.
    
    Returns:
    - np.ndarray: The generated Gaussian-modulated wavelet.
    """
    x = np.arange(-1, 1, 2/p)  # Adjusted to accommodate float 'p'
    wavelet = np.cos(frequency * np.pi * x) * np.exp(-x**2 / (2 * sigma**2))
    return wavelet


def create_morlet_wavelet(p, frequency=3, sigma=1.0):
    """
    Create a Morlet wavelet with specified frequency and sigma.
    
    Parameters:
    - p (float): The period or length to generate the wavelet.
    - frequency (int): Frequency of the cosine wave.
    - sigma (float): Standard deviation for the Gaussian envelope.
    
    Returns:
    - np.ndarray: The generated complex Morlet wavelet.
    """
    x = np.arange(-1, 1, 2/p)  # Adjusted to accommodate float 'p'
    real_part = np.cos(frequency * np.pi * x) * np.exp(-x**2 / (2 * sigma**2))
    imag_part = np.sin(frequency * np.pi * x) * np.exp(-x**2 / (2 * sigma**2))
    wavelet = real_part + 1j * imag_part  # Complex wavelet
    return wavelet


def create_real_morlet_wavelet(p, frequency=3, sigma=1.0):
    """
    Create a real-valued Morlet wavelet with specified frequency and sigma.
    
    Parameters:
    - p (float): The period or length to generate the wavelet.
    - frequency (int): Frequency of the cosine wave.
    - sigma (float): Standard deviation for the Gaussian envelope.
    
    Returns:
    - np.ndarray: The generated real Morlet wavelet.
    """
    x = np.arange(-1, 1, 2/p)  # Adjusted to accommodate float 'p'
    # Real component of the Morlet wavelet
    wavelet = np.cos(frequency * np.pi * x) * np.exp(-x**2 / (2 * sigma**2))
    return wavelet


def create_mexican_hat_wavelet(p, frequency=None, sigma=1.0):
    """
    Create a Mexican Hat wavelet (Ricker wavelet) with specified sigma.
    
    Parameters:
    - p (float): The period or length to generate the wavelet.
    - sigma (float): Standard deviation for the Gaussian envelope.
    
    Returns:
    - np.ndarray: The generated Mexican Hat wavelet.
    """
    x = np.arange(-1, 1, 2/p)  # Adjusted to accommodate float 'p'
    wavelet = (1 - x**2 / sigma**2) * np.exp(-x**2 / (2 * sigma**2))
    return wavelet


def create_haar_wavelet(p):
    """
    Create a Haar wavelet with specified period `p`.

    Parameters:
    - p (float): The period or length to generate the wavelet.
    
    Returns:
    - np.ndarray: The generated Haar wavelet.
    """
    if p <= 0:
        raise ValueError("The period `p` must be greater than zero.")

    # Create the Haar wavelet
    x = np.arange(0, p)  # Discrete points to create the wavelet
    # The Haar wavelet has a step function: +1 for the first half, -1 for the second half
    half = len(x) // 2
    wavelet = np.zeros(len(x))
    wavelet[:half] = 1
    wavelet[half:] = -1

    return wavelet


def create_daubechies_db2_wavelet(p):
    """
    Create a Daubechies db2 wavelet with specified period `p`.
    
    Parameters:
    - p (int): The period or length to generate the wavelet.
    
    Returns:
    - np.ndarray: The generated Daubechies db2 wavelet.
    """
    if p <= 0:
        raise ValueError("The period `p` must be greater than zero.")

    # Coefficients for the Daubechies db2 wavelet
    # These are the scaling coefficients for the db2 wavelet
    coeffs = np.array([
        (1 + np.sqrt(3)) / 4,
        (3 + np.sqrt(3)) / 4,
        (3 - np.sqrt(3)) / 4,
        (1 - np.sqrt(3)) / 4,
    ])

    # Generate a base wavelet of the specified length `p`
    # To create the wavelet, replicate the coefficients to fit the desired period `p`
    base_wavelet = np.tile(coeffs, int(np.ceil(p / len(coeffs))))[:p]
    
    return base_wavelet


def repeat_wavelet(p, t, frequency=3, sigma=1.0, wavelet='morlet'):
    """
    Create and repeat a wavelet of period `p` to achieve length `t`.
    
    Parameters:
    - p (float): The period of the wavelet.
    - t (int): Total length of the repeated wavelet.
    - frequency (int): Frequency of the wavelet.
    - sigma (float): Standard deviation for the Gaussian envelope.
    - wavelet (str): Type of wavelet to create ("morlet", "gaussian", "ricker").
    
    Returns:
    - np.ndarray: The repeated wavelet truncated to length `t`.
    """
    if p <= 0 or t <= 0:
        raise ValueError("`p` and `t` must be greater than zero.")

    # Create the base wavelet with the specified frequency and sigma
    if wavelet == 'morlet':
        base_wavelet = create_real_morlet_wavelet(p, frequency, sigma)
    elif wavelet == "ricker":
        base_wavelet = create_mexican_hat_wavelet(p, frequency, sigma)
    elif wavelet == "gaussian":
        base_wavelet = create_gaussian_wavelet(p, frequency, sigma)
    elif wavelet == "haar":
        base_wavelet = create_haar_wavelet(p)
    elif wavelet == "db2":
        base_wavelet = create_daubechies_db2_wavelet(p)
    else:
        raise ValueError(f"Unknown wavelet type: {wavelet}")

    # Calculate how many times the wavelet needs to be repeated to reach or exceed `t`
    num_repeats = int(np.ceil(t / len(base_wavelet)))

    # Repeat the wavelet and then truncate to the desired length `t`
    repeated_wavelet = np.tile(base_wavelet, num_repeats)[:t]

    return repeated_wavelet

def offset_wavelet(p, t, order=5, frequency=3, sigma=1.0, wavelet="morlet"):
    """
    Create an offset collection of wavelets with `order` offsets.
    
    Parameters:
    - p (float): The period of the base wavelet.
    - t (int): Total length of the wavelet feature collection.
    - order (int): The number of offsets.
    - frequency (int): Frequency of the wavelet.
    - sigma (float): Standard deviation for Gaussian envelope.
    - wavelet (str): Type of wavelet to use.
    
    Returns:
    - np.ndarray: A 2D array with `order` wavelets along axis 1.
    """
    if order <= 0 or p <= 0 or t <= 0:
        raise ValueError("`order`, `p`, and `t` must be greater than zero.")

    wavelet_features = []
    base_wavelet = repeat_wavelet(p, t, frequency, sigma, wavelet=wavelet)

    # To ensure the offsets align without gaps, we need to calculate the offset
    # so that it's a fraction of the period `p`.
    for n in range(order):
        # Offset based on the fraction of `p` considering `order`
        offset = int((n / order) * len(base_wavelet))  # Adjust offset calculation
        offset_wavelet = np.roll(base_wavelet, offset)
        wavelet_features.append(offset_wavelet)

    # Stack the offset wavelets to create a feature collection
    return np.stack(wavelet_features, axis=1)

# Test the adjusted code
res = offset_wavelet(p=365.25, t=365 * 3, order=5, frequency=3, sigma=1.0)
pd.DataFrame(res).plot()

# Example for weekly seasonality
weekly_wavelets = offset_wavelet(
    p=7,  # Weekly period
    t=365,  # A full year (365 days)
    order=7,  # One offset for each day of the week
    frequency=2 * np.pi / 7,  # Frequency for weekly pattern
    sigma=0.5,  # Smaller sigma for tighter weekly spread
    wavelet='morlet'
)

# Example for yearly seasonality
yearly_wavelets = offset_wavelet(
    p=365.25,  # Yearly period
    t=365 * 3,  # Three full years
    order=12,  # One offset for each month
    frequency=2 * np.pi / 365.25,  # Frequency for yearly pattern
    sigma=2.0,  # Larger sigma for broader yearly spread
    wavelet='morlet'
)

# Display wavelet patterns for visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
pd.DataFrame(weekly_wavelets).plot(title="Weekly Wavelets", ax=plt.gca())
pd.DataFrame(yearly_wavelets).plot(title="Yearly Wavelets", ax=plt.gca())
plt.show()


# Example for yearly seasonality in weekly data
weekly_yearly_wavelets = offset_wavelet(
    p=52,  # Weekly data for a year
    t=52 * 3,  # Data for three full years
    order=12,  # One offset for each month
    frequency=2 * np.pi / 52,  # Frequency for yearly pattern
    sigma=1.5,  # Sigma to capture broader patterns
    wavelet='morlet'  # Morlet wavelet for complex oscillations
)


def continuous_db2_wavelet(m):
    # Generate a range of values from 0 to m-1 to represent time
    x = np.linspace(0, 1, m)

    # Define the continuous wavelet function
    # This is an approximation for demonstration purposes
    # Gaussian-modulated sinusoid
    frequency = 1  # Frequency can be adjusted
    amplitude = 1  # Amplitude can be adjusted
    sigma = 0.1  # Controls the width of the Gaussian
    gaussian_envelope = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
    sinusoidal_component = np.sin(2 * np.pi * frequency * x)

    # Modulate the sinusoidal component with the Gaussian envelope
    wavelet = amplitude * gaussian_envelope * sinusoidal_component

    return wavelet


def create_narrowing_wavelets(p, max_order, t):
    """
    Create a 2D array of wavelets with increasing order of tiling.
    
    Parameters:
    - p (float): The period to generate wavelets within. Allows floating-point.
    - max_order (int): Maximum order of wavelets to create.
    
    Returns:
    - np.ndarray: A 2D array with each row representing a different order of wavelets.
    """
    if p <= 0 or max_order < 0:
        raise ValueError("`p` must be greater than zero and `max_order` must be non-negative.")

    # Ensure p is an integer for array operations
    p = int(p + 0.99)

    wavelets = []

    # Create wavelets for each order from 0 to max_order
    for order in range(max_order):
        m_freq = int((p / (order + 1)) + 0.99)
        # make it so some don't end in 0 at end of p in t
        if order > 0 and order // 3 == 0:
            m_freq = np.int(np.ceil(m_freq * 1.5))
        wave = continuous_db2_wavelet(m=m_freq)
        base_length = len(wave)
        repeats =  int(np.ceil(t / base_length))
        wavelet = np.tile(wave, repeats)[:t]
        wavelets.append(wavelet)

    # Convert the list of wavelets to a 2D array
    return np.array(wavelets).T


# TO DO
# make wavelets anchored to a time array t that will be continous across real time

import numpy as np
import pandas as pd

# Original continuous wavelet function
def continuous_db2_wavelet(m):
    # Generate a range of values from 0 to m-1 to represent time
    x = np.linspace(0, 1, m)

    # Define the continuous wavelet function
    frequency = 1  # Frequency can be adjusted
    amplitude = 1  # Amplitude can be adjusted
    sigma = 0.1  # Controls the width of the Gaussian
    gaussian_envelope = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
    sinusoidal_component = np.sin(2 * np.pi * frequency * x)

    # Modulate the sinusoidal component with the Gaussian envelope
    wavelet = amplitude * gaussian_envelope * sinusoidal_component

    return wavelet


# New function that uses a datetime index for anchoring
def create_narrowing_wavelets_anchored(p, max_order, datetime_index, origin_ts=None):
    """
    Create a 2D array of wavelets with increasing order of tiling, anchored to a specific datetime.
    
    Parameters:
    - p (float): The period to generate wavelets within. Allows floating-point.
    - max_order (int): Maximum order of wavelets to create.
    - t (int): The total length of the output.
    - datetime_index (pd.DatetimeIndex): The datetime index to anchor to.
    - origin_ts (pd.Timestamp): The base timestamp to calculate offset.
    
    Returns:
    - np.ndarray: A 2D array with each row representing a different order of wavelets.
    """
    if p <= 0 or max_order < 0:
        raise ValueError("`p` must be greater than zero and `max_order` must be non-negative.")

    # Ensure p is an integer for array operations
    p = int(p + 0.99)

    if origin_ts is None:
        # Default origin if not specified (e.g., Unix epoch)
        origin_ts = pd.Timestamp("2030-01-01")

    # Convert datetime index to a numeric representation (seconds from origin)
    offset_seconds = (datetime_index - origin_ts).total_seconds()
    t = len(datetime_index)

    # Compute the offset in terms of the period `p`
    base_offset = (offset_seconds % p).astype(int)[0]

    wavelets = []

    # Create wavelets for each order from 0 to max_order
    for order in range(max_order):
        m_freq = int((p / (order + 1)) + 0.99)

        # make it so some don't end in 0 at end of p in t
        if order > 0 and order % 3 == 0:
            m_freq = int(np.ceil(m_freq * 1.5))
        
        wave = continuous_db2_wavelet(m=m_freq)
        base_length = len(wave)
        repeats = int(np.ceil(t / base_length))

        # Create the wavelet with the appropriate offset
        wavelet = np.tile(wave, repeats)[base_offset:base_offset + t]
        wavelets.append(wavelet)

    # Convert the list of wavelets to a 2D array
    return pd.DataFrame(np.array(wavelets).T, index=datetime_index)
