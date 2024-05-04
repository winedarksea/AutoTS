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
    x = np.arange(-1, 1, 2 / p)  # Adjusted to accommodate float 'p'
    wavelet = np.cos(frequency * np.pi * x) * np.exp(-(x**2) / (2 * sigma**2))
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
    x = np.arange(-1, 1, 2 / p)  # Adjusted to accommodate float 'p'
    real_part = np.cos(frequency * np.pi * x) * np.exp(-(x**2) / (2 * sigma**2))
    imag_part = np.sin(frequency * np.pi * x) * np.exp(-(x**2) / (2 * sigma**2))
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
    x = np.arange(-1, 1, 2 / p)  # Adjusted to accommodate float 'p'
    # Real component of the Morlet wavelet
    wavelet = np.cos(frequency * np.pi * x) * np.exp(-(x**2) / (2 * sigma**2))
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
    x = np.arange(-1, 1, 2 / p)  # Adjusted to accommodate float 'p'
    wavelet = (1 - x**2 / sigma**2) * np.exp(-(x**2) / (2 * sigma**2))
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
    coeffs = np.array(
        [
            (1 + np.sqrt(3)) / 4,
            (3 + np.sqrt(3)) / 4,
            (3 - np.sqrt(3)) / 4,
            (1 - np.sqrt(3)) / 4,
        ]
    )

    # Generate a base wavelet of the specified length `p`
    # To create the wavelet, replicate the coefficients to fit the desired period `p`
    base_wavelet = np.tile(coeffs, int(np.ceil(p / len(coeffs))))[:p]

    return base_wavelet


##############################################################################


def create_wavelet(t, p, sigma=1.0, phase_shift=0, wavelet_type="morlet"):
    """
    Create a real-valued wavelet based on real-world anchored time steps in t,
    with an additional phase shift and a choice of wavelet type.

    Parameters:
    - t (np.ndarray): Array of time steps (in days) from a specified origin.
    - p (float): The period of the wavelet in the same units as t (typically days).
    - sigma (float): Standard deviation for the Gaussian envelope.
    - phase_shift (float): Phase shift to adjust the position of the wavelet peak.
    - wavelet_type (str): Type of wavelet ('morlet' or 'ricker').

    Returns:
    - np.ndarray: The generated wavelet values for each time step.
    """
    x = (t + phase_shift) % p - p / 2  # Normalize and center t around 0

    if wavelet_type == "morlet":
        return np.cos(2 * np.pi * x / p) * np.exp(-(x**2) / (2 * sigma**2))
    elif wavelet_type == "ricker":
        # Ricker (Mexican Hat) wavelet calculation
        a = 2 * sigma**2
        return (1 - (x**2 / a)) * np.exp(-(x**2) / (2 * sigma**2))
    else:
        raise ValueError("Unsupported wavelet type. Choose 'morlet' or 'ricker'.")


def offset_wavelet(p, t, order=5, sigma=1.0, wavelet_type="morlet"):
    """
    Create an offset collection of wavelets with `order` offsets, ensuring that
    peaks are spaced p/order apart.

    Parameters:
    - p (float): Period of the wavelet in the same units as t (typically days).
    - t (np.ndarray): Array of time steps.
    - order (int): The number of offsets.
    - sigma (float): Standard deviation for the Gaussian envelope.
    - wavelet_type (str): Type of wavelet ('morlet' or 'ricker').

    Returns:
    - np.ndarray: A 2D array with `order` wavelets along axis 1.
    """
    wavelet_features = []
    phase_offsets = np.linspace(
        0, p, order, endpoint=False
    )  # Properly space phase shifts over one period

    for phase_shift in phase_offsets:
        wavelet = create_wavelet(t, p, sigma, phase_shift, wavelet_type)
        wavelet_features.append(wavelet)

    return np.stack(wavelet_features, axis=1)


if False:
    DTindex = pd.date_range("2020-01-01", "2024-01-01", freq="D")
    origin_ts = "2030-01-01"
    t = (DTindex - pd.Timestamp(origin_ts)).total_seconds() / 86400

    p = 7
    weekly_wavelets = offset_wavelet(
        p=p,  # Weekly period
        t=t,  # A full year (365 days)
        # origin_ts=origin_ts,
        order=7,  # One offset for each day of the week
        # frequency=2 * np.pi / p,  # Frequency for weekly pattern
        sigma=0.5,  # Smaller sigma for tighter weekly spread
        wavelet_type="morlet",
    )

    # Example for yearly seasonality
    p = 365.25
    yearly_wavelets = offset_wavelet(
        p=p,  # Yearly period
        t=t,  # Three full years
        # origin_ts=origin_ts,
        order=12,  # One offset for each month
        # frequency=2 * np.pi / p,  # Frequency for yearly pattern
        sigma=2.0,  # Larger sigma for broader yearly spread
        wavelet_type="morlet",
    )
    yearly_wavelets2 = offset_wavelet(
        p=p,  # Yearly period
        t=t[-100:],  # Three full years
        # origin_ts=origin_ts,
        order=12,  # One offset for each month
        # frequency=2 * np.pi / p,  # Frequency for yearly pattern
        sigma=2.0,  # Larger sigma for broader yearly spread
        wavelet_type="morlet",
    )
    print(np.allclose(yearly_wavelets[-100:], yearly_wavelets2))

    # Display wavelet patterns for visualization
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    pd.DataFrame(weekly_wavelets).plot(title="Weekly Wavelets", ax=plt.gca())
    pd.DataFrame(yearly_wavelets).plot(title="Yearly Wavelets", ax=plt.gca())
    plt.show()

    pd.DataFrame(weekly_wavelets[0:50]).plot(title="Weekly Wavelets", ax=plt.gca())
    plt.show()


##############################################################################


def continuous_db2_wavelet(t, p, order, sigma):
    # Normalize t to [0, 1) interval based on period p, scaled by order to include multiple cycles
    x = (order * t % p) / p
    if order % 3 == 0:
        x = x + 0.3
    gaussian_envelope = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
    sinusoidal_component = np.sin(2 * np.pi * x)
    wavelet = gaussian_envelope * sinusoidal_component
    return wavelet


def create_narrowing_wavelets(p, max_order, t, sigma=0.5):
    wavelets = []
    for order in range(1, max_order + 1):
        sigma = sigma / order  # Narrow the Gaussian envelope as order increases
        wavelet = continuous_db2_wavelet(t, p, order, sigma)
        wavelets.append(wavelet)
    return np.array(wavelets).T


if False:
    # Example usage
    DTindex = pd.date_range("2020-01-01", "2024-01-01", freq="D")
    origin_ts = "2020-01-01"
    t_full = (DTindex - pd.Timestamp(origin_ts)).total_seconds() / 86400

    p = 365.25  # Example period
    max_order = 5  # Example maximum order

    # Full set of wavelets
    wavelets = create_narrowing_wavelets(p, max_order, t_full)

    # Wavelets for the last 100 days
    t_subset = t_full[-100:]
    wavelet_short = create_narrowing_wavelets(p, max_order, t_subset)

    # Check if the last 100 days of the full series match the subset
    print(np.allclose(wavelets[-100:], wavelet_short))  # This should be true

    # Plotting the wavelets
    plt.figure(figsize=(12, 6))
    for i in range(max_order):
        plt.plot(DTindex[-100:], wavelets[-100:, i], label=f"Order {i+1}")
        plt.plot(
            DTindex[-100:],
            wavelet_short[:, i],
            label=f"Subset Order {i+1}",
            linestyle="--",
        )
    plt.title("Comparison of Full Wavelets and Subset")
    plt.legend()
    plt.show()
