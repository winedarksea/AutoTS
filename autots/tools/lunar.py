"""Phases of the moon.
Modified from https://stackoverflow.com/a/2531541/9492254
by keturn and earlier from John Walker
"""
from math import sin, cos, floor, sqrt, pi, radians
import bisect
import numpy as np
import pandas as pd


def moon_phase(
    datetime_index,
    epsilon=1e-6,
    epoch=2444237.905,
    ecliptic_longitude_epoch=278.833540,
    ecliptic_longitude_perigee=282.596403,
    eccentricity=0.016718,
    moon_mean_longitude_epoch=64.975464,
    moon_mean_perigee_epoch=349.383063,
):
    """Numpy version. Takes a pd.DatetimeIndex and returns moon phase (%illuminated).
    Epoch can be adjust slightly (0.5 = half day) to adjust for time zones. This is for US. epoch=2444238.5 for Asia generally.
    """
    # set time to Noon if not otherwise given, as midnight is confusingly close to previous day
    if np.sum(datetime_index.hour) == 0:
        datetime_index = datetime_index + pd.Timedelta(hours=12)
    days = datetime_index.to_julian_date() - epoch

    # Mean anomaly of the Sun
    a = (360 / 365.2422) * days
    N = a - 360.0 * np.floor(a / 360.0)
    N = N + ecliptic_longitude_epoch - ecliptic_longitude_perigee
    # Convert from perigee coordinates to epoch 1980
    M = a - 360.0 * np.floor(N / 360.0)

    m = torad(M)
    e = m.copy()
    while 1:
        delta = e - eccentricity * np.sin(e) - m
        e = e - delta / (1.0 - eccentricity * np.cos(e))
        if abs(delta).max() <= epsilon:
            break

    Ec = sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(e / 2.0)
    # True anomaly
    Ec = 2 * todeg(np.arctan(Ec))
    # Suns's geometric ecliptic longuitude
    a = Ec + ecliptic_longitude_perigee
    lambda_sun = a - 360.0 * np.floor(a / 360.0)

    # Calculation of the Moon's position

    # Moon's mean longitude
    a = 13.1763966 * days + moon_mean_longitude_epoch
    moon_longitude = a - 360.0 * np.floor(a / 360.0)

    # Moon's mean anomaly
    a = moon_longitude - 0.1114041 * days - moon_mean_perigee_epoch
    MM = a - 360.0 * np.floor(a / 360.0)

    # Moon's ascending node mean longitude
    # MN = fixangle(c.node_mean_longitude_epoch - 0.0529539 * day)

    evection = 1.2739 * np.sin(torad(2 * (moon_longitude - lambda_sun) - MM))

    # Annual equation
    annual_eq = 0.1858 * np.sin(torad(M))

    # Correction term
    A3 = 0.37 * np.sin(torad(M))

    MmP = MM + evection - annual_eq - A3

    # Correction for the equation of the centre
    mEc = 6.2886 * np.sin(torad(MmP))

    # Another correction term
    A4 = 0.214 * np.sin(torad(2 * MmP))

    # Corrected longitude
    lP = moon_longitude + evection + mEc - annual_eq + A4

    # Variation
    variation = 0.6583 * np.sin(torad(2 * (lP - lambda_sun)))

    # True longitude
    lPP = lP + variation

    # Calculation of the phase of the Moon

    # Age of the Moon, in degrees
    moon_age = lPP - lambda_sun

    # Phase of the Moon
    moon_phase = (1 - np.cos(torad(moon_age))) / 2.0
    return moon_phase
    # return pd.Series(moon_phase, index=datetime_index)


def moon_phase_df(datetime_index, epoch=2444237.905):
    """Convert pandas DatetimeIndex to moon phases. Note timezone and hour can matter slightly.
    Epoch can be adjust slightly (0.5 = half day) to adjust for time zones.
    2444237.905 is for US Central. epoch=2444238.5 for Asia generally.
    """
    moon = pd.Series(moon_phase(datetime_index, epoch=epoch), index=datetime_index)
    full_moon = ((moon > moon.shift(1)) & (moon > moon.shift(-1))).astype(int)
    new_moon = ((moon < moon.shift(1)) & (moon < moon.shift(-1))).astype(int)
    # account for end (shift) being new_moon
    if new_moon.tail(29).sum() == 0:
        new_moon.iloc[-1] = 1
    if full_moon.tail(29).sum() == 0:
        full_moon.iloc[-1] = 1
    moon_df = pd.concat([moon, full_moon, new_moon], axis=1)
    moon_df.columns = ['phase', 'full_moon', 'new_moon']
    return moon_df


# Little mathematical functions
def fixangle(a):
    return a - 360.0 * floor(a / 360.0)


def torad(d):
    return d * pi / 180.0


def todeg(r):
    return r * 180.0 / pi


def dsin(d):
    return sin(torad(d))


def dcos(d):
    return cos(torad(d))


def kepler(m, ecc=0.016718):
    """Solve the equation of Kepler."""

    epsilon = 1e-6

    m = torad(m)
    e = m
    while 1:
        delta = e - ecc * sin(e) - m
        e = e - delta / (1.0 - ecc * cos(e))

        if abs(delta) <= epsilon:
            break

    return e


def phase_string(
    p, precision=0.05, new=0.0, first=0.25, full=0.4, last=0.75, nextnew=1.0
):
    phase_strings = (
        (new + precision, "new"),
        (first - precision, "waxing crescent"),
        (first + precision, "first quarter"),
        (full - precision, "waxing gibbous"),
        (full + precision, "full"),
        (last - precision, "waning gibbous"),
        (last + precision, "last quarter"),
        (nextnew - precision, "waning crescent"),
        (nextnew + precision, "new"),
    )

    i = bisect.bisect([a[0] for a in phase_strings], p)

    return phase_strings[i][1]


"""Phases of the moon.
Version based on "Astronomical Algorithms" by Jean Meeus.
Tested and did not notice significant difference from previous version. Epoch needs to be properly adjusted to align.
"""


def moon_phase_alternative(
    datetime_index,
    epoch=2451545.0,  # J2000.0
):
    """Numpy version. Takes a pd.DatetimeIndex and returns moon phase (%illuminated)."""
    # Handle single timestamp input
    if isinstance(datetime_index, pd.Timestamp):
        datetime_index = pd.DatetimeIndex([datetime_index])

    # set time to Noon if not otherwise given, as midnight is confusingly close to previous day
    if np.sum(datetime_index.hour) == 0:
        datetime_index = datetime_index + pd.Timedelta(hours=12)

    jd_series = datetime_index.to_julian_date()
    jd = jd_series.values if hasattr(jd_series, 'values') else np.array([jd_series])
    T = (jd - epoch) / 36525.0

    # Sun's mean longitude
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
    L0 = L0 - 360.0 * np.floor(L0 / 360.0)

    # Sun's mean anomaly
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
    M = M - 360.0 * np.floor(M / 360.0)
    M_rad = np.radians(M)

    # Sun's equation of center
    C = (
        (1.914602 - 0.004817 * T - 0.000014 * T**2) * np.sin(M_rad)
        + (0.019993 - 0.000101 * T) * np.sin(2 * M_rad)
        + 0.000289 * np.sin(3 * M_rad)
    )

    # Sun's true longitude
    lambda_sun = L0 + C

    # Moon's mean longitude
    L_prime = (
        218.3164477
        + 481267.88123421 * T
        - 0.0015786 * T**2
        + T**3 / 538841
        - T**4 / 65194000
    )
    L_prime = L_prime - 360.0 * np.floor(L_prime / 360.0)

    # Moon's mean elongation
    D = (
        297.8501921
        + 445267.1114034 * T
        - 0.0018819 * T**2
        + T**3 / 545868
        - T**4 / 113065000
    )
    D = D - 360.0 * np.floor(D / 360.0)

    # Sun's mean anomaly (already calculated as M)

    # Moon's mean anomaly
    M_prime = (
        134.9633964
        + 477198.8675055 * T
        + 0.0087414 * T**2
        + T**3 / 69699
        - T**4 / 14712000
    )
    M_prime = M_prime - 360.0 * np.floor(M_prime / 360.0)

    # Moon's argument of latitude
    F = (
        93.2720950
        + 483202.0175233 * T
        - 0.0036539 * T**2
        - T**3 / 3526000
        + T**4 / 863310000
    )
    F = F - 360.0 * np.floor(F / 360.0)

    D_rad = np.radians(D)
    M_rad = np.radians(M)
    M_prime_rad = np.radians(M_prime)
    F_rad = np.radians(F)

    # Perturbations in longitude (degrees)
    delta_l = (
        -1.274 * np.sin(M_prime_rad - 2 * D_rad)
        + 0.658 * np.sin(2 * D_rad)
        - 0.186 * np.sin(M_rad)
        - 0.059 * np.sin(2 * M_prime_rad - 2 * D_rad)
        - 0.057 * np.sin(M_prime_rad - 2 * D_rad + M_rad)
        + 0.053 * np.sin(M_prime_rad + 2 * D_rad)
        + 0.046 * np.sin(2 * D_rad - M_rad)
        + 0.041 * np.sin(M_prime_rad - M_rad)
        - 0.035 * np.sin(D_rad)
        - 0.031 * np.sin(M_prime_rad + M_rad)
        - 0.015 * np.sin(2 * F_rad - 2 * D_rad)
        + 0.011 * np.sin(M_prime_rad - 4 * D_rad)
    )

    # Moon's true longitude
    lambda_moon = L_prime + delta_l

    # Age of the Moon, in degrees
    moon_age = lambda_moon - lambda_sun

    # Phase of the Moon
    moon_phase = (1 - np.cos(np.radians(moon_age))) / 2.0

    # Return scalar if input was a single timestamp
    if isinstance(datetime_index, pd.DatetimeIndex) and len(datetime_index) == 1:
        return moon_phase[0]
    return moon_phase


def moon_phase_df_alternative(datetime_index, epoch=None):
    """Convert pandas DatetimeIndex to moon phases. Note timezone and hour can matter slightly.

    Args:
        datetime_index: pandas DatetimeIndex
        epoch: If provided, uses the approximate (legacy) method for backward compatibility.
               If None, uses the new more accurate Jean Meeus method.
    """
    if epoch is not None:
        # Use the approximate method for backward compatibility
        moon = pd.Series(
            moon_phase_approx(datetime_index, epoch=epoch), index=datetime_index
        )
    else:
        # Use the new accurate method
        moon = pd.Series(moon_phase(datetime_index), index=datetime_index)

    full_moon = ((moon > moon.shift(1)) & (moon > moon.shift(-1))).astype(int)
    new_moon = ((moon < moon.shift(1)) & (moon < moon.shift(-1))).astype(int)
    # account for end (shift) being new_moon
    if new_moon.tail(29).sum() == 0:
        new_moon.iloc[-1] = 1
    if full_moon.tail(29).sum() == 0:
        full_moon.iloc[-1] = 1
    moon_df = pd.concat([moon, full_moon, new_moon], axis=1)
    moon_df.columns = ['phase', 'full_moon', 'new_moon']
    return moon_df
