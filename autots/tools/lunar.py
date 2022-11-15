"""Phases of the moon.
Modified from https://stackoverflow.com/a/2531541/9492254
by keturn and earlier from John Walker
"""

from math import sin, cos, floor, sqrt, pi
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
