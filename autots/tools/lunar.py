"""Phases of the moon.
Modified from https://stackoverflow.com/a/2531541/9492254
by keturn and earlier from John Walker
"""

from math import sin, cos, floor, sqrt, pi, tan, atan
import bisect
import datetime
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


class MoonPhase:
    """Describe the phase of the moon.

    Args:
        date - a DateTime instance
        phase - my phase, in the range 0.0 .. 1.0
        phase_text - a string describing my phase
        illuminated - the percentage of the face of the moon illuminated
        angular_diameter - as seen from Earth, in degrees.
        sun_angular_diameter - as seen from Earth, in degrees.

        new_date - the date of the most recent new moon
        q1_date - the date the moon reaches 1st quarter in this cycle
        full_date - the date of the full moon in this cycle
        q3_date - the date the moon reaches 3rd quarter in this cycle
        nextnew_date - the date of the next new moon
    """

    def __init__(self, date=None):
        """MoonPhase constructor.

        Give a date, as either a Julian Day Number or a DateTime
        object."""
        if date is None:
            date = datetime.datetime.now()
        if not isinstance(date, datetime.date):
            self.date = pd.to_datetime(date, infer_datetime_format=True)
        else:
            self.date = date

        self.__dict__.update(phase(self.date))

        self.phase_text = phase_string(self.phase)

    def __getattr__(self, a):
        if a in ["new_date", "q1_date", "full_date", "q3_date", "nextnew_date"]:

            (
                self.new_date,
                self.q1_date,
                self.full_date,
                self.q3_date,
                self.nextnew_date,
            ) = phase_hunt(self.date)

            return getattr(self, a)
        raise AttributeError(a)

    def __repr__(self):
        if type(self.date) is int:
            jdn = self.date
        else:
            jdn = self.date.jdn

        return "<%s(%d)>" % (self.__class__, jdn)

    def __str__(self):
        s = "%s for %s, %s (%%%.2f illuminated)" % (
            self.__class__,
            self.date.strftime("%Y-%m-%d"),
            self.phase_text,
            self.illuminated * 100,
        )

        return s


class AstronomicalConstants:

    # JDN stands for Julian Day Number
    # Angles here are in degrees

    # 1980 January 0.0 in JDN, but standard yields 4713 BC
    epoch = 2444238.5  # 2444239.5 (pd) 2444238.5 (original)

    # Ecliptic longitude of the Sun at epoch 1980.0
    ecliptic_longitude_epoch = 278.833540

    # Ecliptic longitude of the Sun at perigee
    ecliptic_longitude_perigee = 282.596403

    # Eccentricity of Earth's orbit
    eccentricity = 0.016718

    # Semi-major axis of Earth's orbit, in kilometers
    sun_smaxis = 1.49585e8

    # Sun's angular size, in degrees, at semi-major axis distance
    sun_angular_size_smaxis = 0.533128

    # Elements of the Moon's orbit, epoch 1980.0

    # Moon's mean longitude at the epoch
    moon_mean_longitude_epoch = 64.975464
    # Mean longitude of the perigee at the epoch
    moon_mean_perigee_epoch = 349.383063

    # Mean longitude of the node at the epoch
    node_mean_longitude_epoch = 151.950429

    # Inclination of the Moon's orbit
    moon_inclination = 5.145396

    # Eccentricity of the Moon's orbit
    moon_eccentricity = 0.054900

    # Moon's angular size at distance a from Earth
    moon_angular_size = 0.5181

    # Semi-mojor axis of the Moon's orbit, in kilometers
    moon_smaxis = 384401.0
    # Parallax at a distance a from Earth
    moon_parallax = 0.9507

    # Synodic month (new Moon to new Moon), in days
    synodic_month = 29.53058868

    # Base date for E. W. Brown's numbered series of lunations (1923 January 16)
    lunations_base = 2423436.0

    # Properties of the Earth
    earth_radius = 6378.16


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


def phase(phase_date=None):
    """Calculate phase of moon as a fraction:

    The argument is the time for which the phase is requested,
    expressed in either a datetime or by Julian Day Number.

    Returns a dictionary containing the terminator phase angle as a
    percentage of a full circle (i.e., 0 to 1), the illuminated
    fraction of the Moon's disc, the Moon's age in days and fraction,
    the distance of the Moon from the centre of the Earth, and the
    angular diameter subtended by the Moon as seen by an observer at
    the centre of the Earth."""
    if phase_date is None:
        phase_date = datetime.datetime.now()

    # Calculation of the Sun's position
    c = AstronomicalConstants()

    # date within the epoch
    # if hasattr(phase_date, "jdn"):
    #     day = phase_date.jdn - c.epoch
    # else:
    #     day = phase_date - c.epoch
    day = pd.Timestamp(phase_date).to_julian_date()
    day = day - c.epoch

    # Mean anomaly of the Sun
    N = fixangle((360 / 365.2422) * day)
    # Convert from perigee coordinates to epoch 1980
    M = fixangle(N + c.ecliptic_longitude_epoch - c.ecliptic_longitude_perigee)

    # Solve Kepler's equation
    Ec = kepler(M, c.eccentricity)
    Ec = sqrt((1 + c.eccentricity) / (1 - c.eccentricity)) * tan(Ec / 2.0)
    # True anomaly
    Ec = 2 * todeg(atan(Ec))
    # Suns's geometric ecliptic longuitude
    lambda_sun = fixangle(Ec + c.ecliptic_longitude_perigee)

    # Orbital distance factor
    F = (1 + c.eccentricity * cos(torad(Ec))) / (1 - c.eccentricity**2)

    # Distance to Sun in km
    sun_dist = c.sun_smaxis / F
    sun_angular_diameter = F * c.sun_angular_size_smaxis

    ########
    #
    # Calculation of the Moon's position

    # Moon's mean longitude
    moon_longitude = fixangle(13.1763966 * day + c.moon_mean_longitude_epoch)

    # Moon's mean anomaly
    MM = fixangle(moon_longitude - 0.1114041 * day - c.moon_mean_perigee_epoch)

    # Moon's ascending node mean longitude
    # MN = fixangle(c.node_mean_longitude_epoch - 0.0529539 * day)

    evection = 1.2739 * sin(torad(2 * (moon_longitude - lambda_sun) - MM))

    # Annual equation
    annual_eq = 0.1858 * sin(torad(M))

    # Correction term
    A3 = 0.37 * sin(torad(M))

    MmP = MM + evection - annual_eq - A3

    # Correction for the equation of the centre
    mEc = 6.2886 * sin(torad(MmP))

    # Another correction term
    A4 = 0.214 * sin(torad(2 * MmP))

    # Corrected longitude
    lP = moon_longitude + evection + mEc - annual_eq + A4

    # Variation
    variation = 0.6583 * sin(torad(2 * (lP - lambda_sun)))

    # True longitude
    lPP = lP + variation

    # Calculation of the phase of the Moon

    # Age of the Moon, in degrees
    moon_age = lPP - lambda_sun

    # Phase of the Moon
    moon_phase = (1 - cos(torad(moon_age))) / 2.0

    # Calculate distance of Moon from the centre of the Earth
    moon_dist = (c.moon_smaxis * (1 - c.moon_eccentricity**2)) / (
        1 + c.moon_eccentricity * cos(torad(MmP + mEc))
    )

    # Calculate Moon's angular diameter
    moon_diam_frac = moon_dist / c.moon_smaxis
    moon_angular_diameter = c.moon_angular_size / moon_diam_frac

    # Calculate Moon's parallax (unused?)
    # moon_parallax = c.moon_parallax / moon_diam_frac

    res = {
        "phase": fixangle(moon_age) / 360.0,
        "illuminated": moon_phase,
        "age": c.synodic_month * fixangle(moon_age) / 360.0,
        "distance": moon_dist,
        "angular_diameter": moon_angular_diameter,
        "sun_distance": sun_dist,
        "sun_angular_diameter": sun_angular_diameter,
    }
    return res


def phase_hunt(sdate):
    """Find time of phases of the moon which surround the current date.

    Five phases are found, starting and ending with the new moons
    which bound the current lunation.
    """

    # if not hasattr(sdate, "jdn"):
    #     sdate = DateTime.DateTimeFromJDN(sdate)

    adate = sdate + datetime.timedelta(days=-45)

    k1 = floor((adate.year + ((adate.month - 1) * (1.0 / 12.0)) - 1900) * 12.3685)

    nt1 = meanphase(adate, k1)
    adate = nt1

    sdate = sdate.jdn

    while 1:
        adate = adate + 29.53058868  # c.synodic_month
        k2 = k1 + 1
        nt2 = meanphase(adate, k2)
        if nt1 <= sdate < nt2:
            break
        nt1 = nt2
        k1 = k2

    phases = list(
        map(
            truephase,
            [k1, k1, k1, k1, k2],
            [0 / 4.0, 1 / 4.0, 2 / 4.0, 3 / 4.0, 0 / 4.0],
        )
    )

    return phases


def meanphase(sdate, k):
    """Calculates time of the mean new Moon for a given base date.

    This argument K to this function is the precomputed synodic month
    index, given by:

                        K = (year - 1900) * 12.3685

    where year is expressed as a year and fractional year.
    """

    # Time in Julian centuries from 1900 January 0.5
    if not hasattr(sdate, "jdn"):
        delta_t = sdate - datetime.datetime(1900, 1, 1, 12).jdn
        t = delta_t / 36525
    else:
        delta_t = sdate - datetime.datetime(1900, 1, 1, 12)
        t = delta_t.days / 36525

    # square for frequent use
    t2 = t * t
    # and cube
    t3 = t2 * t

    nt1 = (
        2415020.75933
        + 29.53058868 * k  # c.synodic_month * k
        + 0.0001178 * t2
        - 0.000000155 * t3
        + 0.00033 * dsin(166.56 + 132.87 * t - 0.009173 * t2)
    )

    return nt1


def truephase(k, tphase):
    """Given a K value used to determine the mean phase of the new
    moon, and a phase selector (0.0, 0.25, 0.5, 0.75), obtain the
    true, corrected phase time."""

    apcor = False

    # add phase to new moon time
    k = k + tphase
    # Time in Julian centuries from 1900 January 0.5
    t = k / 1236.85

    t2 = t * t
    t3 = t2 * t

    # Mean time of phase
    pt = (
        2415020.75933
        + 29.53058868 * k  # c.synodic_month * k
        + 0.0001178 * t2
        - 0.000000155 * t3
        + 0.00033 * dsin(166.56 + 132.87 * t - 0.009173 * t2)
    )

    # Sun's mean anomaly
    m = 359.2242 + 29.10535608 * k - 0.0000333 * t2 - 0.00000347 * t3

    # Moon's mean anomaly
    mprime = 306.0253 + 385.81691806 * k + 0.0107306 * t2 + 0.00001236 * t3

    # Moon's argument of latitude
    f = 21.2964 + 390.67050646 * k - 0.0016528 * t2 - 0.00000239 * t3

    if (tphase < 0.01) or (abs(tphase - 0.5) < 0.01):

        # Corrections for New and Full Moon

        pt = pt + (
            (0.1734 - 0.000393 * t) * dsin(m)
            + 0.0021 * dsin(2 * m)
            - 0.4068 * dsin(mprime)
            + 0.0161 * dsin(2 * mprime)
            - 0.0004 * dsin(3 * mprime)
            + 0.0104 * dsin(2 * f)
            - 0.0051 * dsin(m + mprime)
            - 0.0074 * dsin(m - mprime)
            + 0.0004 * dsin(2 * f + m)
            - 0.0004 * dsin(2 * f - m)
            - 0.0006 * dsin(2 * f + mprime)
            + 0.0010 * dsin(2 * f - mprime)
            + 0.0005 * dsin(m + 2 * mprime)
        )

        apcor = True
    elif (abs(tphase - 0.25) < 0.01) or (abs(tphase - 0.75) < 0.01):

        pt = pt + (
            (0.1721 - 0.0004 * t) * dsin(m)
            + 0.0021 * dsin(2 * m)
            - 0.6280 * dsin(mprime)
            + 0.0089 * dsin(2 * mprime)
            - 0.0004 * dsin(3 * mprime)
            + 0.0079 * dsin(2 * f)
            - 0.0119 * dsin(m + mprime)
            - 0.0047 * dsin(m - mprime)
            + 0.0003 * dsin(2 * f + m)
            - 0.0004 * dsin(2 * f - m)
            - 0.0006 * dsin(2 * f + mprime)
            + 0.0021 * dsin(2 * f - mprime)
            + 0.0003 * dsin(m + 2 * mprime)
            + 0.0004 * dsin(m - 2 * mprime)
            - 0.0003 * dsin(2 * m + mprime)
        )
        if tphase < 0.5:
            #  First quarter correction
            pt = pt + 0.0028 - 0.0004 * dcos(m) + 0.0003 * dcos(mprime)
        else:
            #  Last quarter correction
            pt = pt + -0.0028 + 0.0004 * dcos(m) - 0.0003 * dcos(mprime)
        apcor = True

    if not apcor:
        raise ValueError("TRUEPHASE called with invalid phase selector", tphase)

    # return datetime.DateTimeFromJDN(pt)
    return pd.to_datetime(pt, unit='D', origin='julian')
