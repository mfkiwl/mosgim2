import numpy as np
from mosgim2.utils.time_utils import sec_of_day
from mosgim2.consts.phys_consts import POLE_PHI, POLE_THETA  



# Geodetic to Geomagnetic transform: http://www.nerc-bas.ac.uk/uasd/instrums/magnet/gmrot.html
GEOGRAPHIC_TRANSFORM = np.array([
    [np.cos(POLE_THETA)*np.cos(POLE_PHI), np.cos(POLE_THETA)*np.sin(POLE_PHI), -np.sin(POLE_THETA)],
    [-np.sin(POLE_PHI), np.cos(POLE_PHI), 0],
    [np.sin(POLE_THETA)*np.cos(POLE_PHI), np.sin(POLE_THETA)*np.sin(POLE_PHI), np.cos(POLE_THETA)]
])


def subsol(year, doy, ut):
    '''Finds subsolar geocentric longitude and latitude.


    Parameters
    ==========
    year : int [1601, 2100]
        Calendar year
    doy : int [1, 365/366]
        Day of year
    ut : float
        Seconds since midnight on the specified day

    Returns
    =======
    sbsllon : float
        Subsolar longitude [rad] for the given date/time
    sbsllat : float
        Subsolar co latitude [rad] for the given date/time

    Notes
    =====

    Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    (U.S. Government Printing Office, 1994). Usable for years 1601-2100,
    inclusive. According to the Almanac, results are good to at least 0.01
    degree latitude and 0.025 degrees longitude between years 1950 and 2050.
    Accuracy for other years has not been tested. Every day is assumed to have
    exactly 86400 seconds; thus leap seconds that sometimes occur on December
    31 are ignored (their effect is below the accuracy threshold of the
    algorithm).

    After Fortran code by A. D. Richmond, NCAR. Translated from IDL
    by K. Laundal.

    '''

    from numpy import sin, cos, pi, arctan2, arcsin

    yr = year - 2000

    if year >= 2101:
        print('subsol.py: subsol invalid after 2100. Input year is:', year)

    nleap = np.floor((year-1601)/4)
    nleap = nleap - 99
    if year <= 1900:
        if year <= 1600:
            print('subsol.py: subsol invalid before 1601. Input year is:', year)
        ncent = np.floor((year-1601)/100)
        ncent = 3 - ncent
        nleap = nleap + ncent

    l0 = -79.549 + (-0.238699*(yr-4*nleap) + 3.08514e-2*nleap)

    g0 = -2.472 + (-0.2558905*(yr-4*nleap) - 3.79617e-2*nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut/86400 - 1.5) + doy

    # Addition to Mean longitude of Sun since January 1 of IYR:
    lf = 0.9856474*df

    # Addition to Mean anomaly since January 1 of IYR:
    gf = 0.9856003*df

    # Mean longitude of Sun:
    l = l0 + lf

    # Mean anomaly:
    g = g0 + gf
    grad = g*pi/180

    # Ecliptic longitude:
    lmbda = l + 1.915*sin(grad) + 0.020*sin(2*grad)
    lmrad = lmbda*pi/180
    sinlm = sin(lmrad)

    # Days (including fraction) since 12 UT on January 1 of 2000:
    n = df + 365*yr + nleap

    # Obliquity of ecliptic:
    epsilon = 23.439 - 4e-7*n
    epsrad = epsilon*pi/180

    # Right ascension:
    alpha = arctan2(cos(epsrad)*sinlm, cos(lmrad)) * 180/pi

    # Declination:
    delta = arcsin(sin(epsrad)*sinlm) * 180/pi

    # Subsolar latitude:
    sbsllat = delta

    # Equation of time (degrees):
    etdeg = l - alpha
    nrot = round(etdeg/360)
    etdeg = etdeg - 360*nrot

    # Apparent time (degrees):
    aptime = ut/240 + etdeg    # Earth rotates one degree every 240 s.

    # Subsolar longitude:
    sbsllon = 180 - aptime
    nrot = round(sbsllon/360)
    sbsllon = sbsllon - 360*nrot

    return np.deg2rad(sbsllon), np.pi / 2 - np.deg2rad(sbsllat)


def geo2mag(theta, phi, date):

    ut = sec_of_day(date)
    doy = date.timetuple().tm_yday
    year = date.year

    phi_sbs, theta_sbs = subsol(year, doy, ut)
    r_sbs = np.array([np.sin(theta_sbs) * np.cos(phi_sbs), np.sin(theta_sbs) * np.sin(phi_sbs), np.cos(theta_sbs)])

    r_sbs_mag = GEOGRAPHIC_TRANSFORM.dot(r_sbs)
    theta_sbs_m = np.arccos(r_sbs_mag[2])
    phi_sbs_m = np.arctan2(r_sbs_mag[1], r_sbs_mag[0])
    if phi_sbs_m < 0.:
        phi_sbs_m = phi_sbs_m + 2. * np.pi


    r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    r_mag = GEOGRAPHIC_TRANSFORM.dot(r)
    theta_m = np.arccos(r_mag[2])
    phi_m = np.arctan2(r_mag[1], r_mag[0])
    if phi_m < 0.:
        phi_m = phi_m + 2. * np.pi


    mlt = phi_m - phi_sbs_m + np.pi # np.radians(15.) * ut /3600. + phi_m + POLE_PHI  
    if mlt < 0.:
        mlt = mlt + 2. * np.pi
    if mlt > 2. * np.pi:
        mlt = mlt - 2. * np.pi

    return theta_m, mlt
geo2mag = np.vectorize(geo2mag)

