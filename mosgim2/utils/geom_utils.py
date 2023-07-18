import numpy as np
from mosgim2.consts.phys_consts import *


def MF(el, IPPh):
    """
    :param el: elevation angle in rads
    """
    return 1./np.sqrt(1 - (RE * np.cos(el) / (RE + IPPh)) ** 2)


def sub_ionospheric(st_lat, st_lon, IPPh, az, el):
    """
    Calculates subionospheric point and delatas from site
    Parameters:
        st_lat, st_lon - site latitude and longitude in radians
        IPPh - ionposheric maximum height (m)
        az, el - azimuth and elevation of the site-sattelite line of sight in
            radians
    """
    psi = np.pi / 2 - el - np.arcsin(np.cos(el) * RE / (RE + IPPh))
    lat = np.arcsin(np.sin(st_lat) * np.cos(psi) + np.cos(st_lat) * np.sin(psi) * np.cos(az))
    lon = st_lon + np.arcsin(np.sin(psi) * np.sin(az) / np.cos(lat))
    
#    lon[lon > np.pi] = lon[lon > np.pi] - 2 * np.pi 
#    lon[lon < -np.pi] = lon[lon < -np.pi] + 2 * np.pi 
    return lat, lon
