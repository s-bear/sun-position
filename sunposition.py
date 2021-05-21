# The MIT License (MIT)
# 
# Copyright (c) 2021 Samuel Bear Powell
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
from datetime import datetime

_JIT = not os.environ.get('NUMBA_DISABLE_JIT',False)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser(prog='sunposition',description='Compute sun position parameters given the time and location')
    parser.add_argument('--test',dest='test',action='store_true',help='Run tests')
    parser.add_argument('--version',action='version',version='%(prog)s 1.0')
    parser.add_argument('--citation',dest='cite',action='store_true',help='Print citation information')
    parser.add_argument('-t,--time',dest='t',type=str,default='now',help='"now" or date and time (UTC) in "YYYY-MM-DD hh:mm:ss.ssssss" format or a (UTC) POSIX timestamp')
    parser.add_argument('-lat,--latitude',dest='lat',type=float,default=51.48,help='latitude, in decimal degrees, positive for north')
    parser.add_argument('-lon,--longitude',dest='lon',type=float,default=0.0,help='longitude, in decimal degrees, positive for east')
    parser.add_argument('-e,--elevation',dest='elev',type=float,default=0,help='elevation, in meters')
    parser.add_argument('-T,--temperature',dest='temp',type=float,default=14.6,help='temperature, in degrees celcius')
    parser.add_argument('-p,--pressure',dest='p',type=float,default=1013.0,help='atmospheric pressure, in millibar')
    parser.add_argument('-a,--atmos_refract',dest='a',type=float,default=0.5667,help='atmospheric refraction at sunrise and sunset, in degrees')
    parser.add_argument('-dt',type=float,default=0.0,help='difference between earth\'s rotation time (TT) and universal time (UT1)')
    parser.add_argument('-r,--radians',dest='rad',action='store_true',help='Output in radians instead of degrees')
    parser.add_argument('--csv',dest='csv',action='store_true',help='Comma separated values (time,dt,lat,lon,elev,temp,pressure,az,zen,RA,dec,H)')
    parser.add_argument('--jit',dest='jit',action='store_true',help='Enable Numba acceleration (jit compilation time may overwhelm speed-up)')
    args = parser.parse_args()
    if args.cite:
        print("Implementation: Samuel Bear Powell, 2016")
        print("Algorithm:")
        print("Ibrahim Reda, Afshin Andreas, \"Solar position algorithm for solar radiation applications\", Solar Energy, Volume 76, Issue 5, 2004, Pages 577-589, ISSN 0038-092X, doi:10.1016/j.solener.2003.12.003")
        sys.exit(0)
    
    _JIT = args.jit
    
    if args.t == "now":
        args.t = datetime.utcnow()
    elif ":" in args.t and "-" in args.t:
        try:
            args.t = datetime.strptime(args.t,'%Y-%m-%d %H:%M:%S.%f') #with microseconds
        except:
            try:
                args.t = datetime.strptime(args.t,'%Y-%m-%d %H:%M:%S.') #without microseconds
            except:
                args.t = datetime.strptime(args.t,'%Y-%m-%d %H:%M:%S')
    else:
        args.t = datetime.utcfromtimestamp(int(args.t))

if _JIT:
    try:
        import numba
        from numba import jit
        _JIT = (numba.config.DISABLE_JIT == 0)

        @jit(nopython=True)
        def _polyval(p, x):
            y = 0.0
            for i,v in enumerate(p):
                y = y*x + v
            return y
    except:
        _JIT = False

if not _JIT:
    def jit(*args, **kwargs):
        #decorator that does nothing
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # called as @decorator
            return args[0]
        else:
            # called as @decorator(*args, **kwargs)
            return jit
    _polyval = np.polyval

def _calendar_time(dt):
    try:
        x = dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond
        return x
    except AttributeError:
        try:
            return _calendar_time(datetime.utcfromtimestamp(dt)) #will raise OSError if dt is not acceptable
        except:
            raise TypeError('dt must be datetime object or POSIX timestamp')

@jit(nopython=True)
def _julian_day(dt):
    """Calculate the Julian Day from a (year, month, day, hour, minute, second, microsecond) tuple"""
    # year and month numbers
    yr, mo, dy, hr, mn, sc, us = dt
    if mo <= 2:  # From paper: "if M = 1 or 2, then Y = Y - 1 and M = M + 12"
        mo += 12
        yr -= 1
    # day of the month with decimal time
    dy = dy + hr/24.0 + mn/(24.0*60.0) + sc/(24.0*60.0*60.0) + us/(24.0*60.0*60.0*1e6)
    # b is equal to 0 for the julian calendar and is equal to (2- A +
    # INT(A/4)), A = INT(Y/100), for the gregorian calendar
    a = int(yr / 100)
    b = 2 - a + int(a / 4)
    jd = int(365.25 * (yr + 4716)) + int(30.6001 * (mo + 1)) + dy + b - 1524.5
    return jd

@jit(nopython=True)
def _julian_ephemeris_day(jd, deltat):
    """Calculate the Julian Ephemeris Day from the Julian Day and delta-time = (terrestrial time - universal time) in seconds"""
    return jd + deltat / 86400.0

@jit(nopython=True)
def _julian_century(jd):
    """Caluclate the Julian Century from Julian Day or Julian Ephemeris Day"""
    return (jd - 2451545.0) / 36525.0

@jit(nopython=True)
def _julian_millennium(jc):
    """Calculate the Julian Millennium from Julian Ephemeris Century"""
    return jc / 10.0

@jit(nopython=True)
def _cos_sum(x, coeffs):
    y = np.zeros(len(coeffs))
    for i, abc in enumerate(coeffs):
        for a,b,c in abc:
            y[i] += a*np.cos(b + c*x)
    return y

# Earth Heliocentric Longitude coefficients (L0, L1, L2, L3, L4, and L5 in paper)
_EHL = (
    #L5:
    np.array([(1.0, 3.14, 0.0)]),
    #L4:
    np.array([(114.0, 3.142, 0.0), (8.0, 4.13, 6283.08), (1.0, 3.84, 12566.15)]),
    #L3:
    np.array([(289.0, 5.844, 6283.076), (35.0, 0.0, 0.0,), (17.0, 5.49, 12566.15),
    (3.0, 5.2, 155.42), (1.0, 4.72, 3.52), (1.0, 5.3, 18849.23),
    (1.0, 5.97, 242.73)]),
    #L2:
    np.array([(52919.0, 0.0, 0.0), (8720.0, 1.0721, 6283.0758), (309.0, 0.867, 12566.152),
    (27.0, 0.05, 3.52), (16.0, 5.19, 26.3), (16.0, 3.68, 155.42),
    (10.0, 0.76, 18849.23), (9.0, 2.06, 77713.77), (7.0, 0.83, 775.52),
    (5.0, 4.66, 1577.34), (4.0, 1.03, 7.11), (4.0, 3.44, 5573.14),
    (3.0, 5.14, 796.3), (3.0, 6.05, 5507.55), (3.0, 1.19, 242.73),
    (3.0, 6.12, 529.69), (3.0, 0.31, 398.15), (3.0, 2.28, 553.57),
    (2.0, 4.38, 5223.69), (2.0, 3.75, 0.98)]),
    #L1:
    np.array([(628331966747.0, 0.0, 0.0), (206059.0, 2.678235, 6283.07585), (4303.0, 2.6351, 12566.1517),
    (425.0, 1.59, 3.523), (119.0, 5.796, 26.298), (109.0, 2.966, 1577.344),
    (93.0, 2.59, 18849.23), (72.0, 1.14, 529.69), (68.0, 1.87, 398.15),
    (67.0, 4.41, 5507.55), (59.0, 2.89, 5223.69), (56.0, 2.17, 155.42),
    (45.0, 0.4, 796.3), (36.0, 0.47, 775.52), (29.0, 2.65, 7.11),
    (21.0, 5.34, 0.98), (19.0, 1.85, 5486.78), (19.0, 4.97, 213.3),
    (17.0, 2.99, 6275.96), (16.0, 0.03, 2544.31), (16.0, 1.43, 2146.17),
    (15.0, 1.21, 10977.08), (12.0, 2.83, 1748.02), (12.0, 3.26, 5088.63),
    (12.0, 5.27, 1194.45), (12.0, 2.08, 4694), (11.0, 0.77, 553.57),
    (10.0, 1.3, 3286.6), (10.0, 4.24, 1349.87), (9.0, 2.7, 242.73),
    (9.0, 5.64, 951.72), (8.0, 5.3, 2352.87), (6.0, 2.65, 9437.76),
    (6.0, 4.67, 4690.48)]),
    #L0:
    np.array([(175347046.0, 0.0, 0.0), (3341656.0, 4.6692568, 6283.07585), (34894.0, 4.6261, 12566.1517),
    (3497.0, 2.7441, 5753.3849), (3418.0, 2.8289, 3.5231), (3136.0, 3.6277, 77713.7715),
    (2676.0, 4.4181, 7860.4194), (2343.0, 6.1352, 3930.2097), (1324.0, 0.7425, 11506.7698),
    (1273.0, 2.0371, 529.691), (1199.0, 1.1096, 1577.3435), (990.0, 5.233, 5884.927),
    (902.0, 2.045, 26.298), (857.0, 3.508, 398.149), (780.0, 1.179, 5223.694),
    (753.0, 2.533, 5507.553), (505.0, 4.583, 18849.228), (492.0, 4.205, 775.523),
    (357.0, 2.92, 0.067), (317.0, 5.849, 11790.629), (284.0, 1.899, 796.298),
    (271.0, 0.315, 10977.079), (243.0, 0.345, 5486.778), (206.0, 4.806, 2544.314),
    (205.0, 1.869, 5573.143), (202.0, 2.4458, 6069.777), (156.0, 0.833, 213.299),
    (132.0, 3.411, 2942.463), (126.0, 1.083, 20.775), (115.0, 0.645, 0.98),
    (103.0, 0.636, 4694.003), (102.0, 0.976, 15720.839), (102.0, 4.267, 7.114),
    (99.0, 6.21, 2146.17), (98.0, 0.68, 155.42), (86.0, 5.98, 161000.69),
    (85.0, 1.3, 6275.96), (85.0, 3.67, 71430.7), (80.0, 1.81, 17260.15),
    (79.0, 3.04, 12036.46), (71.0, 1.76, 5088.63), (74.0, 3.5, 3154.69),
    (74.0, 4.68, 801.82), (70.0, 0.83, 9437.76), (62.0, 3.98, 8827.39),
    (61.0, 1.82, 7084.9), (57.0, 2.78, 6286.6), (56.0, 4.39, 14143.5),
    (56.0, 3.47, 6279.55), (52.0, 0.19, 12139.55), (52.0, 1.33, 1748.02),
    (51.0, 0.28, 5856.48), (49.0, 0.49, 1194.45), (41.0, 5.37, 8429.24),
    (41.0, 2.4, 19651.05), (39.0, 6.17, 10447.39), (37.0, 6.04, 10213.29),
    (37.0, 2.57, 1059.38), (36.0, 1.71, 2352.87), (36.0, 1.78, 6812.77),
    (33.0, 0.59, 17789.85), (30.0, 0.44, 83996.85), (30.0, 2.74, 1349.87),
    (25.0, 3.16, 4690.48)])
)

@jit(nopython=True)
def _heliocentric_longitude(jme):
    """Compute the Earth Heliocentric Longitude (L) in degrees given the Julian Ephemeris Millennium"""
    #L5, ..., L0
    Li = _cos_sum(jme, _EHL)
    L = _polyval(Li, jme) / 1e8
    L = np.rad2deg(L) % 360
    return L

#Earth Heliocentric Latitude coefficients (B0 and B1 in paper)
_EHB = ( 
    #B1:
    np.array([(9.0, 3.9, 5507.55), (6.0, 1.73, 5223.69)]),
    #B0:
    np.array([(280.0, 3.199, 84334.662), (102.0, 5.422, 5507.553), (80.0, 3.88, 5223.69),
    (44.0, 3.7, 2352.87), (32.0, 4.0, 1577.34)])
)

@jit(nopython=True)
def _heliocentric_latitude(jme):
    """Compute the Earth Heliocentric Latitude (B) in degrees given the Julian Ephemeris Millennium"""
    Bi = _cos_sum(jme, _EHB)
    B = _polyval(Bi, jme) / 1e8
    B = np.rad2deg(B) % 360
    return B

#Earth Heliocentric Radius coefficients (R0, R1, R2, R3, R4)
_EHR = (
    #R4:
    np.array([(4.0, 2.56, 6283.08)]),
    #R3:
    np.array([(145.0, 4.273, 6283.076), (7.0, 3.92, 12566.15)]),
    #R2:
    np.array([(4359.0, 5.7846, 6283.0758), (124.0, 5.579, 12566.152), (12.0, 3.14, 0.0),
    (9.0, 3.63, 77713.77), (6.0, 1.87, 5573.14), (3.0, 5.47, 18849)]),
    #R1:
    np.array([(103019.0, 1.10749, 6283.07585), (1721.0, 1.0644, 12566.1517), (702.0, 3.142, 0.0),
    (32.0, 1.02, 18849.23), (31.0, 2.84, 5507.55), (25.0, 1.32, 5223.69),
    (18.0, 1.42, 1577.34), (10.0, 5.91, 10977.08), (9.0, 1.42, 6275.96),
    (9.0, 0.27, 5486.78)]),
    #R0:
    np.array([(100013989.0, 0.0, 0.0), (1670700.0, 3.0984635, 6283.07585), (13956.0, 3.05525, 12566.1517),
    (3084.0, 5.1985, 77713.7715), (1628.0, 1.1739, 5753.3849), (1576.0, 2.8469, 7860.4194),
    (925.0, 5.453, 11506.77), (542.0, 4.564, 3930.21), (472.0, 3.661, 5884.927),
    (346.0, 0.964, 5507.553), (329.0, 5.9, 5223.694), (307.0, 0.299, 5573.143),
    (243.0, 4.273, 11790.629), (212.0, 5.847, 1577.344), (186.0, 5.022, 10977.079),
    (175.0, 3.012, 18849.228), (110.0, 5.055, 5486.778), (98.0, 0.89, 6069.78),
    (86.0, 5.69, 15720.84), (86.0, 1.27, 161000.69), (85.0, 0.27, 17260.15),
    (63.0, 0.92, 529.69), (57.0, 2.01, 83996.85), (56.0, 5.24, 71430.7),
    (49.0, 3.25, 2544.31), (47.0, 2.58, 775.52), (45.0, 5.54, 9437.76),
    (43.0, 6.01, 6275.96), (39.0, 5.36, 4694), (38.0, 2.39, 8827.39),
    (37.0, 0.83, 19651.05), (37.0, 4.9, 12139.55), (36.0, 1.67, 12036.46),
    (35.0, 1.84, 2942.46), (33.0, 0.24, 7084.9), (32.0, 0.18, 5088.63),
    (32.0, 1.78, 398.15), (28.0, 1.21, 6286.6), (28.0, 1.9, 6279.55),
    (26.0, 4.59, 10447.39)])
)

@jit(nopython=True)
def _heliocentric_radius(jme):
    """Compute the Earth Heliocentric Radius (R) in astronimical units given the Julian Ephemeris Millennium"""
    
    Ri = _cos_sum(jme, _EHR)
    R = _polyval(Ri, jme) / 1e8
    return R

@jit(nopython=True)
def _heliocentric_position(jme):
    """Compute the Earth Heliocentric Longitude, Latitude, and Radius given the Julian Ephemeris Millennium
        Returns (L, B, R) where L = longitude in degrees, B = latitude in degrees, and R = radius in astronimical units
    """
    return _heliocentric_longitude(jme), _heliocentric_latitude(jme), _heliocentric_radius(jme)

@jit(nopython=True)
def _geocentric_position(helio_pos):
    """Compute the geocentric latitude (Theta) and longitude (beta) (in degrees) of the sun given the earth's heliocentric position (L, B, R)"""
    L,B,R = helio_pos
    th = L + 180
    b = -B
    return (th, b)

@jit(nopython=True)
def _ecliptic_obliquity(jme, delta_epsilon):
    """Calculate the true obliquity of the ecliptic (epsilon, in degrees) given the Julian Ephemeris Millennium and the obliquity"""
    u = jme/10
    e0 = _polyval((2.45, 5.79, 27.87, 7.12, -39.05, -249.67, -51.38, 1999.25, -1.55, -4680.93, 84381.448), u)
    e = e0/3600.0 + delta_epsilon
    return e

#Nutation Longitude and Obliquity coefficients (Y)
_NLO_Y = np.array([(0.0,   0.0,   0.0,   0.0,   1.0), (-2.0,  0.0,   0.0,   2.0,   2.0), (0.0,   0.0,   0.0,   2.0,   2.0),
        (0.0,   0.0,   0.0,   0.0,   2.0), (0.0,   1.0,   0.0,   0.0,   0.0), (0.0,   0.0,   1.0,   0.0,   0.0),
        (-2.0,  1.0,   0.0,   2.0,   2.0), (0.0,   0.0,   0.0,   2.0,   1.0), (0.0,   0.0,   1.0,   2.0,   2.0),
        (-2.0,  -1.0,  0.0,   2.0,   2.0), (-2.0,  0.0,   1.0,   0.0,   0.0), (-2.0,  0.0,   0.0,   2.0,   1.0),
        (0.0,   0.0,   -1.0,  2.0,   2.0), (2.0,   0.0,   0.0,   0.0,   0.0), (0.0,   0.0,   1.0,   0.0,   1.0),
        (2.0,   0.0,   -1.0,  2.0,   2.0), (0.0,   0.0,   -1.0,  0.0,   1.0), (0.0,   0.0,   1.0,   2.0,   1.0),
        (-2.0,  0.0,   2.0,   0.0,   0.0), (0.0,   0.0,   -2.0,  2.0,   1.0), (2.0,   0.0,   0.0,   2.0,   2.0),
        (0.0,   0.0,   2.0,   2.0,   2.0), (0.0,   0.0,   2.0,   0.0,   0.0), (-2.0,  0.0,   1.0,   2.0,   2.0),
        (0.0,   0.0,   0.0,   2.0,   0.0), (-2.0,  0.0,   0.0,   2.0,   0.0), (0.0,   0.0,   -1.0,  2.0,   1.0),
        (0.0,   2.0,   0.0,   0.0,   0.0), (2.0,   0.0,   -1.0,  0.0,   1.0), (-2.0,  2.0,   0.0,   2.0,   2.0),
        (0.0,   1.0,   0.0,   0.0,   1.0), (-2.0,  0.0,   1.0,   0.0,   1.0), (0.0,   -1.0,  0.0,   0.0,   1.0),
        (0.0,   0.0,   2.0,   -2.0,  0.0), (2.0,   0.0,   -1.0,  2.0,   1.0), (2.0,   0.0,   1.0,   2.0,   2.0),
        (0.0,   1.0,   0.0,   2.0,   2.0), (-2.0,  1.0,   1.0,   0.0,   0.0), (0.0,   -1.0,  0.0,   2.0,   2.0),
        (2.0,   0.0,   0.0,   2.0,   1.0), (2.0,   0.0,   1.0,   0.0,   0.0), (-2.0,  0.0,   2.0,   2.0,   2.0),
        (-2.0,  0.0,   1.0,   2.0,   1.0), (2.0,   0.0,   -2.0,  0.0,   1.0), (2.0,   0.0,   0.0,   0.0,   1.0),
        (0.0,   -1.0,  1.0,   0.0,   0.0), (-2.0,  -1.0,  0.0,   2.0,   1.0), (-2.0,  0.0,   0.0,   0.0,   1.0),
        (0.0,   0.0,   2.0,   2.0,   1.0), (-2.0,  0.0,   2.0,   0.0,   1.0), (-2.0,  1.0,   0.0,   2.0,   1.0),
        (0.0,   0.0,   1.0,   -2.0,  0.0), (-1.0,  0.0,   1.0,   0.0,   0.0), (-2.0,  1.0,   0.0,   0.0,   0.0),
        (1.0,   0.0,   0.0,   0.0,   0.0), (0.0,   0.0,   1.0,   2.0,   0.0), (0.0,   0.0,   -2.0,  2.0,   2.0),
        (-1.0,  -1.0,  1.0,   0.0,   0.0), (0.0,   1.0,   1.0,   0.0,   0.0), (0.0,   -1.0,  1.0,   2.0,   2.0),
        (2.0,   -1.0,  -1.0,  2.0,   2.0), (0.0,   0.0,   3.0,   2.0,   2.0), (2.0,   -1.0,  0.0,   2.0,   2.0)])

#Nutation Longitude and Obliquity coefficients (a,b)
_NLO_AB = np.array([(-171996.0, -174.2), (-13187.0, -1.6), (-2274.0, -0.2), (2062.0, 0.2), (1426.0, -3.4), (712.0, 0.1),
        (-517.0, 1.2), (-386.0, -0.4), (-301.0, 0.0), (217.0, -0.5), (-158.0, 0.0), (129.0, 0.1),
        (123.0, 0.0), (63.0,  0.0), (63.0,  0.1), (-59.0, 0.0), (-58.0, -0.1), (-51.0, 0.0),
        (48.0,  0.0), (46.0,  0.0), (-38.0, 0.0), (-31.0, 0.0), (29.0,  0.0), (29.0,  0.0),
        (26.0,  0.0), (-22.0, 0.0), (21.0,  0.0), (17.0,  -0.1), (16.0,  0.0), (-16.0, 0.1),
        (-15.0, 0.0), (-13.0, 0.0), (-12.0, 0.0), (11.0,  0.0), (-10.0, 0.0), (-8.0,  0.0),
        (7.0,   0.0), (-7.0,  0.0), (-7.0,  0.0), (-7.0,  0.0), (6.0,   0.0), (6.0,   0.0),
        (6.0,   0.0), (-6.0,  0.0), (-6.0,  0.0), (5.0,   0.0), (-5.0,  0.0), (-5.0,  0.0),
        (-5.0,  0.0), (4.0,   0.0), (4.0,   0.0), (4.0,   0.0), (-4.0,  0.0), (-4.0,  0.0),
        (-4.0,  0.0), (3.0,   0.0), (-3.0,  0.0), (-3.0,  0.0), (-3.0,  0.0), (-3.0,  0.0),
        (-3.0,  0.0), (-3.0,  0.0), (-3.0,  0.0)])
#Nutation Longitude and Obliquity coefficients (c,d)
_NLO_CD = np.array([(92025.0,   8.9), (5736.0,    -3.1), (977.0, -0.5), (-895.0,    0.5),
        (54.0,  -0.1), (-7.0,  0.0), (224.0, -0.6), (200.0, 0.0),
        (129.0, -0.1), (-95.0, 0.3), (0.0,   0.0), (-70.0, 0.0),
        (-53.0, 0.0), (0.0,   0.0), (-33.0, 0.0), (26.0,  0.0),
        (32.0,  0.0), (27.0,  0.0), (0.0,   0.0), (-24.0, 0.0),
        (16.0,  0.0), (13.0,  0.0), (0.0,   0.0), (-12.0, 0.0),
        (0.0,   0.0), (0.0,   0.0), (-10.0, 0.0), (0.0,   0.0),
        (-8.0,  0.0), (7.0,   0.0), (9.0,   0.0), (7.0,   0.0),
        (6.0,   0.0), (0.0,   0.0), (5.0,   0.0), (3.0,   0.0),
        (-3.0,  0.0), (0.0,   0.0), (3.0,   0.0), (3.0,   0.0),
        (0.0,   0.0), (-3.0,  0.0), (-3.0,  0.0), (3.0,   0.0),
        (3.0,   0.0), (0.0,   0.0), (3.0,   0.0), (3.0,   0.0),
        (3.0,   0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
        (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
        (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
        (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])

@jit(nopython=True)
def _nutation_obliquity(jce):
    """compute the nutation in longitude (delta_psi) and the true obliquity (epsilon) given the Julian Ephemeris Century"""
    #mean elongation of the moon from the sun, in radians:
    #x0 = 297.85036 + 445267.111480*jce - 0.0019142*(jce**2) + (jce**3)/189474
    x0 = np.deg2rad(_polyval((1./189474, -0.0019142, 445267.111480, 297.85036),jce))
    #mean anomaly of the sun (Earth), in radians:
    x1 = np.deg2rad(_polyval((-1/3e5, -0.0001603, 35999.050340, 357.52772), jce))
    #mean anomaly of the moon, in radians:
    x2 = np.deg2rad(_polyval((1./56250, 0.0086972, 477198.867398, 134.96298), jce))
    #moon's argument of latitude, in radians:
    x3 = np.deg2rad(_polyval((1./327270, -0.0036825, 483202.017538, 93.27191), jce))
    #Longitude of the ascending node of the moon's mean orbit on the ecliptic
    # measured from the mean equinox of the date, in radians
    x4 = np.deg2rad(_polyval((1./45e4, 0.0020708, -1934.136261, 125.04452), jce))

    x = np.array([x0, x1, x2, x3, x4])

    a,b = _NLO_AB.T
    c,d = _NLO_CD.T
    dp = np.sum((a + b*jce)*np.sin(np.dot(_NLO_Y, x)))/36e6
    de = np.sum((c + d*jce)*np.cos(np.dot(_NLO_Y, x)))/36e6
    
    e = _ecliptic_obliquity(_julian_millennium(jce), de)

    return dp, e

@jit(nopython=True)
def _abberation_correction(R):
    """Calculate the abberation correction (delta_tau, in degrees) given the Earth Heliocentric Radius (in AU)"""
    return -20.4898/(3600*R)

@jit(nopython=True)
def _sun_longitude(helio_pos, delta_psi):
    """Calculate the apparent sun longitude (lambda, in degrees) and geocentric latitude (beta, in degrees) given the earth heliocentric position and delta_psi"""
    L,B,R = helio_pos
    theta = L + 180 #geocentric longitude
    beta = -B #geocentric latitude
    ll = theta + delta_psi + _abberation_correction(R)
    return ll, beta

@jit(nopython=True)
def _greenwich_sidereal_time(jd, delta_psi, epsilon):
    """Calculate the apparent Greenwich sidereal time (v, in degrees) given the Julian Day"""
    jc = _julian_century(jd)
    #mean sidereal time at greenwich, in degrees:
    v0 = (280.46061837 + 360.98564736629*(jd - 2451545) + 0.000387933*(jc**2) - (jc**3)/38710000) % 360
    v = v0 + delta_psi*np.cos(np.deg2rad(epsilon))
    return v

@jit(nopython=True)
def _sun_ra_decl(llambda, epsilon, beta):
    """Calculate the sun's geocentric right ascension (alpha, in degrees) and declination (delta, in degrees)"""
    l = np.deg2rad(llambda)
    e = np.deg2rad(epsilon)
    b = np.deg2rad(beta)
    alpha = np.arctan2(np.sin(l)*np.cos(e) - np.tan(b)*np.sin(e), np.cos(l)) #x1 / x2
    alpha = np.rad2deg(alpha) % 360
    delta = np.arcsin(np.sin(b)*np.cos(e) + np.cos(b)*np.sin(e)*np.sin(l))
    delta = np.rad2deg(delta)
    return alpha, delta

@jit(nopython=True)
def _sun_topo_ra_decl_hour(latitude, longitude, elevation, jd, delta_t = 0):
    """Calculate the sun's topocentric right ascension (alpha'), declination (delta'), and hour angle (H')"""
    
    jde = _julian_ephemeris_day(jd, delta_t)
    jce = _julian_century(jde)
    jme = _julian_millennium(jce)

    helio_pos = _heliocentric_position(jme)
    R = helio_pos[-1]
    phi, E = np.deg2rad(latitude), elevation
    #equatorial horizontal parallax of the sun, in radians
    xi = np.deg2rad(8.794/(3600*R)) #
    #rho = distance from center of earth in units of the equatorial radius
    #phi-prime = geocentric latitude
    #NB: These equations look like their based on WGS-84, but are rounded slightly
    # The WGS-84 reference ellipsoid has major axis a = 6378137 m, and flattening factor 1/f = 298.257223563
    # minor axis b = a*(1-f) = 6356752.3142 = 0.996647189335*a
    u = np.arctan(0.99664719*np.tan(phi)) #
    x = np.cos(u) + E*np.cos(phi)/6378140 #rho sin(phi-prime)
    y = 0.99664719*np.sin(u) + E*np.sin(phi)/6378140 #rho cos(phi-prime)

    delta_psi, epsilon = _nutation_obliquity(jce) #

    llambda, beta = _sun_longitude(helio_pos, delta_psi) #
    
    alpha, delta = _sun_ra_decl(llambda, epsilon, beta) #

    v = _greenwich_sidereal_time(jd, delta_psi, epsilon) #

    H = v + longitude - alpha #
    Hr, dr = np.deg2rad(H), np.deg2rad(delta)

    dar = np.arctan2(-x*np.sin(xi)*np.sin(Hr), np.cos(dr)-x*np.sin(xi)*np.cos(Hr))
    delta_alpha = np.rad2deg(dar) #
    
    alpha_prime = alpha + delta_alpha #
    delta_prime = np.rad2deg(np.arctan2((np.sin(dr) - y*np.sin(xi))*np.cos(dar), np.cos(dr) - y*np.sin(xi)*np.cos(Hr))) #
    H_prime = H - delta_alpha #

    return alpha_prime, delta_prime, H_prime

@jit(nopython=True)
def _sun_topo_azimuth_zenith(latitude, delta_prime, H_prime, temperature=14.6, pressure=1013, atmos_refract=0.5667):
    """Compute the sun's topocentric azimuth and zenith angles
    azimuth is measured eastward from north, zenith from vertical
    temperature = average temperature in C (default is 14.6 = global average in 2013)
    pressure = average pressure in mBar (default 1013 = global average)
    """
    SUN_RADIUS = 0.26667
    phi = np.deg2rad(latitude)
    dr, Hr = np.deg2rad(delta_prime), np.deg2rad(H_prime)
    P, T = pressure, temperature
    e0 = np.rad2deg(np.arcsin(np.sin(phi)*np.sin(dr) + np.cos(phi)*np.cos(dr)*np.cos(Hr)))
    delta_e = 0.0
    if e0 >= -1*(SUN_RADIUS + atmos_refract):
        tmp = np.deg2rad(e0 + 10.3/(e0+5.11))
        delta_e = (P/1010.0)*(283.0/(273+T))*(1.02/(60*np.tan(tmp)))

    e = e0 + delta_e
    zenith = 90 - e

    gamma = np.rad2deg(np.arctan2(np.sin(Hr), np.cos(Hr)*np.sin(phi) - np.tan(dr)*np.cos(phi))) % 360
    Phi = (gamma + 180) % 360 #azimuth from north
    return Phi, zenith, delta_e

@jit(nopython=True)
def _norm_lat_lon(lat,lon):
    if lat < -90 or lat > 90:
        #convert to cartesian and back
        x = np.cos(np.deg2rad(lon))*np.cos(np.deg2rad(lat))
        y = np.sin(np.deg2rad(lon))*np.cos(np.deg2rad(lat))
        z = np.sin(np.deg2rad(lat))
        r = np.sqrt(x**2 + y**2 + z**2)
        lon = np.rad2deg(np.arctan2(y,x)) % 360
        lat = np.rad2deg(np.arcsin(z/r))
    elif lon < 0 or lon > 360:
        lon = lon % 360
    return lat,lon

@jit(nopython=True)
def _topo_pos(jd,lat,lon,elev,dt,radians):
    """compute RA,dec,H, all in degrees"""
    lat,lon = _norm_lat_lon(lat,lon)
    RA, dec, H = _sun_topo_ra_decl_hour(lat, lon, elev, jd, dt)
    if radians:
        return np.deg2rad(RA), np.deg2rad(dec), np.deg2rad(H)
    else:
        return RA, dec, H

_topo_pos_v = np.vectorize(_topo_pos)

@jit(nopython=True)
def _pos(jd,lat,lon,elev,temp,press,atmos_refract,dt,radians):
    """Compute azimuth,zenith,RA,dec,H"""
    lat,lon = _norm_lat_lon(lat,lon)
    RA, dec, H = _sun_topo_ra_decl_hour(lat, lon, elev, jd, dt)
    azimuth, zenith, delta_e = _sun_topo_azimuth_zenith(lat, dec, H, temp, press, atmos_refract)
    if radians:
        return np.deg2rad(azimuth), np.deg2rad(zenith), np.deg2rad(RA), np.deg2rad(dec), np.deg2rad(H)
    else:
        return azimuth,zenith,RA,dec,H

_pos_v = np.vectorize(_pos)

@np.vectorize
def julian_day(dt):
    """Convert UTC datetimes or UTC timestamps to Julian days

    Parameters
    ----------
    dt : array_like
        UTC datetime objects or UTC timestamps (as per datetime.utcfromtimestamp)

    Returns
    -------
    jd : ndarray
        datetimes converted to fractional Julian days
    """
    t = _calendar_time(dt)
    return _julian_day(t)

@jit
def arcdist(p0,p1,radians=False):
    """Angular distance between azimuth,zenith pairs
    
    Parameters
    ----------
    p0 : array_like, shape (..., 2)
    p1 : array_like, shape (..., 2)
        p[...,0] = azimuth angles, p[...,1] = zenith angles
    radians : boolean (default False)
        If False, angles are in degrees, otherwise in radians

    Returns
    -------
    ad :  array_like, shape is broadcast(p0,p1).shape
        Arcdistances between corresponding pairs in p0,p1
        In degrees by default, in radians if radians=True
    """
    #formula comes from translating points into cartesian coordinates
    #taking the dot product to get the cosine between the two vectors
    #then arccos to return to angle, and simplify everything assuming real inputs
    p0,p1 = np.asarray(p0), np.asarray(p1)
    if not radians:
        p0,p1 = np.deg2rad(p0), np.deg2rad(p1)
    a0,z0 = p0[...,0], p0[...,1]
    a1,z1 = p1[...,0], p1[...,1]
    d = np.arccos(np.cos(z0)*np.cos(z1)+np.cos(a0-a1)*np.sin(z0)*np.sin(z1))
    if radians:
        return d
    else:
        return np.rad2deg(d)

def topocentric_sunpos(dt, latitude, longitude, elevation, delta_t=0, radians=False):
    """Compute the topocentric coordinates of the sun as viewed at the given time and location.

    Parameters
    ----------
    dt : array_like of datetime or float
        UTC datetime objects or UTC timestamps (as per datetime.utcfromtimestamp) representing the times of observations
    latitude, longitude : array_like of float
        decimal degrees, positive for north of the equator and east of Greenwich
    elevation : array_like of float
        meters, relative to the WGS-84 ellipsoid
    delta_t : array_like of float, optional
        seconds, default is 0, difference between the earth's rotation time (TT) and universal time (UT)
    radians : bool, optional
        return results in radians if True, degrees if False (default)

    Returns
    -------
    right_ascension : ndarray, topocentric
    declination : ndarray, topocentric
    hour_angle : ndarray, topocentric
    """
      
    jd = julian_day(dt)
    return _topo_pos_v(jd, latitude, longitude, elevation, delta_t, radians)

def sunpos(dt, latitude, longitude, elevation, temperature=None, pressure=None, atmos_refract=None, delta_t=0, radians=False):
    """Compute the observed and topocentric coordinates of the sun as viewed at the given time and location.

    Parameters
    ----------
    dt : array_like of datetime or float
        UTC datetime objects or UTC timestamps (as per datetime.utcfromtimestamp) representing the times of observations
    latitude, longitude : array_like of float
        decimal degrees, positive for north of the equator and east of Greenwich
    elevation : array_like of float
        meters, relative to the WGS-84 ellipsoid
    temperature : None or array_like of float, optional
        celcius, default is 14.6 (global average in 2013)
    pressure : None or array_like of float, optional
        millibar, default is 1013 (global average in ??)
    atmos_refract : None or array_like of float, optional
        Atmospheric refraction at sunrise and sunset, in degrees. Default is 0.5667
    delta_t : array_like of float, optional
        seconds, default is 0, difference between the earth's rotation time (TT) and universal time (UT)
    radians : bool, optional
        return results in radians if True, degrees if False (default)

    Returns
    -------
    azimuth_angle : ndarray, measured eastward from north
    zenith_angle : ndarray, measured down from vertical
    right_ascension : ndarray, topocentric
    declination : ndarray, topocentric
    hour_angle : ndarray, topocentric
    """

    if temperature is None:
        temperature = 14.6
    if pressure is None:
        pressure = 1013
    if atmos_refract is None:
        atmos_refract = 0.5667
    
    jd = julian_day(dt)
    return _pos_v(jd, latitude, longitude, elevation, temperature, pressure, atmos_refract, delta_t, radians)

def observed_sunpos(dt, latitude, longitude, elevation, temperature=None, pressure=None, atmos_refract=None, delta_t=0, radians=False):
    """Compute the observed coordinates of the sun as viewed at the given time and location.

    Parameters
    ----------
    dt : array_like of datetime or float
        UTC datetime objects or UTC timestamps (as per datetime.utcfromtimestamp) representing the times of observations
    latitude, longitude : array_like of float
        decimal degrees, positive for north of the equator and east of Greenwich
    elevation : array_like of float
        meters, relative to the WGS-84 ellipsoid
    temperature : None or array_like of float, optional
        celcius, default is 14.6 (global average in 2013)
    pressure : None or array_like of float, optional
        millibar, default is 1013 (global average in ??)
    atmos_refract : None or array_like of float, optional
        Atmospheric refraction at sunrise and sunset, in degrees. Default is 0.5667
    delta_t : array_like of float, optional
        seconds, default is 0, difference between the earth's rotation time (TT) and universal time (UT)
    radians : bool, optional
        return results in radians if True, degrees if False (default)

    Returns
    -------
    azimuth_angle : ndarray, measured eastward from north
    zenith_angle : ndarray, measured down from vertical
    """
    return sunpos(dt, latitude, longitude, elevation, temperature, pressure, atmos_refract, delta_t, radians)[:2]

def test():
    test_file = 'test_1.txt'
    # Parse and compare results from https://midcdmz.nrel.gov/solpos/spa.html
    param_names = ['syear','smonth','sday','eyear','emonth','eday','otype','step','stepunit','hr','min','sec','latitude','longitude','timezone','elev','press','temp','dut1','deltat','azmrot','slope','refract']
    param_dtype = np.dtype([(name, float) for name in param_names])
    params = np.loadtxt(test_file, param_dtype, delimiter=',', skiprows=2, max_rows=1)
    
    row_type = np.dtype([
            ('Date_M/D/YYYY', 'S10'),
            ('Time_H:MM:SS', 'S8'),
            ('Topo_zen', float),
            ('Topo_az', float),
            ('Julian_day', float),
            ('Julian_century', float),
            ('Julian_ephemeris_day', float),
            ('Julian_ephemeris_century', float),
            ('Julian_ephemeris_millennium', float),
            ('Earth_heliocentric_longitude', float),
            ('Earth_heliocentric_latitude', float),
            ('Earth_radius_vector', float),
            ('Geocentric_longitude', float),
            ('Geocentric_latitude', float),
            ('Mean_elongation', float),
            ('Mean_anomaly_sun', float),
            ('Mean_anomaly_moon', float),
            ('Argument_latitude_moon', float),
            ('Ascending_longitude_moon', float),
            ('Nutation_longitude', float),
            ('Nutation_obliquity', float),
            ('Ecliptic_mean_obliquity', float),
            ('Ecliptic_true_obliquity', float),
            ('Aberration_correction', float),
            ('Apparent_sun_longitude', float),
            ('Greenwich_mean_sidereal_time', float),
            ('Greenwich_sidereal_time', float),
            ('Geocentric_sun_right_ascension', float),
            ('Geocentric_sun_declination', float),
            ('Observer_hour_angle', float),
            ('Sun_equatorial_horizontal_parallax', float),
            ('Sun_right_ascension_parallax', float),
            ('Topo_sun_declination', float),
            ('Topo_sun_right_ascension', float),
            ('Topo_local_hour_angle', float),
            ('Topo_elevation_angle_uncorrected', float),
            ('Atmospheric_refraction_correction', float),
            ('Topo_elevation_angle_corrected', float),
            ('Equation_of_time', float),
            ('Sunrise_hour_angle', float),
            ('Sunset_hour_angle', float),
            ('Sun_transit_altitude', float)])
    
    true_data = np.loadtxt(test_file, row_type, delimiter=',', skiprows=4)
    
    def to_datetime(date_time_pair):
        s = str(b' '.join(date_time_pair),'UTF-8')
        return datetime.strptime(s, '%m/%d/%Y %H:%M:%S')

    def angle_diff(a1, a2, period=2*np.pi):
        """(a1 - a2 + d) % (2*d) - d; d = period/2"""
        d = period/2
        return ((a1 - a2 + d) % (period)) - d
    
    dts = [to_datetime(dt_pair) for dt_pair in true_data[['Date_M/D/YYYY','Time_H:MM:SS']]]
    lat,lon,elev,temp,press,deltat = params['latitude'],params['longitude'],params['elev'],params['temp'],params['press'],params['deltat']
    all_errs = []
    for dt,truth in zip(dts,true_data):
        t = _calendar_time(dt)
        jd = _julian_day(t) #Julian_day
        jde = _julian_ephemeris_day(jd, deltat) #Julian_ephemeris_day
        jce = _julian_century(jde) #Julian_ephemeris_century
        jme = _julian_millennium(jce) #Julian_ephemeris_millenium
        L,B,R = _heliocentric_position(jme) #Earth_heliocentric_longitude, Earth_heliocentric_latitude, Earth_radius_vector
        delta_psi, epsilon = _nutation_obliquity(jce) #Nutation_longitude, Ecliptic_true_obliquity
        theta,beta = _geocentric_position((L,B,R)) #Geocentric_longitude, Geocentric_latitude
        delta_tau = _abberation_correction(R) #Aberration_correction
        llambda, beta = _sun_longitude((L,B,R), delta_psi) #Apparent_sun_longitude, Geocentric_latitude (identical to previous)
        v = _greenwich_sidereal_time(jd, delta_psi, epsilon) #Greenwich_sidereal_time
        alpha, delta = _sun_ra_decl(llambda, epsilon, beta) #Geocentric_sun_right_ascension, Geocentric_sun_declination
        alpha_p, delta_p, H_p = _sun_topo_ra_decl_hour(lat,lon,elev,jd,deltat) #Topo_sun_right_ascension, Topo_sun_declination, Topo_local_hour_angle
        az, zen, delta_e = _sun_topo_azimuth_zenith(lat,delta_p,H_p,temp,press) #Topo_az, Topo_zen, Atmospheric_refraction_correction
        
        jd_err = jd - truth['Julian_day']
        jde_err = jde - truth['Julian_ephemeris_day']
        jce_err = jce - truth['Julian_ephemeris_century']
        jme_err = jme - truth['Julian_ephemeris_millennium']
        L_err = L - truth['Earth_heliocentric_longitude']
        B_err = B - truth['Earth_heliocentric_latitude']
        R_err = R - truth['Earth_radius_vector']
        delta_psi_err = delta_psi - truth['Nutation_longitude']
        epsilon_err = epsilon - truth['Ecliptic_true_obliquity']
        theta_err = theta - truth['Geocentric_longitude']
        beta_err = beta - truth['Geocentric_latitude']
        delta_tau_err = delta_tau - truth['Aberration_correction']
        lambda_err = llambda - truth['Apparent_sun_longitude']
        v_err = v - truth['Greenwich_sidereal_time']
        alpha_err = alpha - truth['Geocentric_sun_right_ascension']
        delta_err = delta - truth['Geocentric_sun_declination']
        alpha_prime_err = alpha_p - truth['Topo_sun_right_ascension']
        delta_prime_err = delta_p - truth['Topo_sun_declination']
        H_prime_err = angle_diff(H_p, truth['Topo_local_hour_angle'], 360)
        az_err = angle_diff(az, truth['Topo_az'], 360)
        delta_e_err = delta_e - truth['Atmospheric_refraction_correction']
        zen_err = zen - truth['Topo_zen']
        all_errs.append([jd_err,jde_err,jce_err,jme_err,L_err,B_err,R_err,delta_psi_err,
                    epsilon_err,theta_err,beta_err,delta_tau_err,lambda_err,
                    v_err,alpha_err,delta_err,alpha_prime_err,delta_prime_err,
                    H_prime_err,az_err,delta_e_err, zen_err])
    rms_err = np.sqrt(np.mean(np.array(all_errs)**2,0))
    err_names = ['Julian day', 'Julian ephemeris day', 'Julian ephemeris century', 'Julian ephemeris millennium', 'Earth heliocentric longitude', 'Earth heliocentric latitude', 'Earth radius vector', 'Nutation longitude', 'Ecliptic true obliquity', 'Geocentric longitude', 'Geocentric latitude', 'Aberration correction', 'Apparent sun longitude', 'Greenwich sidereal time', 'Geocentric sun right ascension', 'Geocentric sun declination', 'Topo sun right ascension', 'Topo sun declination', 'Topo local hour angle', 'Topo az', 'Atmospheric_refraction_correction','Topo zen']
    print('RMS Errors')
    for n, e in zip(err_names, rms_err):
        print('{}: {}'.format(n, e))

def main(args):
    az, zen, ra, dec, h = sunpos(args.t, args.lat, args.lon, args.elev, args.temp, args.p, args.dt, args.rad)
    if args.csv:
        #machine readable
        print('{t}, {dt}, {lat}, {lon}, {elev}, {temp}, {p}, {az}, {zen}, {ra}, {dec}, {h}'.format(t=args.t, dt=args.dt, lat=args.lat, lon=args.lon, elev=args.elev,temp=args.temp, p=args.p,az=az, zen=zen, ra=ra, dec=dec, h=h))
    else:
        dr='deg'
        if args.rad:
            dr='rad'
        print("Computing sun position at T = {t} + {dt} s".format(t=args.t, dt=args.dt))
        print("Lat, Lon, Elev = {lat} deg, {lon} deg, {elev} m".format(lat=args.lat, lon=args.lon, elev=args.elev))
        print("T, P = {temp} C, {press} mbar".format(temp=args.temp, press=args.p))
        print("Results:")
        print("Azimuth, zenith = {az} {dr}, {zen} {dr}".format(az=az,zen=zen,dr=dr))
        print("RA, dec, H = {ra} {dr}, {dec} {dr}, {h} {dr}".format(ra=ra, dec=dec, h=h, dr=dr))

if __name__ == '__main__':
    if args.test:
        test()
    else:
        main(args)
