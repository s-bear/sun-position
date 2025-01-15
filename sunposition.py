# The MIT License (MIT)
# 
# Copyright (c) 2025 Samuel Bear Powell
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

import os, sys, argparse, re
import numpy as np
import time,datetime

try:
    #scipy is required for numba's linear algebra routines to work
    import numba, scipy
except:
    numba = None

VERSION = '1.2.0'

_arg_parser = argparse.ArgumentParser(prog='sunposition',description='Compute sun position parameters given the time and location')
_arg_parser.add_argument('--test',help='Test against output from https://midcdmz.nrel.gov/solpos/spa.html')
_arg_parser.add_argument('--version',action='version',version=f'%(prog)s {VERSION}')
_arg_parser.add_argument('--citation',action='store_true',help='Print citation information')
_arg_parser.add_argument('-t','--time',type=str,default='now',help='"now" or date and time in ISO8601 format or a (UTC) POSIX timestamp')
_arg_parser.add_argument('-lat','--latitude',type=float,default=51.48,help='observer latitude, in decimal degrees, positive for north')
_arg_parser.add_argument('-lon','--longitude',type=float,default=0.0,help='observer longitude, in decimal degrees, positive for east')
_arg_parser.add_argument('-e','--elevation',type=float,default=0,help='observer elevation, in meters')
_arg_parser.add_argument('-T','--temperature',type=float,default=14.6,help='temperature, in degrees celcius')
_arg_parser.add_argument('-p','--pressure',type=float,default=1013.0,help='atmospheric pressure, in millibar')
_arg_parser.add_argument('-a','--atmos_refract',type=float,default=0.5667,help='atmospheric refraction at sunrise and sunset, in degrees')
_arg_parser.add_argument('-dt',type=float,default=0.0,help='difference between earth\'s rotation time (TT) and universal time (UT1)')
_arg_parser.add_argument('-r','--radians',action='store_true',help='Output in radians instead of degrees')
_arg_parser.add_argument('--csv',action='store_true',help='Comma separated values (time,dt,lat,lon,elev,temp,pressure,az,zen,RA,dec,H)')
_arg_parser.add_argument('--jit',action='store_true',help='Enable Numba acceleration (likely to cause slowdown for a single computation!)')

def empty_decorator(f = None, *args, **kw):
    if callable(f):
        return f
    return empty_decorator

if numba is not None:
    # register_jitable informs numba that a function may be compiled when
    # called from jit'ed code, but doesn't jit it by default
    register_jitable = numba.extending.register_jitable

    # overload informs numba of an *alternate implementation* of a function to
    # use within jit'ed code -- we use it to provide a jit-able version of polyval
    overload = numba.extending.overload
    
    #njit compiles code -- we use this for our top-level functions
    njit = numba.njit

    _ENABLE_JIT = not numba.config.DISABLE_JIT and not os.environ.get('NUMBA_DISABLE_JIT',False)
else:
    #if numba is not available, use empty_decorator instead
    njit = empty_decorator
    overload = lambda *a,**k: empty_decorator
    register_jitable = empty_decorator
    _ENABLE_JIT = False


def enable_jit(en = True):
    global _ENABLE_JIT
    if en and numba is None:
        print('WARNING: JIT unavailable (requires numba and scipy)',file=sys.stderr)
    #We set the _ENABLE_JIT flag regardless of whether numba is available, just to test that code path!
    _ENABLE_JIT = en

def disable_jit():
    enable_jit(False)

def main(args=None, **kwargs):
    """Run sunposition command-line tool.

    If run without arguments, uses sys.argv, otherwise arguments may be
    specified by a list of strings to be parsed, e.g.:
        main(['--time','now'])
    or as keyword arguments:
        main(time='now')
    or as an argparse.Namespace object (as produced by argparse.ArgumentParser)

    Parameters
    ----------
    args : list of str or argparse.Namespace, optional
        Command-line arguments. sys.argv is used if not provided.
    test : path to test file
        Run for the test case in the test file and compare results.
    version : bool
        If true, print the version information and quit
    citation : bool
        If true, print citation information and quit
    time : str
        "now" or date and time in ISO8601 format or a UTC POSIX timestamp
    latitude : float
        observer latitude in decimal degrees, positive for north
    longitude : float
        observer longitude in decimal degrees, positive for east
    elevation : float
        observer elevation in meters
    temperature : float
        temperature, in degrees celcius
    pressure : float
        atmospheric pressure, in millibar
    atmos_refract : float
        atmospheric refraction at sunrise and sunset, in degrees
    dt : float
        difference between Earth's rotation time (TT) and universal time (UT1)
    radians : bool
        If True, output in radians instead of degrees
    csv : bool
        If True, output as comma separated values (time, dt, lat, lon, elev, temp, pressure, az, zen, RA, dec, H)
    """
    if args is None and not kwargs:
        args = _arg_parser.parse_args()
    elif isinstance(args,(list,tuple)):
        args = _arg_parser.parse_args(args)
    
    for kw in kwargs:
        setattr(args,kw,kwargs[kw])
    
    if args.citation:
        print("Algorithm:")
        print("  Ibrahim Reda, Afshin Andreas, \"Solar position algorithm for solar radiation applications\",")
        print("  Solar Energy, Volume 76, Issue 5, 2004, Pages 577-589, ISSN 0038-092X,")
        print("  doi:10.1016/j.solener.2003.12.003")
        print("Implemented by Samuel Powell, 2016-2025, https://github.com/s-bear/sun-position")
        return 0

    enable_jit(args.jit)
    
    if args.test:
        return test(args)

    t = _string_to_posix_time(args.time)
    lat, lon, elev = args.latitude, args.longitude, args.elevation
    temp, p, ar, dt = args.temperature, args.pressure, args.atmos_refract, args.dt 
    rad = args.radians

    az, zen, ra, dec, h = sunpos(t, lat, lon, elev, temp, p, ar, dt, rad)
    if args.csv:
        #machine readable
        print(f'{t}, {dt}, {lat}, {lon}, {elev}, {temp}, {p}, {az:0.6f}, {zen:0.6f}, {ra:0.6f}, {dec:0.6f}, {h:0.6f}')
    else:
        dr = 'rad' if args.radians else 'deg'
        ts = _posix_time_to_string(t)
        print(f"Computing sun position at T = {ts} + {dt} s")
        print(f"Lat, Lon, Elev = {lat} deg, {lon} deg, {elev} m")
        print(f"T, P = {temp} C, {p} mbar")
        print("Results:")
        print(f"Azimuth, zenith = {az:0.6f} {dr}, {zen:0.6f} {dr}")
        print(f"RA, dec, H = {ra:0.6f} {dr}, {dec:0.6f} {dr}, {h:0.6f} {dr}")\

    return 0

@njit
def _arcdist(p0,p1):
    a0,z0 = p0[...,0], p0[...,1]
    a1,z1 = p1[...,0], p1[...,1]
    return np.arccos(np.cos(z0)*np.cos(z1)+np.cos(a0-a1)*np.sin(z0)*np.sin(z1))

@njit
def _arcdist_deg(p0,p1):
    return np.rad2deg(_arcdist(np.deg2rad(p0),np.deg2rad(p1)))

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
    p0,p1 = np.broadcast_arrays(p0, p1)
    if radians:
        return _arcdist(p0,p1)
    else:
        return _arcdist_deg(p0,p1)

def sunpos(dt, latitude, longitude, elevation, temperature=None, pressure=None, atmos_refract=None, delta_t=0, radians=False, jit=None):
    """Compute the observed and topocentric coordinates of the sun as viewed at the given time and location.

    Parameters
    ----------
    dt : array_like of datetime, datetime64, str, or float
        datetime.datetime, numpy.datetime64, ISO8601 strings, or POSIX timestamps (float or int)
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
    jit : bool, optional
        override module jit settings. True to enable Numba acceleration (default if Numba is available), False to disable.
    
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
    if jit is None:
        jit = _ENABLE_JIT

    t = to_timestamp(dt)

    if jit:
        args = np.broadcast_arrays(t, latitude, longitude, elevation, temperature, pressure, atmos_refract, delta_t)
        sp = _sunpos_vec_jit(*args)
        sp = tuple(a[()] for a in sp) #unwrap np.array() from scalars
    else:
        sp = _sunpos_vec(t, latitude, longitude, elevation, temperature, pressure, atmos_refract, delta_t)
    if radians:
        sp = tuple(np.deg2rad(a) for a in sp)

    return sp

def topocentric_sunpos(dt, latitude, longitude, elevation, delta_t=0, radians=False, jit=None):
    """Compute the topocentric coordinates of the sun as viewed at the given time and location.

    Parameters
    ----------
    dt : array_like of datetime, datetime64, str, or float
        datetime.datetime, numpy.datetime64, ISO8601 strings, or POSIX timestamps (float or int)
    latitude, longitude : array_like of float
        decimal degrees, positive for north of the equator and east of Greenwich
    elevation : array_like of float
        meters, relative to the WGS-84 ellipsoid
    delta_t : array_like of float, optional
        seconds, default is 0, difference between the earth's rotation time (TT) and universal time (UT)
    radians : bool, optional
        return results in radians if True, degrees if False (default)
    jit : bool, optional
        override module jit settings. True to enable Numba acceleration (default if Numba is available), False to disable.

    Returns
    -------
    right_ascension : ndarray, topocentric
    declination : ndarray, topocentric
    hour_angle : ndarray, topocentric
    """
    if jit is None:
        jit = _ENABLE_JIT
    t = to_timestamp(dt)
    if jit:
        args = np.broadcast_arrays(t, latitude, longitude, elevation, delta_t)
        sp = _topo_sunpos_vec_jit(*args)
        sp = tuple(a[()] for a in sp) #unwrap np.array() from scalars
    else:
        sp = _topo_sunpos_vec(t, latitude, longitude, elevation, delta_t)
    if radians:
        sp = tuple(np.deg2rad(a) for a in sp)
    return sp

def observed_sunpos(dt, latitude, longitude, elevation, temperature=None, pressure=None, atmos_refract=None, delta_t=0, radians=False, jit=None):
    """Compute the observed coordinates of the sun as viewed at the given time and location.

    Parameters
    ----------
    dt : array_like of datetime, datetime64, str, or float
        datetime.datetime, numpy.datetime64, ISO8601 strings, or POSIX timestamps (float or int)
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
    jit : bool, optional
        override module jit settings. True to enable Numba acceleration (default if Numba is available), False to disable.

    Returns
    -------
    azimuth_angle : ndarray, measured eastward from north
    zenith_angle : ndarray, measured down from vertical
    """
    return sunpos(dt, latitude, longitude, elevation, temperature, pressure, atmos_refract, delta_t, radians, jit)[:2]

def test(args):
    test_file = args.test
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
    
    def to_timestamp(date_time_pair):
        s = str(b' '.join(date_time_pair),'UTF-8')
        dt = datetime.datetime.strptime(s, '%m/%d/%Y %H:%M:%S')
        return dt.replace(tzinfo=datetime.timezone.utc).timestamp()

    def angle_diff(a1, a2, period=2*np.pi):
        """(a1 - a2 + d) % (2*d) - d; d = period/2"""
        d = period/2
        return ((a1 - a2 + d) % (period)) - d
    
    ts = [to_timestamp(dt_pair) for dt_pair in true_data[['Date_M/D/YYYY','Time_H:MM:SS']]]
    lat,lon,elev,temp,press,deltat = params['latitude'],params['longitude'],params['elev'],params['temp'],params['press'],params['deltat']
    all_errs = []
    for t,truth in zip(ts,true_data):
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

## Dates and times
# this application has unusual date/time requirements that are not supported by
# Python's time or datetime libraries. Specifically:
# 1. The subroutines use Julian days to represent time
# 2. The algorithm supports dates from year -2000 to 6000 (datetime supports years 1-9999)

# For that reason, we will use our own date & time codes
# The Julian Day conversion in Reda & Andreas's paper required the Gregorian
#  (year, month, day), with the time of day specified as a fractional day, all in UTC.
# We can use the algorithms in "Euclidean affine functions and their application to calendar algorithms" C. Neri, L. Schneider (2022) https://doi.org/10.1002/spe.3172
#  to convert rata die timestamps (day number since epoch) to Gregorian (year, month, day)
# We need to be able to convert the following to Julian days:
#  datetime.datetime
#  numpy.datetime64
#  int/float POSIX timestamp
#  ISO8601 string

@register_jitable
def _date_to_rd(year, month, day):
    '''Convert year, month, day to rata die (day number)
    Based on the algorithm in: "Euclidean affine functions and their application to calendar algorithms" C. Neri, L. Schneider (2022) https://doi.org/10.1002/spe.3172
    day may be fractional, fractional part is added to return value
    '''
    # we include explicit types everywhere to match the reference implementation, which is all 32-bit
    # Python will still use 64-bit math for some of the intermediate calculations, but it shouldn't change the results
    Y_G = np.int32(year)
    M_G = np.uint32(month)
    D_G = np.uint32(day)
    tod = day - D_G #split fractional part

    s = np.uint32(82) #static uint32_t constexpr s = 82;
    K = np.uint32(719468 + 146097 * s) #static uint32_t constexpr K = 719468 + 146097 * s;
    L = np.uint32(400*s) #static uint32_t constexpr L = 400 * s;
    #  int32_t to_rata_die(int32_t Y_G, uint32_t M_G, uint32_t D_G) {
    # Map. (Notice the year correction, including type change.)
    J = (M_G <= 2) # uint32_t const J = M_G <= 2;
    Y = np.uint32(Y_G + L - J) # uint32_t const Y = (uint32_t(Y_G) + L) - J;
    # uint32_t const M = J ? M_G + 12 : M_G;
    if J: 
        M = np.uint32(M_G + 12)
    else:
        M = M_G
    
    D = np.uint32(D_G - 1) # uint32_t const D = D_G - 1;
    C = np.uint32(Y // 100) # uint32_t const C = Y / 100;

    # Rata die.
    y_star = np.uint32(1461 * Y // 4 - C + C // 4) # uint32_t const y_star = 1461 * Y / 4 - C + C / 4;
    m_star = np.uint32((979 * M - 2919) // 32) # uint32_t const m_star = (979 * M - 2919) / 32;
    N      = np.uint32(y_star + m_star + D) # uint32_t const N      = y_star + m_star + D;
    
    # Rata die shift.
    N_U = np.int32(N) - np.int32(K) # uint32_t const N_U = N - K; -- the original casts to int32 at return, we do it here
    return N_U + tod

@register_jitable
def _date_to_posix_time(year, month, day):
    return _date_to_rd(year, month, day)*86400 # rata die (fractional day number) x 86400 seconds per day

@register_jitable
def _rd_to_date(rd):
    '''convert integer rata die (day number) to year, month, day
    "Euclidean affine functions and their application to calendar algorithms" C. Neri, L. Schneider (2022) https://doi.org/10.1002/spe.3172
    '''
    #  date32_t to_date(int32_t N_U) {
    N_U = np.int32(rd)
    tod = rd - N_U #split fractional part

    s = np.uint32(82) #static uint32_t constexpr s = 82;
    K = np.uint32(719468 + 146097 * s) #static uint32_t constexpr K = 719468 + 146097 * s;
    L = np.uint32(400*s) #static uint32_t constexpr L = 400 * s;
    
    # Rata die shift.
    N = np.uint32(N_U + K) # uint32_t const N = N_U + K;

    # Century.
    N_1 = np.uint32(4 * N + 3) # uint32_t const N_1 = 4 * N + 3;
    # uint32_t const C   = N_1 / 146097;
    # uint32_t const N_C = N_1 % 146097 / 4;
    C, N_C = divmod(N_1, 146097)
    C, N_C = np.uint32(C), np.uint32(N_C // 4)

    # Year.
    N_2 = np.uint32(4 * N_C + 3) # uint32_t const N_2 = 4 * N_C + 3;
    P_2 = 2939745 * np.uint64(N_2) # uint64_t const P_2 = uint64_t(2939745) * N_2;
    # uint32_t const Z   = uint32_t(P_2 / 4294967296);
    # uint32_t const N_Y = uint32_t(P_2 % 4294967296) / 2939745 / 4;
    Z, N_Y = divmod(P_2, 4294967296)
    Z, N_Y = np.uint32(Z), np.uint32(N_Y)
    N_Y = np.uint32((N_Y // 2939745) // 4)
    Y   = np.uint32(100 * C + Z) # uint32_t const Y   = 100 * C + Z;

    # Month and day. 
    N_3 = np.uint32(2141 * N_Y + 197913) # uint32_t const N_3 = 2141 * N_Y + 197913;
    # uint32_t const M   = N_3 / 65536;
    # uint32_t const D   = N_3 % 65536 / 2141;
    M, D = divmod(N_3, 65536)
    M, D = np.uint32(M), np.uint32(D // 2141)

    # Map. (Notice the year correction, including type change.)
    J   = (N_Y >= 306) # uint32_t const J   = N_Y >= 306;
    Y_G = np.int32(Y - L + J) # int32_t  const Y_G = (Y - L) + J;
    # uint32_t const M_G = J ? M - 12 : M;
    if J:
        M_G = np.uint32(M - 12)
    else:
        M_G = M

    D_G = np.uint32(D + 1) # uint32_t const D_G = D + 1;
    # return { Y_G, M_G, D_G };
    return (Y_G, M_G, D_G + tod)

@register_jitable
def _posix_time_to_date(t):
    '''POSIX timestamp to (year, month, day)'''
    return _rd_to_date(t/86400) # (timestamp in seconds)/86400 -> rata die (fractional day number)

_iso8601_re = re.compile(r'([+-]?\d{1,4})-?([01]\d)-?([0-3]\d)[T ]([012]\d):?([0-6]\d):?([0-6]\d(?:\.\d+)?)?(?:Z|(?:([+-]\d{2})(?::?(\d{2}))?))?')
def _string_to_posix_time(s):
    '''parse timestamp string to posix time, assumes UTC if timezone is not specified
    strings may be:
     - "now" -- which gets the current time
     - POSIX timestamp string
     - ISO 8601 formatted string (including negative years)
    '''
    if s == 'now':
        return time.time()
    try:
        return float(s)
    except ValueError: #
        pass
    m = _iso8601_re.match(s)
    if not m:
        raise ValueError('Could not parse timestamp string (must be "now" or float or ISO8601)')
    year,month,day,hour,minute,second,tz_hour,tz_minute = m.groups()
    if second is None: second = 0
    if tz_hour is None: tz_hour = 0
    if tz_minute is None: tz_minute = 0
    year, month, day = int(year),int(month),int(day)
    hour,minute,second = int(hour), int(minute), float(second)
    tod = (hour + (minute + second/60)/60)/24 # time of day, in days
    tz_hour, tz_minute = int(tz_hour), int(tz_minute)
    if tz_hour < 0:
        tz = tz_hour - tz_minute/60 # timezone offset, in hours
    else:
        tz = tz_hour + tz_minute/60 # timezone offset, in hours
    rd = _date_to_rd(year, month, day) #to rata die
    #validate the date using _rd_to_date(_date_to_rd()) round trip
    if _rd_to_date(rd) != (year, month, day):
        raise ValueError('Invalid date')
    if tod > 1:
        raise ValueError('Invalid time')
    # UTC offsets vary from -12:00 (US Minor Outlying Islands) to +14:00 (Kiribati)
    if tz < -12 or tz > 14:
        raise ValueError('Invalid timezone')
    #apply tod and tz to rd & multiply by seconds per day
    return 86400*(rd + tod - tz/24)


def _posix_time_to_string(t):
    '''Format a POSIX timestamp as ISO8601 with millisecond precision'''
    #we need our own because datetime.datetime.fromtimestamp() doesn't support dates before epoch
    year, month, fday = _posix_time_to_date(t)
    day = int(fday)
    ms = round((fday - day)*86400000) #milliseconds into the day
    hour, ms = divmod(ms, 3600000) #hour, ms into the hour
    minute, ms = divmod(ms, 60000) #minute, ms into the minute
    sec, ms = divmod(ms, 1000) #second, millisecond
    return f'{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{sec:02}.{ms:03}Z'
    
#we can't use numba to accelerate _string_to_posix_time, so use np.vectorize here
_string_to_posix_time_v = np.vectorize(_string_to_posix_time)

@register_jitable
def _julian_day(t):
    """Calculate the Julian Day from posix timestamp (seconds since epoch)"""
    year, month, day = _posix_time_to_date(t)
    # year and month numbers
    if month <= 2:  # From paper: "if M = 1 or 2, then Y = Y - 1 and M = M + 12"
        month += 12
        year -= 1
    # b is equal to 0 for the julian calendar and is equal to (2- A +
    # INT(A/4)), A = INT(Y/100), for the gregorian calendar
    a = int(year / 100)
    b = 2 - a + int(a / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    return jd

_julian_day_vec = np.vectorize(_julian_day)

@njit
def _julian_day_vec_jit(t):
    ts = np.asarray(t)
    ts_flat = ts.flat
    n = len(ts_flat)
    jds = np.empty(n, dtype=np.float64)
    for i, t in enumerate(ts_flat):
        jds[i] = _julian_day(t)
    jds = jds.reshape(ts.shape)
    return jds[()]

#we can't use numba to accelerate _get_timestamp, so use np.vectorize here
@np.vectorize
def _get_timestamp(dt):
    '''get the timestamp from a datetime.datetime object'''
    return dt.timestamp()

def to_timestamp(dt):
    '''Convert various date/time formats to POSIX timestamps
    
    Parameters
    ----------
    dt : array_like of datetime.datetime, numpy.datetime64, ISO8601 strings
        date/times to convert to POSIX timestamps. 
    
    Returns
    -------
    t : ndarray
        dt converted to POSIX timestamps, or if dt is not one of the listed formats
        it is returned unchanged.
    '''
    dt = np.asarray(dt)

    if np.issubdtype(dt.dtype, str):
        t = _string_to_posix_time_v(dt)
    elif np.issubdtype(dt.dtype, datetime.datetime):
        t = _get_timestamp(dt)
    elif np.issubdtype(dt.dtype, np.datetime64):
        t = dt.astype('datetime64[us]').astype(np.int64)/1e6
    else:
        t = dt
    return t[()] #unwrap scalar values out of np.array

def julian_day(dt, jit=None):
    """Convert timestamps from various formats to Julian days

    Parameters
    ----------
    dt : array_like
        datetime.datetime, numpy.datetime64, ISO8601 strings, or POSIX timestamps (float or int)
    jit : bool or None
        override module jit settings, to True/False to enable/disable numba acceleration
    
    Returns
    -------
    jd : ndarray
        datetimes converted to fractional Julian days
    """

    t = to_timestamp(dt)
    if jit is None: jit = _ENABLE_JIT
    if jit:
        jd = _julian_day_vec_jit(t)
    else:
        jd = _julian_day_vec(t)
    return jd[()] # use [()] to "unwrap" scalar values out of np.array

@register_jitable
def _julian_ephemeris_day(jd, deltat):
    """Calculate the Julian Ephemeris Day from the Julian Day and delta-time = (terrestrial time - universal time) in seconds"""
    return jd + deltat / 86400.0

@register_jitable
def _julian_century(jd):
    """Caluclate the Julian Century from Julian Day or Julian Ephemeris Day"""
    return (jd - 2451545.0) / 36525.0

@register_jitable
def _julian_millennium(jc):
    """Calculate the Julian Millennium from Julian Ephemeris Century"""
    return jc / 10.0

@register_jitable
def _cos_sum(x, coeffs):
    y = np.zeros(len(coeffs))
    for i, abc in enumerate(coeffs):
        for a,b,c in abc:
            y[i] += a*np.cos(b + c*x)
    return y

#implement np.polyval for numba using overload
@overload(np.polyval)
def _polyval_jit(p, x):
    def _polyval_impl(p, x):
        y = 0.0
        for v in p:
            y = y*x + v
        return y
    return _polyval_impl

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
    (10.0, 1.3, 6286.6), (10.0, 4.24, 1349.87), (9.0, 2.7, 242.73),
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
    (205.0, 1.869, 5573.143), (202.0, 2.458, 6069.777), (156.0, 0.833, 213.299),
    (132.0, 3.411, 2942.463), (126.0, 1.083, 20.775), (115.0, 0.645, 0.98),
    (103.0, 0.636, 4694.003), (102.0, 0.976, 15720.839), (102.0, 4.267, 7.114),
    (99.0, 6.21, 2146.17), (98.0, 0.68, 155.42), (86.0, 5.98, 161000.69),
    (85.0, 1.3, 6275.96), (85.0, 3.67, 71430.7), (80.0, 1.81, 17260.15),
    (79.0, 3.04, 12036.46), (75.0, 1.76, 5088.63), (74.0, 3.5, 3154.69),
    (74.0, 4.68, 801.82), (70.0, 0.83, 9437.76), (62.0, 3.98, 8827.39),
    (61.0, 1.82, 7084.9), (57.0, 2.78, 6286.6), (56.0, 4.39, 14143.5),
    (56.0, 3.47, 6279.55), (52.0, 0.19, 12139.55), (52.0, 1.33, 1748.02),
    (51.0, 0.28, 5856.48), (49.0, 0.49, 1194.45), (41.0, 5.37, 8429.24),
    (41.0, 2.4, 19651.05), (39.0, 6.17, 10447.39), (37.0, 6.04, 10213.29),
    (37.0, 2.57, 1059.38), (36.0, 1.71, 2352.87), (36.0, 1.78, 6812.77),
    (33.0, 0.59, 17789.85), (30.0, 0.44, 83996.85), (30.0, 2.74, 1349.87),
    (25.0, 3.16, 4690.48)])
)

@register_jitable
def _heliocentric_longitude(jme):
    """Compute the Earth Heliocentric Longitude (L) in degrees given the Julian Ephemeris Millennium"""
    #L5, ..., L0
    Li = _cos_sum(jme, _EHL)
    L = np.polyval(Li, jme) / 1e8
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

@register_jitable
def _heliocentric_latitude(jme):
    """Compute the Earth Heliocentric Latitude (B) in degrees given the Julian Ephemeris Millennium"""
    Bi = _cos_sum(jme, _EHB)
    B = np.polyval(Bi, jme) / 1e8
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
    (9.0, 3.63, 77713.77), (6.0, 1.87, 5573.14), (3.0, 5.47, 18849.23)]),
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
    (86.0, 5.69, 15720.84), (86.0, 1.27, 161000.69), (65.0, 0.27, 17260.15),
    (63.0, 0.92, 529.69), (57.0, 2.01, 83996.85), (56.0, 5.24, 71430.7),
    (49.0, 3.25, 2544.31), (47.0, 2.58, 775.52), (45.0, 5.54, 9437.76),
    (43.0, 6.01, 6275.96), (39.0, 5.36, 4694), (38.0, 2.39, 8827.39),
    (37.0, 0.83, 19651.05), (37.0, 4.9, 12139.55), (36.0, 1.67, 12036.46),
    (35.0, 1.84, 2942.46), (33.0, 0.24, 7084.9), (32.0, 0.18, 5088.63),
    (32.0, 1.78, 398.15), (28.0, 1.21, 6286.6), (28.0, 1.9, 6279.55),
    (26.0, 4.59, 10447.39)])
)

@register_jitable
def _heliocentric_radius(jme):
    """Compute the Earth Heliocentric Radius (R) in astronimical units given the Julian Ephemeris Millennium"""
    
    Ri = _cos_sum(jme, _EHR)
    R = np.polyval(Ri, jme) / 1e8
    return R

@register_jitable
def _heliocentric_position(jme):
    """Compute the Earth Heliocentric Longitude, Latitude, and Radius given the Julian Ephemeris Millennium
        Returns (L, B, R) where L = longitude in degrees, B = latitude in degrees, and R = radius in astronimical units
    """
    return _heliocentric_longitude(jme), _heliocentric_latitude(jme), _heliocentric_radius(jme)

@register_jitable
def _geocentric_position(helio_pos):
    """Compute the geocentric latitude (Theta) and longitude (beta) (in degrees) of the sun given the earth's heliocentric position (L, B, R)"""
    L,B,R = helio_pos
    th = L + 180
    b = -B
    return (th, b)

@register_jitable
def _ecliptic_obliquity(jme, delta_epsilon):
    """Calculate the true obliquity of the ecliptic (epsilon, in degrees) given the Julian Ephemeris Millennium and the obliquity"""
    u = jme/10
    eq24_coeffs = np.array([2.45, 5.79, 27.87, 7.12, -39.05, -249.67, -51.38, 1999.25, -1.55, -4680.93, 84381.448])
    e0 = np.polyval(eq24_coeffs, u)
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

@register_jitable
def _nutation_obliquity(jce):
    """compute the nutation in longitude (delta_psi) and the true obliquity (epsilon) given the Julian Ephemeris Century"""
    #mean elongation of the moon from the sun, in radians:
    #x0 = 297.85036 + 445267.111480*jce - 0.0019142*(jce**2) + (jce**3)/189474
    eq15_coeffs = np.array([1./189474, -0.0019142, 445267.111480, 297.85036])
    x0 = np.deg2rad(np.polyval(eq15_coeffs,jce))
    #mean anomaly of the sun (Earth), in radians:
    eq16_coeffs = np.array([-1/3e5, -0.0001603, 35999.050340, 357.52772])
    x1 = np.deg2rad(np.polyval(eq16_coeffs, jce))
    #mean anomaly of the moon, in radians:
    eq17_coeffs = np.array([1./56250, 0.0086972, 477198.867398, 134.96298])
    x2 = np.deg2rad(np.polyval(eq17_coeffs, jce))
    #moon's argument of latitude, in radians:
    eq18_coeffs = np.array([1./327270, -0.0036825, 483202.017538, 93.27191])
    x3 = np.deg2rad(np.polyval(eq18_coeffs, jce))
    #Longitude of the ascending node of the moon's mean orbit on the ecliptic
    # measured from the mean equinox of the date, in radians
    eq19_coeffs = np.array([1./45e4, 0.0020708, -1934.136261, 125.04452])
    x4 = np.deg2rad(np.polyval(eq19_coeffs, jce))

    x = np.array([x0, x1, x2, x3, x4])

    a,b = _NLO_AB.T
    c,d = _NLO_CD.T
    dp = np.sum((a + b*jce)*np.sin(np.dot(_NLO_Y, x)))/36e6
    de = np.sum((c + d*jce)*np.cos(np.dot(_NLO_Y, x)))/36e6
    
    e = _ecliptic_obliquity(_julian_millennium(jce), de)

    return dp, e

@register_jitable
def _abberation_correction(R):
    """Calculate the abberation correction (delta_tau, in degrees) given the Earth Heliocentric Radius (in AU)"""
    return -20.4898/(3600*R)

@register_jitable
def _sun_longitude(helio_pos, delta_psi):
    """Calculate the apparent sun longitude (lambda, in degrees) and geocentric latitude (beta, in degrees) given the earth heliocentric position and delta_psi"""
    L,B,R = helio_pos
    theta = L + 180 #geocentric longitude
    beta = -B #geocentric latitude
    ll = theta + delta_psi + _abberation_correction(R)
    return ll, beta

@register_jitable
def _greenwich_sidereal_time(jd, delta_psi, epsilon):
    """Calculate the apparent Greenwich sidereal time (v, in degrees) given the Julian Day"""
    jc = _julian_century(jd)
    #mean sidereal time at greenwich, in degrees:
    v0 = (280.46061837 + 360.98564736629*(jd - 2451545) + 0.000387933*(jc**2) - (jc**3)/38710000) % 360
    v = v0 + delta_psi*np.cos(np.deg2rad(epsilon))
    return v

@register_jitable
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

@register_jitable
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

@register_jitable
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

@register_jitable
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

@register_jitable
def _topo_sunpos(t, lat, lon, elev, dt):
    """compute RA,dec,H, all in degrees"""
    jd = _julian_day(t)
    lat,lon = _norm_lat_lon(lat,lon)
    RA, dec, H = _sun_topo_ra_decl_hour(lat, lon, elev, jd, dt)
    return RA, dec, H

_topo_sunpos_vec = np.vectorize(_topo_sunpos)

@njit
def _topo_sunpos_vec_jit(t, lat, lon, elev, dt):
    '''Compute RA, dec, H, all in degrees; vectorized for use with Numba
    Arguments must be broadcast before calling: Numba's broadcast does not match Numpy's with scalar arguments
    Return values must be unwrapped after calling because Numba can't handle dynamic return types
    '''
    #broadcast
    out_shape = t.shape
    #flatten
    args_flat = t.flat, lat.flat, lon.flat, elev.flat, dt.flat
    n = len(args_flat[0])
    #allocate outputs as flat arrays
    RA, dec, H = np.empty(n), np.empty(n), np.empty(n)
    #do calculations over flattened inputs
    for i, arg in enumerate(zip(*args_flat)):
        RA[i], dec[i], H[i] = _topo_sunpos(*arg)
    #reshape to final dims
    RA, dec, H = RA.reshape(out_shape), dec.reshape(out_shape), H.reshape(out_shape)
    return RA, dec, H


@register_jitable
def _sunpos(t, lat, lon, elev, temp, press, atmos_refract, dt):
    """Compute azimuth,zenith,RA,dec,H"""
    jd = _julian_day(t)
    lat,lon = _norm_lat_lon(lat,lon)
    RA, dec, H = _sun_topo_ra_decl_hour(lat, lon, elev, jd, dt)
    azimuth, zenith, delta_e = _sun_topo_azimuth_zenith(lat, dec, H, temp, press, atmos_refract)
    return azimuth, zenith, RA, dec, H

_sunpos_vec = np.vectorize(_sunpos)

@njit
def _sunpos_vec_jit(t, lat, lon, elev, temp, press, atmos_refract, dt):
    '''Compute azimuth,zenith,RA,ec,H; vectorized for use with Numba
    Note that arguments must be broadcast in advance as numba's broadcast does not match numpy's with scalar arguments
    Return values must be unwrapped after calling because Numba can't handle dynamic return types
    '''
    out_shape = t.shape #final output shape
    
    #flatten inputs
    args_flat = t.flat, lat.flat, lon.flat, elev.flat, temp.flat, press.flat, atmos_refract.flat, dt.flat
    n = len(args_flat[0])
    #allocate outputs as flat arrays
    azimuth, zenith = np.empty(n), np.empty(n)
    RA, dec, H = np.empty(n), np.empty(n), np.empty(n)
    #do calculations over flattened inputs
    for i, arg in enumerate(zip(*args_flat)):
        t, lat, lon, elev, temp, press, atmos_refract, dt = arg
        azimuth[i], zenith[i], RA[i], dec[i], H[i] = _sunpos(t, lat, lon, elev, temp, press, atmos_refract, dt)
    #reshape outputs to final dimensions
    azimuth, zenith = azimuth.reshape(out_shape), zenith.reshape(out_shape)
    RA, dec, H = RA.reshape(out_shape), dec.reshape(out_shape), H.reshape(out_shape)
    return azimuth, zenith, RA, dec, H

if __name__ == '__main__':
    sys.exit(main())
