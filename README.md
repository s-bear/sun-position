# sunposition

## Description

`sunposition` is a python module for computing the sun's position based on the algorithms from "Solar position algorithm for solar radiation applications" by Ibrahim Reda and Afshin Anreas, Solar Energy (2004).
The algorithm calculates "the solar zenith and azimuth angles in the period from the year −2000 to 6000, with uncertainties of ±0.0003°".
See http://dx.doi.org/10.1016/j.solener.2003.12.003 for more information.

In this code, the latitude and longitude are positive for North and East, respectively.
The azimuth angle is 0 at North and positive towards the east.
The zenith angle is 0 at vertical and positive towards the horizon.

The code is hosted at https://github.com/s-bear/sun-position

The module is a single python file `sunposition.py` and may be used as a command-line utility or imported into a script.
The module depends only on [NumPy](https://numpy.org) but can optionally use [Numba](https://numba.pydata.org/) for performance improvements.

## Installation

`sunposition` is hosted at [https://pypi.org/project/sunposition/](https://pypi.org/project/sunposition/) and may be installed using `pip`:

```
$ pip install sunposition
```

## Example usage on the command line

```
$ sunposition --help
usage: sunposition [-h] [--test TEST] [--version] [--citation] [-t TIME] [-lat LATITUDE] [-lon LONGITUDE] [-e ELEVATION] [-T TEMPERATURE] [-p PRESSURE] [-a ATMOS_REFRACT] [-dt DT] [-r] [--csv] [--jit]

Compute sun position parameters given the time and location

options:
  -h, --help            show this help message and exit
  --test TEST           Test against output from https://midcdmz.nrel.gov/solpos/spa.html
  --version             show program's version number and exit
  --citation            Print citation information
  -t TIME, --time TIME  "now" or date and time in ISO8601 format or a (UTC) POSIX timestamp
  -lat LATITUDE, --latitude LATITUDE
                        observer latitude, in decimal degrees, positive for north
  -lon LONGITUDE, --longitude LONGITUDE
                        observer longitude, in decimal degrees, positive for east
  -e ELEVATION, --elevation ELEVATION
                        observer elevation, in meters
  -T TEMPERATURE, --temperature TEMPERATURE
                        temperature, in degrees celcius
  -p PRESSURE, --pressure PRESSURE
                        atmospheric pressure, in millibar
  -a ATMOS_REFRACT, --atmos_refract ATMOS_REFRACT
                        atmospheric refraction at sunrise and sunset, in degrees
  -dt DT                difference between earth's rotation time (TT) and universal time (UT1)
  -r, --radians         Output in radians instead of degrees
  --csv                 Comma separated values (time,dt,lat,lon,elev,temp,pressure,az,zen,RA,dec,H)
  --jit                 Enable Numba acceleration (likely to cause slowdown for a single computation!)

$ sunposition
Computing sun position at T = 2025-01-15T06:26:55.969Z + 0.0 s
Lat, Lon, Elev = 51.48 deg, 0.0 deg, 0 m
T, P = 14.6 C, 1013.0 mbar
Results:
Azimuth, zenith = 106.729483 deg, 103.700963 deg
RA, dec, H = 297.315158 deg, -21.068459 deg, -85.618177 deg

$ sunposition -t "1953-05-29 05:45:00" -lat 27.9881 -lon 86.9253 -e 8848
Computing sun position at T = 1953-05-29T05:45:00.000Z + 0.0 s
Lat, Lon, Elev = 27.9881 deg, 86.9253 deg, 8848.0 m
T, P = 14.6 C, 1013.0 mbar
Results:
Azimuth, zenith = 137.736768 deg, 8.481270 deg
RA, dec, H = 65.760501 deg, 21.576417 deg, 353.875172 deg
```

An example test file is provided at https://raw.githubusercontent.com/s-bear/sun-position/master/sunposition_test.txt

## Example usage in a script

```python
import numpy as np
import matplotlib.pyplot as plt
# When imported as a module, sunposition will use numba.jit if available
# This may negatively impact performance if few positions are being computed
# For a rough guideline, on the author's machine:
#    jit:    5.5 seconds + 35 microseconds per computation
#    no-jit  1.4 milliseconds per computation
#    break-even: ~4000 computations
# There are several methods to disable jit:
#    1. If numba.config.DISABLE_JIT or the environment variable NUMBA_DISABLE_JIT
#       are set *before* sunposition is imported, jit will be disabled by default.
#    2. After sunposition is imported, use
#          sunposition.disable_jit()
#       or
#          sunposition.enable_jit(False)
#    3. Pass `jit=False` as a keyword argument to the function
import sunposition

#evaluate on a 2 degree grid
lon  = np.linspace(-180,180,181)
lat = np.linspace(-90,90,91)
LON, LAT = np.meshgrid(lon,lat)
# to_timestamp(s) converts a string to a POSIX-style timestamp (seconds since epoch)
# s may be 'now', which returns the current time using time.time()
#   or an ISO-8601 formatted date & time, e.g. '2024-04-08T11:09:34-07:00' 
now = sunposition.to_timestamp('now')
az,zen = sunposition.sunpos(now,LAT,LON,0)[:2] #discard RA, dec, H
#convert zenith to elevation
elev = 90 - zen
#convert azimuth to vectors
u, v = np.cos((90-az)*np.pi/180), np.sin((90-az)*np.pi/180)
#plot
fig, ax = plt.subplots(figsize=(6,3),layout='constrained')
img = ax.imshow(elev,cmap=plt.cm.CMRmap,origin='lower',vmin=-90,vmax=90,extent=(-181,181,-91,91))
s = slice(5,-1,5) # equivalent to 5:-1:5
ax.quiver(lon[s],lat[s],u[s,s],v[s,s],pivot='mid',scale_units='xy')
ax.contour(lon,lat,elev,[0])
ax.set_aspect('equal')
ax.set_xticks(np.arange(-180,181,45))
ax.set_yticks(np.arange(-90,91,45))
ax.set_xlabel('Longitude (deg)')
ax.set_ylabel('Latitude (deg)')
cb = plt.colorbar(img,ax=ax,shrink=0.8,pad=0.03)
cb.set_label('Sun Elevation (deg)')
#display plot
plt.show() #unnecessary in interactive sessions

```

## Citation
Ibrahim Reda, Afshin Andreas, Solar position algorithm for solar radiation applications, Solar Energy, Volume 76, Issue 5, 2004, Pages 577-589, ISSN 0038-092X, http://dx.doi.org/10.1016/j.solener.2003.12.003.
Keywords: Global solar irradiance; Solar zenith angle; Solar azimuth angle; VSOP87 theory; Universal time; ΔUT1

# LICENSE

Copyright (c) 2025 Samuel Bear Powell, samuel.powell@uq.edu.au

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
