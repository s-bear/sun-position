# sunposition

Python module for computing sun position based on the algorithms from "Solar position algorithm for solar radiation applications" by Ibrahim Reda and Afshin Anreas, Solar Energy (2004).
The algorithm calculates "the solar zenith and azimuth angles in the period from the year −2000 to 6000, with uncertainties of ±0.0003°".
See http://dx.doi.org/10.1016/j.solener.2003.12.003 for more information.

In this code, the latitude and longitude are positive for North and East, respectively. The azimuth angle is 0 at North and positive towards the east. The zenith angle is 0 at vertical and positive towards the horizon.

Example usage on the command line:
```
$ sunposition --help
usage: sunposition [-h] [--test TEST] [--version] [--citation] [-t TIME] [-lat LATITUDE] [-lon LONGITUDE]
                   [-e ELEVATION] [-T TEMPERATURE] [-p PRESSURE] [-a ATMOS_REFRACT] [-dt DT] [-r] [--csv] [--jit]

Compute sun position parameters given the time and location

options:
  -h, --help            show this help message and exit
  --test TEST           Test against output from https://midcdmz.nrel.gov/solpos/spa.html
  --version             show program's version number and exit
  --citation            Print citation information
  -t TIME, --time TIME  "now" or date and time (UTC) in "YYYY-MM-DD hh:mm:ss.ssssss" format or a (UTC) POSIX timestamp
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
  --jit                 Enable Numba acceleration (jit compilation time may overwhelm speed-up)

$ sunposition
Computing sun position at T = 2021-05-21 06:47:44.644873 + 0.0 s
Lat, Lon, Elev = 51.48 deg, 0.0 deg, 0 m
T, P = 14.6 C, 1013.0 mbar
Results:
Azimuth, zenith = 86.68229367131721 deg, 66.38510410296101 deg
RA, dec, H = 58.28648711185745 deg, 20.241411055526044 deg, 282.7836435018984 deg

$ sunposition -t "1953-05-29 05:45:00" -lat 27.9881 -lon 86.9253 -e 8848
Computing sun position at T = 1953-05-29 05:45:00 + 0.0 s
Lat, Lon, Elev = 27.9881 deg, 86.9253 deg, 8848.0 m
T, P = 14.6 C, 1013.0 mbar
Results:
Azimuth, zenith = 137.73675146015 deg, 8.481271417778686 deg
RA, dec, H = 65.7605040841157 deg, 21.576417030912577 deg, 353.8751689030205 deg
```

An example test file is provided at https://raw.githubusercontent.com/s-bear/sun-position/master/sunposition_test.txt

Example usage in code:
```python
from pylab import *
#sunposition will use numba.jit if available, which may negatively
#impact performance if few positions are being computed.
#To disable jit, before importing sunposition, either set 
#the environment variable NUMBA_DISABLE_JIT to 1 or
#set numba.config.DISABLE_JIT = False
# e.g. import os; os.environ['NUMBA_DISABLE_JIT'] = 1
# or import numba; numba.config.DISABLE_JIT = True
from sunposition import sunpos
from datetime import datetime

#evaluate on a 2 degree grid
lon  = linspace(-180,180,181)
lat = linspace(-90,90,91)
LON, LAT = meshgrid(lon,lat)
#at the current time
now = datetime.utcnow()
az,zen = sunpos(now,LAT,LON,0)[:2] #discard RA, dec, H
#convert zenith to elevation
elev = 90 - zen
#convert azimuth to vectors
u, v = cos((90-az)*pi/180), sin((90-az)*pi/180)
#plot
figure()
imshow(elev,cmap=cm.CMRmap,origin='lower',vmin=-90,vmax=90,extent=(-180,180,-90,90))
s = slice(5,-1,5) # equivalent to 5:-1:5
quiver(lon[s],lat[s],u[s,s],v[s,s])
contour(lon,lat,elev,[0])
cb = colorbar()
cb.set_label('Elevation Angle (deg)')
gca().set_aspect('equal')
xticks(arange(-180,181,45)); yticks(arange(-90,91,45))
```

# Citation
Ibrahim Reda, Afshin Andreas, Solar position algorithm for solar radiation applications, Solar Energy, Volume 76, Issue 5, 2004, Pages 577-589, ISSN 0038-092X, http://dx.doi.org/10.1016/j.solener.2003.12.003.
Keywords: Global solar irradiance; Solar zenith angle; Solar azimuth angle; VSOP87 theory; Universal time; ΔUT1

# LICENSE

Copyright (c) 2023 Samuel Bear Powell

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
