
import urllib
import urllib.request
import sunposition
import numpy as np

# We can query the official online implementation of the sun position algorithm for reference data:
# e.g. as of 2025-01-16, the following query
#   https://midcdmz.nrel.gov/apps/spa.pl?syear=2020&smonth=1&sday=1&eyear=2020&emonth=1&eday=1&step=10&stepunit=1&otype=1&hr=8&min=15&sec=9&latitude=39.743&longitude=-105.178&timezone=-7.0&elev=1829&press=835&temp=10&dut1=0.0&deltat=64.797&azmrot=180&slope=0&refract=0.5667&field=0&field=1&field=7&field=35&field=36&field=37
# Produced this data:
#   Date (M/D/YYYY),Time (H:MM:SS),Topocentric zenith angle,Top. azimuth angle (eastward from N),Julian day,Topocentric sun declination,Topocentric sun right ascension,Topocentric local hour angle
#   1/1/2020,8:15:09,82.258754,128.650576,2458850.135521,-23.010468,281.592489,302.761022

def temperature(h, T0):
    '''approximate temperature at a given altitude'''
    # T = T0 - L*h
    return T0 - 0.0065*h

def pressure(h, T0 = 288.16, p0 = 1013.25):
    '''approximate pressure at a given altitude'''
    # P = P0*(T/T0)**(g/(L*R))
    T = temperature(h, T0)
    return p0*(T/T0)**(9.80665/(0.0065*287.05))

t_to_datetime = np.vectorize(sunposition._time_i64_to_datetime)

n = 50
# variable ranges for random sampling
tmin, tmax = sunposition.time_to_datetime64(['-2000-01-01T00:00','6000-12-31T00:00']).astype(np.int64)
rng = np.random.default_rng()
times = rng.integers(tmin, tmax, n)
years, months, days, hours, minutes, seconds = t_to_datetime(times)[:6]
latitudes = rng.uniform(-90, 90, n)
longitudes = rng.uniform(-180,180, n)
elevations = rng.uniform(-450, 10000, n) #dead sea to airplane
surface_temperatures = rng.uniform(-25, 45, n)
deltats = rng.uniform(0,1,n)
refracts = np.array([0.5667]*n)
temperatures = temperature(elevations, surface_temperatures)
pressures = pressure(elevations, surface_temperatures+274.15)

def _build_query(url, **kwargs):
    parameters = []
    for key, value in kwargs.items():
        fmt = '{}={}'
        if isinstance(value, tuple):
            value, fmt = value
            fmt = f'{{}}={{:{fmt}}}'
        if not isinstance(value, (list, tuple)):
            value = [value]
        for v in value:
            parameters.append(fmt.format(key,v))
    parameters = '&'.join(parameters)
    return f'{url}?{parameters}'

@np.vectorize
def build_query(year, month, day, hour, minute, second, latitude, longitude, elevation, pressure, temperature, dt, refract):
    fields = [0,1,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]
    return _build_query('https://midcdmz.nrel.gov/apps/spa.pl', syear=year, smonth=month, sday=day,
                        eyear=year, emonth=month, eday=day, step=10, stepunit=1, otype=1, 
                        hr=hour, min=minute, sec=second, timezone=0, dut1=0, deltat=f'{dt:0.3f}',
                        latitude=f'{latitude:0.6f}',longitude=f'{longitude:0.6f}',elev=f'{elevation:0.0f}',
                        press=f'{pressure:0.0f}', temp=f'{temperature:0.1f}', refract=f'{refract:0.4f}',
                        azmrot=180, slope=0, field=fields
                        )

queries = build_query(years, months, days, hours, minutes, seconds, latitudes, longitudes, elevations, pressures, temperatures, deltats, refracts)

'''
Date (M/D/YYYY),Time (H:MM:SS),Topocentric zenith angle,Top. azimuth angle (eastward from N),Julian day,Julian century,Julian ephemeris day,Julian ephemeris century,Julian ephemeris millennium,Earth heliocentric longitude,Earth heliocentric latitude,Earth radius vector,Geocentric longitude,Geocentric latitude,Mean elongation (moon-sun),Mean anomaly (sun),Mean anomaly (moon),Argument latitude (moon),Ascending longitude (moon),Nutation longitude,Nutation obliquity,Ecliptic mean obliquity,Ecliptic true obliquity,Aberration correction,Apparent sun longitude,Greenwich mean sidereal time,Greenwich sidereal time,Geocentric sun right ascension,Geocentric sun declination,Observer hour angle,Sun equatorial horizontal parallax,Sun right ascension parallax,Topocentric sun declination,Topocentric sun right ascension,Topocentric local hour angle,Top. elevation angle (uncorrected),Atmospheric refraction correction,Top. elevation angle (corrected),Equation of time
1/1/2020,8:15:09,82.258754,128.650576,2458850.135521,0.200004,2458850.136271,0.200004,0.020000,100.667759,0.000123,0.983282,280.667759,-0.000123,89352.933823,7557.472090,95576.517185,96735.478044,-261.789865,-0.004588,-0.000468,84372.085975,23.436223,-0.005788,280.657383,329.535720,329.531511,281.590741,-23.008609,302.762770,0.002484,0.001748,-23.010468,281.592489,302.761022,7.646688,0.094559,7.741246,-3.383677
'''

def fix_date_time(d, t):
    '''convert M/D/YYYY, H:MM:SS to ISO8601'''
    month, day, year = d.split('/')
    hour, minute, second = t.split(':')
    return f'{year}-{month:>02}-{day:>02}T{hour:>02}:{minute:>02}:{second:>02}Z'

expected_header = b'Date (M/D/YYYY),Time (H:MM:SS),Topocentric zenith angle,Top. azimuth angle (eastward from N),Julian day,Julian century,Julian ephemeris day,Julian ephemeris century,Julian ephemeris millennium,Earth heliocentric longitude,Earth heliocentric latitude,Earth radius vector,Geocentric longitude,Geocentric latitude,Mean elongation (moon-sun),Mean anomaly (sun),Mean anomaly (moon),Argument latitude (moon),Ascending longitude (moon),Nutation longitude,Nutation obliquity,Ecliptic mean obliquity,Ecliptic true obliquity,Aberration correction,Apparent sun longitude,Greenwich mean sidereal time,Greenwich sidereal time,Geocentric sun right ascension,Geocentric sun declination,Observer hour angle,Sun equatorial horizontal parallax,Sun right ascension parallax,Topocentric sun declination,Topocentric sun right ascension,Topocentric local hour angle,Top. elevation angle (uncorrected),Atmospheric refraction correction,Top. elevation angle (corrected),Equation of time\n'
data_headings = str(expected_header,'utf8').strip().split(',')

new_headings_order = ['t','lat','lon','elev','temp','pres','refract','delta_t','Top. azimuth angle (eastward from N)','Topocentric zenith angle','Topocentric sun right ascension','Topocentric sun declination','Topocentric local hour angle','Julian day','Julian century','Julian ephemeris day','Julian ephemeris century','Julian ephemeris millennium','Earth heliocentric longitude','Earth heliocentric latitude','Earth radius vector','Geocentric longitude','Geocentric latitude','Mean elongation (moon-sun)','Mean anomaly (sun)','Mean anomaly (moon)','Argument latitude (moon)','Ascending longitude (moon)','Nutation longitude','Nutation obliquity','Ecliptic mean obliquity','Ecliptic true obliquity','Aberration correction','Apparent sun longitude','Greenwich mean sidereal time','Greenwich sidereal time','Geocentric sun right ascension','Geocentric sun declination','Observer hour angle','Sun equatorial horizontal parallax','Sun right ascension parallax','Top. elevation angle (uncorrected)','Atmospheric refraction correction','Top. elevation angle (corrected)','Equation of time']
new_headings = ['t','lat','lon','elev','temp','pres','refract','delta_t','topo_azimuth','topo_zenith','topo_right_asc','topo_decl','topo_hour','julian_day','julian_century','julian_eph_day','julian_eph_century','julian_eph_millennium','earth_helio_lon','earth_helio_lat','earth_rad','geo_lon','geo_lat','mean_elongation','mean_anomaly_sun','mean_anomaly_moon','arg_lat_moon','asc_lon_moon','nutation_lon','nutation_obliquity','ecliptic_mean_obliquity','ecliptic_obliquity','aberration_correction','sun_lon','greenwich_mean_sidereal_t','greenwich_sidereal_t','geo_right_asc','geo_decl','observer_hour','sun_horizontal_parallax','sun_right_asc_parallax','topo_elevation_uncorrected','atmos_refract','topo_elevation','eq_of_t']
new_header = ','.join(new_headings)

filename = 'test_data.csv'
with open(filename,'w') as outfile:
    outfile.write(new_header)
    outfile.write('\n')

for lat,lon,elev,pres,temp,dt,r,query in zip(latitudes,longitudes,elevations,pressures,temperatures,deltats, refracts, queries):
    print(f'Fetching {query}', flush=True)
    with urllib.request.urlopen(query) as response:
        if response.status != 200:
            print(f'HTTPS Status: {response.status}: {response.reason}')
            continue
        lines = response.readlines()
    header = lines[0]
    data = lines[1:]
    if header != expected_header:
        print('Unexpected data header: ')
        print(header, flush=True)
        continue
    entry = {'lat':f'{lat:0.6f}','lon':f'{lon:0.6f}','elev':f'{elev:0.0f}','temp':f'{temp:0.0f}',
             'pres':f'{pres:0.0f}','refract':f'{r:0.4f}','Delta_t':f'{dt:0.3f}'}
    with open(filename,'a') as outfile:
        for line in data:
            entry.update(zip(data_headings, str(line,'utf8').strip().split(',')))
            date, time = entry['Date (M/D/YYYY)'], entry['Time (H:MM:SS)']
            
            entry['t'] = fix_date_time(date,time)
            
            new_line = ','.join(entry[key] for key in new_headings_order)
            outfile.write(new_line)
            outfile.write('\n')

