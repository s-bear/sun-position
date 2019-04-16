from sunposition import Sunpos_Localised
import pytz
from datetime import datetime

def approx_equal(output, test_case):
    accuracy_limit = 4
#    print(output, test_case)
    assert abs(output[0] -test_case[0]) < accuracy_limit
    assert abs(output[1] -test_case[1]) < accuracy_limit


def test_sunpos_localised():
    """
    Comparison values obtained from www.suncalc.org on 06/04/19
    Also
    https://sunposition.info/sunposition/spc/locations.php#1 on 16/04/19
    It seemes some of the results from the first site were incompatible
    """

        # London UK
    lat1 = 51.5
    long1 = -0.17
    height1 = 24
    tz1 = 'Europe/London'

    localised1 = Sunpos_Localised(lat1, long1, height1, tz1)
    localised1_UTC = Sunpos_Localised(lat1, long1, height1)

        #New York US
    lat2 = 40.64
    long2 = -73.77
    height2 = 3
    tz2 = 'America/New_York'

    localised2 = Sunpos_Localised(lat2, long2, height2, tz2)

        #Brisbane Australia
    lat3 = -27.50
    long3 = 153.02    
    height3 = 10
    tz3 = 'Australia/Brisbane'

    localised3 = Sunpos_Localised(lat3, long3, height3, tz3)

    inputs = [
    datetime(2019, 4, 6, 16, 20),   #0
    datetime(2005, 1, 3, 12, 4),    #1
    datetime(2019, 4, 6, 15, 20),   #2
    datetime(2019, 4, 6, 15, 15),   #3
    datetime(2019, 6, 7, 11, 30),   #4
    datetime(2019, 1, 1, 12, 00),   #5
    datetime(2019, 6, 7, 11, 25)    #6
    ]

    # elevation, azimuth
    results = [
    (29.54, 239.82), #1, '16:20 06/04/2019', UTC+1
    (15.77, 179.69), #1 '12:04, 03/01/2005'
    (37, 224), #1, '15:20 06/04/2019', UTC+1
    (44, 231), #2, '15:20 06/04/2019', UTC-4
    (64, 126), #2, '11:25 06/07/2019', UTC-4
    (85.11, 335.44), #3, '12:00 01/01/2019', UTC-4
    (39.35, 8.23) #3, '11:25 06/07/2019', UTC-4
    ]

    approx_equal(localised1.sunpos(inputs[0]), results[0]) #Good
    approx_equal(localised1.sunpos(inputs[1]), results[1]) #Good
    approx_equal(localised1.sunpos(inputs[2]), results[2])
    approx_equal(localised2.sunpos(inputs[3]), results[3])
    approx_equal(localised2.sunpos(inputs[4]), results[4])
    approx_equal(localised3.sunpos(inputs[5]), results[5]) #Good
    approx_equal(localised3.sunpos(inputs[6]), results[6]) #Good
    approx_equal(localised1_UTC.sunpos(inputs[2]), results[0]) # UTC test case #Good


if __name__ == '__main__':
    test_sunpos_localised()