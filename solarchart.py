"""
Created on Wed Jul 26 11:35:26 2023.

################################################################################

Copyright 2023, Samuel B Powell

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
import numpy as np
import datetime
import matplotlib.pyplot as plt

import sunposition
#sunposition.jit.apply() #use jit in sunpos routines, if available

def _parse_args(args, **kw):
    p = argparse.ArgumentParser()
    p.add_argument('-lat', '--latitude',type=float,default=50)
    p.add_argument('-lon','--longitude',type=float,default=0)
    p.add_argument('-e','--elevation',type=float,default=0)
    p.add_argument('-y','--year',type=int,default=datetime.datetime.now().year)
    p.add_argument('-z','--timezone',type=float,default=None,help='timezone hour offset. Defaults to round(longitude*12/180)')
    p.add_argument('-o','--output',help='Plot file',default='chart.png')
    args = p.parse_args(args, argparse.Namespace(**kw))
    if args.timezone is None:
        args.timezone = round(args.longitude*12/180)
    return args

def find_solstices(sun_zen):
    daily_min_zen = np.min(sun_zen,axis=1)
    winter_solstice = np.argmax(daily_min_zen)
    summer_solstice = np.argmin(daily_min_zen)
    return winter_solstice, summer_solstice

def find_equinoxes(sun_zen, winter_solstice, summer_solstice):
    #each interval is 1 minute, count intervals with sun over the horizon to get minutes of daylight
    daylight_hours = np.sum(sun_zen <= 90, axis=1)/60 #divide by 60 to get hours of daylight
    hours_err = np.abs(daylight_hours - 12) #equinoxes are exactly 12 hour days
    
    s1, s2 = sorted((winter_solstice,summer_solstice))
    eq1 = s1 + np.argmin(hours_err[s1:s2]) #between solstices s1 & s2
    eq2 = np.argmin(hours_err[:s1])
    eq2_b = s2 + np.argmin(hours_err[s2:])
    if hours_err[eq2_b] < hours_err[eq2]:
        eq2 = eq2_b
    if s1 == winter_solstice:
        spring_equinox, fall_equinox = eq1, eq2
    else:
        spring_equinox, fall_equinox = eq2, eq1
    return spring_equinox, fall_equinox
    
def format_timestamp(ts,fmt):
    return datetime.datetime.utcfromtimestamp(ts).strftime(fmt)


def main(args=None, **kw):
    args = _parse_args(args, **kw)
        
    #to compute the solar chart we need:
    #  the position of the sun at a fine resolution for key days of the year
    #  the position of the sun at at certain hours for every day of the year
    
    #the easiest way is to compute sun position at a fine resolution for every day
    # then select the important parts later
    
    #we'll start with timestamps for the first day of the year, and the next year
    # this will automatically deal with e.g. leap years
    # NB. we're specifying UTC and manually subtracting the given timezone because datetime's
    #    timezone classes are a poorly documented mess
    utc = datetime.timezone.utc
    timezone_sec = args.timezone*60*60
    t0 = datetime.datetime(args.year,1,1,tzinfo=utc).timestamp() - timezone_sec
    t1 = datetime.datetime(args.year+1,1,1,tzinfo=utc).timestamp() - timezone_sec
    
    sec_per_day = 24*60*60 #seconds per day
    days = np.arange(0, (t1-t0)//sec_per_day, 1, int) #day number 0..364 (or 365 on leap years)
    days_ts = t0 + sec_per_day*days #timestamp of each day, in seconds
    #days_dt = days_ts.astype('datetime64[s]') #convert timestamps to numpy datetimes
    times_of_day_sec = np.arange(0, sec_per_day, 60) #1 minute intervals over the day
    
    #make the timestamps for each day x time_of_day
    ts = np.add.outer(days_ts, times_of_day_sec) #add each pair
    
    sun_a, sun_z = sunposition.observed_sunpos(ts, args.latitude, args.longitude, args.elevation)
        
    #find the times that correspond to each hour
    #the first sample of each day is midnight, and each interval is 1 minute
    hours_i = np.arange(24)*60
    
    #for the full days that we plot, we want to use the solstices and equinoxes as reference points
    #we'll plot 2 days between each, for a total of 12 days, separated by approximately a month each
        
    ## for plotting, we want the winter-spring lines to be solid, summer-autumn to be dashed
    #the easiest way to do so is to rotate the data so that the winter solstice is the first day in the dataset
    winter_i, summer_i = find_solstices(sun_z)
    ts_rot = np.concatenate((ts[winter_i:], ts[:winter_i]),axis=0)
    sun_a_rot = np.concatenate((sun_a[winter_i:], sun_a[:winter_i]),axis=0)
    sun_z_rot = np.concatenate((sun_z[winter_i:], sun_z[:winter_i]),axis=0)
    # winter is now at 0
    winter_i_rot, summer_i_rot = 0, (summer_i - winter_i) % len(days)
    
    #split each season into 6 approximate months
    d_months = (np.array([summer_i-winter_i, winter_i-summer_i]) % len(days))/6
    months_i = (np.array([winter_i, summer_i]) + np.outer(range(6), d_months)) % len(days)
    months_i = months_i.T.reshape((-1)).astype(np.int64)
    months_i_rot = (months_i - winter_i) % len(days)
    
    #mask out daylight hours, don't plot when the sun is below the horizon
    daylight_mask = sun_z_rot <= 90
    
    #set up a radial plot with N at the top, + is to the East
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(projection='polar')
    ax.set_rlim(0,90)
    ax.set_theta_zero_location('N') #0 at the top
    ax.set_theta_direction(-1) # + is E from N
    
    #plot the a day-line for each month
    for di in months_i_rot:
        m = daylight_mask[di]
        th, r = np.deg2rad(sun_a_rot[di][m]), sun_z_rot[di][m]
        if len(th) == 0: continue #no daylight on this day
        label = format_timestamp(ts_rot[di,0] + timezone_sec,'%Y-%m-%d')
        if di <= summer_i_rot: #spring
            fmt = 'k-'
            label_xy = (th[0], r[0]*1.02)
        else: #fall
            fmt = 'k--'
            label_xy = (th[-1],r[-1]*1.02)
        ha = 'left' #label horizontal alignment
        if label_xy[0] > np.pi: ha = 'right' #the label is on the left side of the figure
        
        ax.plot(th, r, fmt, label=label)
        ax.annotate(label, label_xy, horizontalalignment=ha, annotation_clip=False)
        
    #plot the hour lines
    for ti in hours_i:
        m = daylight_mask[:,ti]
        th, r = np.deg2rad(sun_a_rot[:,ti]), sun_z_rot[:,ti]
        #split into spring & fall segments, overlapping by a day, apply mask
        th_spring, r_spring = th[:summer_i_rot+1][m[:summer_i_rot+1]], r[:summer_i_rot+1][m[:summer_i_rot+1]]
        th_fall, r_fall = th[summer_i_rot:][m[summer_i_rot:]], r[summer_i_rot:][m[summer_i_rot:]]
        if len(th_spring) + len(th_fall) == 0: continue #no daylight at this time on any date
        label = format_timestamp(ts_rot[0,ti] + timezone_sec, '%H')
        #we want to put the labels near the summer solstice line
        label_xy = th_fall[0], r_fall[0]*0.98
        
        ax.plot(th_spring, r_spring, 'k-', label=label)
        ax.plot(th_fall, r_fall, 'k--')
        has = ['center','right','right','center','center','left','left','center'] #horizontal alignment for each eighth
        ha = has[int(label_xy[0]*1.2732395447351628) % 8] # 8/(2 pi)
        
        ax.annotate(label, label_xy, horizontalalignment=ha,annotation_clip=False)
    
    fig.savefig(args.output,dpi=300)
    
    
    
if __name__ == '__main__':
    main()
