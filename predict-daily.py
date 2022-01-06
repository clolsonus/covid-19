#!/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:07:03 2020

@author: rega0051 (heavily modifed by olson126)
"""

import argparse
from os import path
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

"""
bash$

for i in {621..0}; do
  echo $i
  ./predict-daily.py --no-plot --look-back $i
done

ffmpeg -framerate 5 -pattern_type glob -i '*.png' -c:v libsx264 -r 30 -pix_fmt yuv420p video.mp4 

"""

ap = argparse.ArgumentParser(description="Fit a polynomial to data and plot")
ap.add_argument('--fit-days', type=int, default=90, help='Number of previous days to include in the fit.')
ap.add_argument('--predict-days', type=int, default=21, help='Number of days to predict into the future.')
ap.add_argument('--degree', type=int, default=2, help='Degree of polynomial.')
ap.add_argument('--look-back', type=int, default=0, help='Generate plot for n days ago')
ap.add_argument('--no-plot', action='store_true', help='skip the interactive plot, just save the plot to a file.')
args = ap.parse_args()

pathBase = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
fileConfirmed = path.join(pathBase, 'time_series_covid19_confirmed_global.csv')
fileDeath = path.join(pathBase, 'time_series_covid19_deaths_global.csv')
fileRecovered = path.join(pathBase, 'time_series_covid19_recovered_global.csv')

dfConfirmedRaw = pd.read_csv(fileConfirmed)
dfDeathRaw = pd.read_csv(fileDeath)
dfRecoveredRaw = pd.read_csv(fileRecovered)

#%% Melt the dateframe into the right shape and set index
def cleandata(df_raw):
    df_cleaned=df_raw.melt(id_vars=['Province/State','Country/Region','Lat','Long'],value_name='Cases',var_name='Date')
    df_cleaned=df_cleaned.set_index(['Country/Region','Province/State','Date'])
    return df_cleaned 

#%% Get Countrywise Data
def countrydata(df_cleaned,oldname,newname):
    df_country=df_cleaned.groupby(['Country/Region','Date'])['Cases'].sum().reset_index()
    df_country=df_country.set_index(['Country/Region','Date'])
    df_country.index=df_country.index.set_levels([df_country.index.levels[0], pd.to_datetime(df_country.index.levels[1])])
    df_country=df_country.sort_values(['Country/Region','Date'],ascending=True)
    df_country=df_country.rename(columns={oldname:newname})
    return df_country


#%%
# Clean all datasets
dfConfirmed = countrydata(cleandata(dfConfirmedRaw),'Cases','Confirmed')
dfDeath = countrydata(cleandata(dfDeathRaw), 'Cases', 'Deaths')
dfRecovered = countrydata(cleandata(dfRecoveredRaw),'Cases','Recovered')

dfAll = pd.merge(dfConfirmed, dfDeath,how='left',left_index=True,right_index=True)
dfAll = pd.merge(dfAll, dfRecovered,how='left',left_index=True,right_index=True)

#%% Fit models
def ModelLogistic(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

def FitLogistic(x, y, p0):
    iValid = ~np.isnan(y)
    x = x[iValid]
    y = y[iValid]
    
    pFit, pCov = curve_fit(ModelLogistic,x,y,p0=p0)
    pErr = np.sqrt(np.abs(np.diag(pCov)))
    return pFit, pErr

def ModelExponential(x,a,b,c):
    return c/(0+np.exp(-(x-b)/a))

def FitExponential(x, y, p0):
    iValid = ~np.isnan(y)
    x = x[iValid]
    y = y[iValid]
    
    pFit, pCov = curve_fit(ModelExponential,x,y,p0=p0)
    pErr = np.sqrt(np.abs(np.diag(pCov)))
    return pFit, pErr
    
def gen_func( coeffs, min, max, steps ):
    if abs(max-min) < 0.0001:
        max = min + 0.1
    xvals = []
    yvals = []
    step = (max - min) / steps
    func = np.poly1d(coeffs)
    for x in np.arange(min, max+step, step):
        y = func(x)
        xvals.append(x)
        yvals.append(y)
    return xvals, yvals

def derivative(fit, x):
    #print(fit.shape)
    fitd = np.zeros(fit.shape[0] - 1)
    size = fitd.shape[0]
    for i in range(size):
        fitd[i] = fit[i]*(size-i)
    #print(fit, fitd)
    funcd = np.poly1d(fitd)
    return int(round(funcd(x)))

#%%
#regionList = ['Mexico', 'Italy', 'Peru', 'US', 'France', 'United Kingdom', 'India', 'Brazil']
regionList = ['US']
colorList = ['blue', 'red', 'magenta', 'green', 'orange', 'purple', 'gray', 'black']
param = 'Deaths' # Deaths, Confirmed, Recovered

plt.figure( figsize=(25.60,14.40) )

#fit_days = 28
#pred_days = 14

dt = parser.parse("April 29, 2020") # day index = 98

for iList, regionName in enumerate(regionList):
    print(regionName)
    color = colorList[iList]
    data = dfAll[param][regionName].values

    # daily data
    data_daily = [0]
    for i in range(1,len(data)):
        data_daily.append(data[i] - data[i-1])
    data_daily = np.array(data_daily)
    
    if args.look_back > 0:
        data = data[:-args.look_back]
        data_daily = data_daily[:-args.look_back]
    date = dfAll[param][regionName].index
    print("len:", len(data))
    print(data[-args.fit_days:])
    days = list(range(len(data)-args.fit_days, len(data)))
    
    fit_total, res, _, _, _ = np.polyfit( days, data[-args.fit_days:], args.degree, full=True )
    fit_daily, res, _, _, _ = np.polyfit( days, data_daily[-args.fit_days:], args.degree, full=True )
    func_total = np.poly1d(fit_total)
    func_daily = np.poly1d(fit_daily)
    print("fit:", fit_daily)
    print("res:", res)
    xvals, yvals = gen_func(fit_daily, days[0], days[-1]+args.predict_days, 100)

    # derive current rate (differentiate fit function)
    rate = func_daily(len(data)-1)
    
    #plt.scatter(days, data[-args.fit_days:], color=color, marker='*', label="Points in fit [" + regionName + "]")
    plt.plot(days, data_daily[-args.fit_days:], marker='*', label="Points in fit [" + regionName + "]")
    plt.plot(xvals, yvals, label="Fit " + regionName + " (%.0f/day)" % rate)

    # if param in ['Confirmed']:
    #     dataStart = 5e3
    # else:
    #     dataStart = 100

    # indx = data > dataStart
    # #if regionName == 'US':
    # #    indx[-14:] = False
    # print(indx)
    # dateStart = datetime.strptime("2020-01-01", '%Y-%m-%d')
    # day = np.asarray([(d.date() - dateStart.date()).days for d in date])
    
    # dayThres = day[indx]
    # print(dayThres)
    # dataThres = data[indx]
    # print(dataThres)
    
    # dayPred = 14
    # dayFit = list(range(min(dayThres) - dayPred, max(dayThres) + dayPred))

    # # Shift the start day, so the fit starts at dataStart
    # dayShift = np.interp(dataStart, dataFit, dayFit)
    
    if regionName == 'US':
        #print(indx)
        #print(dayThres)
        #print(dayShift)
        print("historic data:")
        print("day, act, fit (rate)")
        #err = []
        for d in range(len(data_daily)-args.fit_days, len(data_daily)):
            rate = func_daily(d)
            print(d, data_daily[d], int(round(func_daily(d))), rate)
            #err.append(data[d] - int(round(func(d))))
        #print("fit errors, mean:", np.mean(err))
        #print("fit errors, std:", np.std(err))
        print("future fit:")
        for d in range(len(data_daily), len(data_daily) + args.predict_days):
            rate = func_daily(d)
            print(d, int(round(func_daily(d))), rate)

        cdt = dt + timedelta(days=len(data)-99)
        filename = cdt.strftime("%04Y%02m%02d") + ".png"

        # most recent day
        d = len(data_daily) - 1
        y = func_daily(d)
        if y > data_daily[-1]:
            voff = data_daily[-1] - 150
            valign = "bottom"
        else:
            voff = data_daily[-1] + 150
            valign = "top"
        plt.annotate(cdt.strftime("%d %b %Y") + ": %s" % format(data[-1], ',.0f'), xy=(len(data_daily)-1, data_daily[-1]),
                     xytext=(len(data_daily), voff),
                     xycoords='data',
                     #arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='center', verticalalignment=valign)

        # annotate rate for US
        if False:
            d = len(data) - 1
            rate = func_daily(d)
            x = len(data) - 1.5
            y = func_daily(x)
            plt.annotate("Rate: %s deaths/day" % format(int(round(rate)), ',d'),
                         xy=(x, y), xytext=(x+0.5, y+500), xycoords='data',
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='right', verticalalignment='top')
        
            d = len(data) - 1 + 7
            pred_7 = int(round(func_daily(d)))
            print(" 7 day:", d, pred_7)
            plt.annotate("7 day: %s" % format(pred_7, ',d'), xy=(d, pred_7),
                         xytext=(d+0.5, pred_7+500), xycoords='data',
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='center', verticalalignment='bottom')
            d = len(data) - 1 + 14
            pred_14 = int(round(func_daily(d)))
            print(" 14 day:", d, pred_14)
            plt.annotate("14 day: %s" % format(pred_14, ',d'), xy=(d, pred_14),
                         xytext=(d+0.5, pred_14+500), xycoords='data',
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='center', verticalalignment='top')
            # d = len(data) - 1 + 21
            # pred_21 = int(round(func_daily(d)))
            # print(" 21 day:", d, pred_21)
            # plt.annotate("21 day: %s" % format(pred_21, ',d'), xy=(d, pred_21),
            #              xytext=(d-1, pred_21+5000), xycoords='data',
            #              arrowprops=dict(facecolor='black', shrink=0.05),
            #              horizontalalignment='right', verticalalignment='top')

        # use an approximation technique that should work for any function
        x = len(data) - 1
        last = int(data[-1] / 10000)*10000
        while x < len(data) + (args.predict_days*5):
            pred = int(round(func_total(x)))
            pred_10k = int(pred/10000)*10000
            pred_daily = func_daily(x)
            print(x, pred, pred_10k, last)
            
            if pred_10k > last:
                last = pred_10k
                cdt = dt + timedelta(days=x+1-98)
                text = cdt.strftime("%d %b %Y")
                text += ": %s" % format(pred_10k, ',d')
                print(text)
                if x < len(data) + args.predict_days:
                    plt.annotate(text,
                                 xy=(x, pred_daily),
                                 xytext=(x+0.5, pred_daily-500), xycoords='data',
                                 arrowprops=dict(facecolor='black', shrink=0.05),
                                 horizontalalignment='left', verticalalignment='top')
            x += 0.01
        
        
    #%
#    plt.scatter(day - dayShift, data, color=color, marker='.', label = "Raw Points [" + regionName + "]")
    #plt.scatter(dayThres - dayShift, dataThres, color=color, marker='*', label="Points in fit [" + regionName + "]")
    #plt.plot(dayFit - dayShift, dataFit, color=color, label="Fit [" + regionName + "]" )

#plt.xlim(0, 70)
#plt.ylim(bottom = dataStart)
#plt.yscale('log')
plt.grid('on')
plt.legend()
plt.title("Data source: Johns Hopkins CSSE (https://github.com/CSSEGISandData/COVID-19)")
plt.ylabel("Deaths per day")
plt.xlabel("Last %d days fit, %d days predict ahead (Polynomial degree: %d)" % (args.fit_days, args.predict_days, args.degree))

plt.savefig(filename)

if not args.no_plot:
    plt.show()
