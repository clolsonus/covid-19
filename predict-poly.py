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

ap = argparse.ArgumentParser(description="Fit a polynomial to data and plot")
ap.add_argument('--fit-days', type=int, default=35, help='Number of previous days to include in the fit.')
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
regionList = ['Mexico', 'Italy', 'Peru', 'US', 'France', 'United Kingdom', 'India', 'Brazil']
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
    if args.look_back > 0:
        data = data[:-args.look_back]
    date = dfAll[param][regionName].index
    print("len:", len(data))
    print(data[-args.fit_days:])
    days = list(range(len(data)-args.fit_days, len(data)))
    
    fit, res, _, _, _ = np.polyfit( days, data[-args.fit_days:], args.degree, full=True )
    func = np.poly1d(fit)
    print("fit:", fit)
    print("res:", res)
    xvals, yvals = gen_func(fit, days[0], days[-1]+args.predict_days, 100)

    # derive current rate (differentiate fit function)
    rate = derivative(fit, len(data)-1)
    
    plt.scatter(days, data[-args.fit_days:], color=color, marker='*', label="Points in fit [" + regionName + "]")
    plt.plot(xvals, yvals, color=color, label="Fit " + regionName + " (%d/day)" % rate)

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
        for d in range(len(data)-args.fit_days, len(data)):
            rate = derivative(fit, d)
            print(d, data[d], int(round(func(d))), rate)
            #err.append(data[d] - int(round(func(d))))
        #print("fit errors, mean:", np.mean(err))
        #print("fit errors, std:", np.std(err))
        print("future fit:")
        for d in range(len(data), len(data) + args.predict_days):
            rate = derivative(fit, d)
            print(d, int(round(func(d))), rate)

        cdt = dt + timedelta(days=len(data)-99)
        filename = cdt.strftime("%04Y%02m%02d") + ".png"

        # most recent day
        plt.annotate(cdt.strftime("%d %b %Y") + ": %s" % format(data[-1], ',.0f'), xy=(len(data)-1, data[-1]),
                     xytext=(len(data)-1.5, data[-1]+10000), xycoords='data',
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='right', verticalalignment='top')

        # annotate rate for US
        d = len(data) - 1
        rate = derivative(fit, d)
        x = len(data) - 1.5
        y = func(x)
        plt.annotate("Rate: %s deaths/day" % format(int(round(rate)), ',d'),
                     xy=(x, y), xytext=(x+0.5, y-10000), xycoords='data',
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='right', verticalalignment='top')
        
        d = len(data) - 1 + 7
        pred_7 = int(round(func(d)))
        print(" 7 day:", d, pred_7)
        plt.annotate("7 day: %s" % format(pred_7, ',d'), xy=(d, pred_7),
                     xytext=(d-1, pred_7+5000), xycoords='data',
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='right', verticalalignment='bottom')
        d = len(data) - 1 + 14
        pred_14 = int(round(func(d)))
        print(" 14 day:", d, pred_14)
        plt.annotate("14 day: %s" % format(pred_14, ',d'), xy=(d, pred_14),
                     xytext=(d-1, pred_14+5000), xycoords='data',
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='right', verticalalignment='top')
        # d = len(data) - 1 + 21
        # pred_21 = int(round(func(d)))
        # print(" 21 day:", d, pred_21)
        # plt.annotate("21 day: %s" % format(pred_21, ',d'), xy=(d, pred_21),
        #              xytext=(d-1, pred_21+5000), xycoords='data',
        #              arrowprops=dict(facecolor='black', shrink=0.05),
        #              horizontalalignment='right', verticalalignment='top')

        # use an approximation technique that should work for any function
        x = len(data) - 1
        last = int(data[-1] / 10000)*10000
        while x < len(data) + (args.predict_days*5):
            pred = int(round(func(x)))
            pred_10k = int(pred/10000)*10000
            # print(x, pred, pred_10k, last)
            
            if pred_10k > last:
                last = pred_10k
                cdt = dt + timedelta(days=x+1-98)
                text = cdt.strftime("%d %b %Y")
                text += ": %s" % format(pred_10k, ',d')
                print(text)
                if x < len(data) + args.predict_days:
                    plt.annotate(text,
                                 xy=(x, pred_10k),
                                 xytext=(x+0.5, pred_10k-10000), xycoords='data',
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
plt.ylabel("Total Deaths (per country)")
plt.xlabel("Last %d days fit, %d days predict ahead (Polynomial degree: %d)" % (args.fit_days, args.predict_days, args.degree))

plt.savefig(filename)

if not args.no_plot:
    plt.show()
