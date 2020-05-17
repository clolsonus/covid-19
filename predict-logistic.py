#!/usr/bin/python3

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:07:03 2020

@author: rega0051
"""

from os import path
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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

#dfAll['Deaths']['Belgium'].tail()

#%% Fit models
def ModelPoly(x,a,b,c,d):
    return a + b*x + c*x*x + d*x*x*x

def FitPoly(x, y, p0):
    iValid = ~np.isnan(y)
    x = x[iValid]
    y = y[iValid] 
    
    pFit, pCov = curve_fit(ModelPoly,x,y,p0=p0)
    pErr = np.sqrt(np.abs(np.diag(pCov)))
    return pFit, pErr

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
    
#%%
regionList = ['Belgium', 'Italy', 'Spain', 'US', 'France', 'United Kingdom', 'Iran', 'Brazil']
colorList = ['blue', 'red', 'magenta', 'green', 'orange', 'purple', 'gray', 'black']
param = 'Deaths' # Deaths, Confirmed, Recovered

plt.figure()

for iList, regionName in enumerate(regionList):
    print(regionName)
    color = colorList[iList]
    data = dfAll[param][regionName].values
    date = dfAll[param][regionName].index
    print(data)
    
    if param in ['Confirmed']:
        dataStart = 5e3
    else:
        dataStart = 100

    indx = data > dataStart
    #if regionName == 'US':
    #    indx[-14:] = False
    print(indx)
    dateStart = datetime.strptime("2020-01-01", '%Y-%m-%d')
    day = np.asarray([(d.date() - dateStart.date()).days for d in date])
    
    dayThres = day[indx]
    print(dayThres)
    dataThres = data[indx]
    print(dataThres)
    
    dayPred = 14
    dayFit = list(range(min(dayThres) - dayPred, max(dayThres) + dayPred))

    if regionName in []:
        pFit, pErr = FitPoly(dayThres, dataThres, p0=[0,0,0,0])
        dataFit = [ModelPoly(i, pFit[0], pFit[1], pFit[2], pFit[3]) for i in dayFit]
        
    elif regionName not in ['Belgium', 'China', 'France', 'Italy', 'Spain', 'Iran', 'United Kingdom', 'US', 'Brazil']:
        pFit, pErr = FitExponential(dayThres, dataThres, p0=[2,100,20000])
        dataFit = [ModelExponential(i,  pFit[0],  pFit[1],  pFit[2]) for i in dayFit]
    else:
        pFit, pErr = FitLogistic(dayThres, dataThres, p0=[2,100,20000])
        dataFit = [ModelLogistic(i, pFit[0], pFit[1], pFit[2]) for i in dayFit]

    # Shift the start day, so the fit starts at dataStart
    dayShift = np.interp(dataStart, dataFit, dayFit)
    
    if regionName == 'US':
        print(indx)
        print(dayThres)
        print(dayShift)
        print("historic data:")
        print("day, act, fit")
        for i, d in enumerate(dayThres):
            print(d, dataThres[i], int(round(ModelLogistic(d, pFit[0], pFit[1], pFit[2]))))
        print("future fit:")
        for d in range(dayThres[-1]+1, dayThres[-1]+15):
            print(d, int(round(ModelLogistic(d, pFit[0], pFit[1], pFit[2]))))

        d = dayThres[-1]+7
        print(" 7 day:", d, int(round(ModelLogistic(d, pFit[0], pFit[1], pFit[2]))))
        d = dayThres[-1]+14
        print("14 day:", d, int(round(ModelLogistic(d, pFit[0], pFit[1], pFit[2]))))
    #%
#    plt.scatter(day - dayShift, data, color=color, marker='.', label = "Raw Points [" + regionName + "]")
    plt.scatter(dayThres - dayShift, dataThres, color=color, marker='*', label="Points in fit [" + regionName + "]")
    plt.plot(dayFit - dayShift, dataFit, color=color, label="Fit [" + regionName + "]" )

plt.xlim(0, 70)
plt.ylim(bottom = dataStart)
plt.yscale('log')
plt.grid('on')
plt.legend()
plt.xlabel("Days since reaching starting threshold [Deaths > " + str(dataStart) + "]")
plt.show()
