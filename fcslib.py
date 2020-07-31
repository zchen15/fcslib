#!/usr/bin/env python

# Loading libraries
import numpy as np
import scipy as sp
import scipy.signal
import pandas as pd
import nptdms

# For interfacing with the file system
import glob
import subprocess
import os
import time
import sys
import logging
import argparse
import json

def model_MAK2(D0,Fb,k,n):
    '''
    Defining the MAK2 model
    '''
    k = abs(k)
    D0 = abs(D0)
    D = np.zeros(n)
    D[0] = D0
    for i in range(1,n):
        v = 1+D[i-1]/k
        D[i] = D[i-1]+k*np.log(v)
    return D

def model_MAK2_error(y, x):
    '''
    Objective function for MAK2 qPCR model
    '''
    [D0,Fb,k] = y
    D = mak2(D0,Fb,k, len(x))
    error = x - Fb - D
    return np.sum(error**2)

def model_MAK2_range(y, th=10):
    '''
    Find range for fitting the MAK2 qPCR model. Needs polymerase to not be limiting reagent.
    --> still requires some debugging to work well
    '''
    filt = [1,1,0,-1,-1]
    y1 = sp.signal.fftconvolve(y,filt, mode='same')
    y2 = sp.signal.fftconvolve(y1,filt, mode='same')
    pos = y2 > 0
    pos[1:] = pos[1:] == ~pos[0:-1]
    thresh = y > th
    L = np.arange(0,len(y))[pos & thresh]
    return L[0]

def model_MAK2_fit(df, chan='FAM', th=10):
    '''
    Curve fitting on MAK2 model
    --> requires more debugging on optimal fit range
    '''
    well = df['well'].drop_duplicates().values
    data = []
    for i in range(0,len(well)):
        w = well[i]
        if i%6==0:
            print('processing well=',w)
        c1 = df['well']==w
        L = get_range(df[c1][chan], th=th)
        c2 = (df['Cycle'] <= 36) & (df['Cycle'] > 10)
        y = df[c1 & c2][chan].values
        res = sp.optimize.minimize(mak2_error, [100,-100,1], args=y, method='BFGS')
        [D0,Fb,k] = res.x
        D0 = abs(D0)
        k = abs(k)
        data.append([w,D0,Fb,k, res.fun, res.success])
    return pd.DataFrame(data, columns=['well','D0','Fb','k','error','success'])

def read_tdms(fname, chan=6):
    '''
    Reads data from tdms file with nptdms
    return dataframe of time and channels
    '''
    tdms_file = nptdms.TdmsFile.read(fname)
    df = tdms_file.as_dataframe(absolute_time=True, scaled_data=False)
    df = df[df.columns[:chan]]
    df.columns = ['ai'+str(i) for i in range(0,chan)]
    # Add time information
    all_groups = tdms_file.groups()
    c = str(all_groups[0]).split('\'')[1]
    group = tdms_file[c]
    t = group.channels()[0].time_track()
    df['time (s)'] = t
    return df

def channel_offset(df):
    '''
    Find time offset between channels and align them
    '''
    N = 6
    chan = ['ai'+str(i) for i in range(0,N)]
    offset = np.zeros((N,N))
    mag = np.zeros((N,N))
    for i in range(0,N):
        for j in range(i,N):
            y1 = df[chan[i]]
            y2 = df[chan[j]]
            z = sp.signal.fftconvolve(y1, y2[::-1], mode='same')
            v = np.argmax(z)
            offset[i,j] = v
            offset[j,i] = offset[i,j]
            mag[i,j] = z[v]
            mag[j,i] = mag[i,j]
    return offset, mag

def filter_bp(y, ntaps=1001, hp=1, lp=2e3, fs=20e3):
    filt = sp.signal.firwin(ntaps, [hp, lp], pass_zero=False, fs=fs)
    fy = sp.signal.fftconvolve(y, filt, mode='same')
    # apply hamming window to remove window noise
    filt = sp.signal.windows.hamming(ntaps)
    filt = filt/np.sum(filt)
    return sp.signal.fftconvolve(fy, filt, mode='same')

def filter_wiener(y, ntaps=9):
    fy = sp.signal.wiener(y, mysize=ntaps)
    filt = sp.signal.windows.hamming(ntaps)
    filt = filt/np.sum(filt)
    return sp.signal.fftconvolve(fy, filt, mode='same')

def filter_med(y, ntaps=9):
    fy = sp.signal.medfilt(y, kernel_size=ntaps)
    # apply hamming window to remove window noise
    filt = sp.signal.windows.hamming(ntaps)
    filt = filt/np.sum(filt)
    return sp.signal.fftconvolve(fy, filt, mode='same')

def filter_sg(y, ntaps=11, order=3):
    fy = sp.signal.savgol_filter(y, ntaps, order)
    # apply hamming window to remove window noise
    filt = sp.signal.windows.hamming(ntaps)
    filt = filt/np.sum(filt)
    return sp.signal.fftconvolve(fy, filt, mode='same')

def plot_line_bokeh(x, y, color='blue', label='signal', p=None, title='', xlabel='time (s)', ylabel='mV', w=800, h=250):
    '''
    Create a bokeh plot
    p = bokeh plot handle if it exists
    title = title string
    w = width of plot in pixels
    h = height of plot in pixels
    return bokeh plot object
    '''
    # define a plot width and
    if p == None:
        p = bokeh.plotting.figure(plot_width=800, plot_height=250)
        # set axis labels
        p.title.text = title
        p.xaxis.axis_label = xlabel
        p.yaxis.axis_label = ylabel
        # add mutable legends
        p.legend.location = "top_left"
        p.legend.click_policy="mute"

    p.line(x, y, line_width=1, color=color, alpha=0.8, muted_color=color, muted_alpha=0.1, legend_label=label)
    return p

def find_peaks(y):
    '''
    Takes filtered signal and finds peak/trough
    return dataframe with columns [index, peak, trough]
    '''
    # get 1st derivative
    filt = [-.5,-.5, 0, .5, .5]
    dy1 = sp.signal.fftconvolve(-1.0*y, filt, mode='same')
    # get positions when 1st deriv crosses zero
    pos = dy1 > 0
    # label peaks and troughs
    pk = [False]*len(y)
    pk[1:] = (pos[0:-1] == ~pos[1:]) & (pos[0:-1] == True)
    tr = [False]*len(y)
    tr[1:] = (pos[0:-1] == ~pos[1:]) & (pos[0:-1] == False)
    data = np.transpose([np.arange(len(y)),pk,tr])
    data =  pd.DataFrame(data, columns=['index','peak','trough'])
    for col in ['peak','trough']:
        data[col] = data[col].astype(bool)
    return data

def find_droplet(df):
    '''
    Takes peaks/troughs and segments droplets from raw signal
    df = dataframe with [time, signal, peak, trough]
    returns dataframe with [time, peak height, width, and etc]
    '''
    # label pulses based on peaks and trough
    df['index'] = range(0,len(df))
    c = (df['peak']) | (df['trough'])
    pk = df[c]
    # start with trough
    if pk.iloc[0]['peak']:
        pk = pk.iloc[1:]
    # find cut points
    s1 = pk.iloc[::2]['index'].values
    s2 = pk.iloc[1::2]['index'].values
    s3 = pk.iloc[2::2]['index'].values
    # make sure lengths are the same
    L = np.min([len(s1),len(s2),len(s3)])
    s1 = s1[:L]
    s2 = s2[:L]
    s3 = s3[:L]
    t = df['time (s)'].values[s2]
    # get heights
    fy = df['signal'].values
    lh = fy[s1]
    ph = fy[s2]
    rh = fy[s3]
    # get widths
    lw = s2-s1
    rw = s3-s2
    # get AOC
    lA = []
    rA = []
    for i in range(0,len(s1)):
        lA.append(np.sum(fy[s1[i]:s2[i]]))
        rA.append(np.sum(fy[s2[i]:s3[i]]))
    data = np.transpose([t,s1,s2,s3,lh,ph,rh,lw,rw,lA,rA])
    col = ['time (s)','x1','x2','x3','lh','ph','rh','lw','rw','lA','rA']
    data = pd.DataFrame(data, columns=col)
    data['w'] = data['lw'] + data['rw']
    data['A'] = data['lA'] + data['rA']
    return data

def main():
    print('Hello world')

if __name__ == "__main__":
    main()
