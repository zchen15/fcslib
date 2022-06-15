#!/usr/bin/env python

# Loading libraries
import numpy as np
import scipy as sp
import scipy.signal
import pandas as pd

class qpcr:
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
        D = qPCR.model_MAK2(D0,Fb,k, len(x))
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

