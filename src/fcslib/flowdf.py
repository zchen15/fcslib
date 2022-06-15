#!/usr/bin/env python

# Loading libraries
import numpy as np
import scipy as sp
import pandas as pd
import FlowCal

# plotting libraries
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec
import bokeh
import bokeh.plotting
import bokeh.io

# For interfacing with the file system
import fnmatch
import glob
import copy

class flowdf:
    fcs_files = []
    gates = []
    params = {'palette':list(bokeh.palettes.Category10_10),
              'subsamples':1000}

    def __init__(self, files):
        self.fcs_files = files

    def read(files):
        '''
        Read fcs files
        '''
        if type(files) == str:
            if '*' in files:
                files = glob.glob(files)
            else:
                files = [files]
        return FlowFrame(files)

    def write(self, header):
        '''
        Write fcs files 
        '''

    def load(fname):
        '''
        Loads data files
        '''
        df = FlowCal.io.FCSData(fname)
        return pd.DataFrame(df, columns=df.channels)

    def to_dataframe(self):
        '''
        Convert underlying data to pandas dataframe
        '''
        out = []
        for fname in self.fcs_files:
            df = self.load(fname)
            df['filename'] = fname
            out.append(df)
        return pd.concat(out)
    
    def get_data(self):
        '''
        Get data for plotting
        '''
        df = self.to_dataframe()
        # take a subsample
        if self.params['subsamples'] > 0:
            idx = np.random.permutation(len(df))[:self.params['subsamples']]
            df = df.iloc[idx]
        return df

    def copy(self):
        '''
        Copy data in object
        '''
        return copy.deepcopy(self)
    
    def isBool(self, value):
        '''
        Check if it is a boolean array
        '''
        values = np.unique(value)
        # splice based on boolean array
        if len(values) < 3 and type(values[0])==np.bool_ and type(values[-1])==np.bool_:
            return True
        else:
            return False

    def __getitem__(self, key):
        '''
        Selects subset of files
        '''
        out = self.copy()
        # handle boolean or integer slices
        if type(key)==slice or self.isBool(key):
            out.fcs_files = out.fcs_files[key]
        elif type(key)==int:
            out.fcs_files = [out.fcs_files[key]]
        # slicing by filename
        elif type(key)==str:
            out.fcs_files = fnmatch.filter(out.fcs_files, key)
            s = np.sort(out.fcs_files)    
            out.fcs_files = s

        # handle list of files
        elif type(key) == list:
            files = []
            for k in key:
                files+=out[k].fcs_files
            out.fcs_files = files
        return out

    def __setitem__(self, key, value):
        '''
        Sets subset to something
        '''

    def __add__(self, other):
        if type(other).__name__ == 'FlowFrame':
            self.fcs_files+= other.fcs_files
        return self

    def __radd__(self, other):
        if type(other).__name__ == 'FlowFrame':
            self.fcs_files = other.fcs_files + self.fcs_files
        return self

    def __len__(self):
        '''
        Returns number of files in the dataset
        '''
        return len(self.fcs_files)
    
    def check_compensation(self, fig, chan):
        '''
        Check gain compensation results
        '''
        N = len(chan)
        gs = fig.add_gridspec(N-1, N-1)
        ax = None
        for i in range(N-1):
            for j in range(i+1,N):
                ax = fig.add_subplot(gs[j-1,i], sharey=ax, sharex=ax)
                plt.xlabel(chan[i])
                if i==0:
                    plt.ylabel(chan[j])
                self.plot_scatter2d([chan[i], chan[j]])
        ax = fig.add_subplot(gs[0,N-2], sharey=ax, sharex=ax)
        plt.legend() 

    def set_compensation(self, channels, matrix):
        '''
        Set gain compensation matrix and channels
        '''
        matrix = np.eye(len(channels))
        self.params['gain_compensation'] = {'channels':channels, 'matrix':matrix}

    def apply_compensation(self):
        '''
        Apply the gain compensation matrix
        '''
    
    def check_gating(self):
        '''
        Display gating setup
        '''

    def apply_gating(self):
        '''
    
        '''

    def polygon_gate(df, chan, pos):
        '''
        Add polygon gate based on convex hull
        df = input data
        chan = channels
        pos = list of coordinates for convex hull
        '''
        x = df[chan].values



    def smart_ellipse_gate(self):
        '''

        '''

    def dbscan_gate(self):
        '''
        '''


    def plot_density2d(self, channels, palette=None):
        '''
        Generate density2d plot with FlowCal
        '''

    def plot_histogram(self, channels, palette=None, bins=np.logspace(0,4,100)):
        '''
        Plot histogram
        '''

    def plot_scatter2d(self, chan, alpha=0.75, size=1):
        '''
        Plot 2D scatter of data points with matplotlib
        '''
        palette = self.params['palette']
        for i in range(len(self)):
            df = self[i].get_data()
            x = df[chan[0]]
            y = df[chan[1]]
            plt.scatter(x, y, alpha=alpha, s=size, c=palette[i%len(palette)], label=self.fcs_files[i])
            plt.xscale('symlog')
            plt.yscale('symlog')
 
