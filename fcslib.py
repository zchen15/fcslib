#!/usr/bin/env python

# Loading libraries
import numpy as np
import scipy as sp
import scipy.signal
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
import subprocess
import os
import time
import sys
import logging
import argparse
import json

class qPCR:
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

class DropletPCR:
    def read_tdms(fname, chan=6, fs=20e3, get_time=False):
        '''
        Reads data from tdms file with nptdms
        chan = number of channels
        fs = sampling rate. Defaults to 20Hz
        get_time = try to get default time encoded in tdms file
        return dataframe of time and channels
        '''
        import nptdms
        tdms_file = nptdms.TdmsFile.read(fname)
        df = tdms_file.as_dataframe(absolute_time=True, scaled_data=False)
        df = df[df.columns[:chan]]
        df.columns = ['ai'+str(i) for i in range(0,chan)]
        if get_time:
            # Add time information
            all_groups = tdms_file.groups()
            c = str(all_groups[0]).split('\'')[1]
            group = tdms_file[c]
            t = group.channels()[0].time_track()
            df['time (s)'] = t
        else:
            df['time (s)'] = np.arange(len(df))/fs
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
        '''
        Takes numpy array and filters the signal with savitzky golay filter and hamming window filter
        '''
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

    def normalize_signal(df, ntaps=9):
        '''
        Normalize the channels relative to each other using background noise
        returns dataframe normalized to noise
        '''
        com = np.ones(len(df))
        for i in range(0,6):
            # get the channel
            y = df['ai'+str(i)]
            # using wiener filter to get the approximate true signal
            fy = filter_sg(y, ntaps)
            # get the noise
            noise = fy-y
            # background subtract and normalize to noise
            ynorm = (y-np.mean(y))/np.std(noise)
            df['norm_ai'+str(i)] = ynorm
            # merge channel into common signal for rough signal segmentation
            com+=ynorm
        # clean up the common channel
        y = com
        # using wiener filter to get the approximate true signal
        fy = filter_sg(y, ntaps)
        # get the noise
        noise = fy-y
        # background subtract and normalize to noise
        ynorm = (y-np.mean(y))/np.std(noise)
        df['com_ai'] = ynorm
        return df

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
        df = dataframe with [time (s), peak, trough, signal]
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
        data = np.transpose([t,lh,ph,rh,lw,rw,lA,rA])
        col = ['time (s)','lH','H','rH','lW','rW','lA','rA']
        data = pd.DataFrame(data, columns=col)
        data['W'] = data['lW'] + data['rW']
        data['A'] = data['lA'] + data['rA']
        return data

    def get_droplets(fname, ntaps=11, N_chan=6, fs=20e3):
        '''
        Pipeline to convert time series signal into FCS signal
        ntaps = length of wiener filter used
        N_chan = name of channels
        fs = sampling frequency
        return dataframe with droplets width, height, area, and location
        '''
        # load tdms file
        print('Loading ', fname)
        df = read_tdms(fname, chan=N_chan, fs=fs)
        # normalize signal intensity using background noise
        print('Normalizing signal intensities and getting common signal')
        df = normalize_signal(df, ntaps=ntaps)
        # find peaks using first derivative on common signal
        print('Finding peaks')
        pk = find_peaks(df['com_ai'])
        # merge peak calling info into time series dataframe
        df['peak'] = pk['peak']
        df['trough'] = pk['trough']
        # get droplet signal for each channel
        data = []
        channels = ['ai'+str(i) for i in range(N_chan)] + ['com_ai']
        for chan in channels:
            print('Computing droplet signal on '+chan)
            x = df[['time (s)','peak','trough']]
            # filter the signal
            x['signal'] = filter_sg(df[chan], ntaps=ntaps, order=3)
            out = find_droplet(x)        
            out.columns = ['time (s)'] + [chan+'-'+col for col in out.columns[1:]]
            data.append(out)
        # merge data frames
        print('Merging dataframes')
        df = data[0]
        for i in range(1,len(data)):
            df = df.merge(data[i], on='time (s)', how='left')
        df['src'] = fname
        return df

    def gate_droplets(df):
        '''
        Gate droplets using hdbscan to find and exclude noise
        '''
        H = df['com_ai-H']
        A = df['com_ai-A']
        W = A/H
        clust = hdbscan.HDBSCAN(min_cluster_size=100)
        clust.fit(np.transpose([W,H]))
        return df[clust.labels_==-1]

    def image_bg_subtract(img, radius=10, mode='valid'):
        '''
        Compute image background and subtract it and normalize intensities
        return normalized and background subtracted image
        '''
        # get local average of pixel intensity
        filt = skimage.morphology.disk(radius)
        filt = filt/np.sum(filt)
        bg = sp.signal.oaconvolve(img, filt, mode=mode)
        if mode=='valid':
            img = image_edge_crop(img,radius)
        # subtract local background
        fimg = img-bg
        # normalize the pixel intensity
        fimg = fimg/np.std(fimg)
        return fimg

    def image_edge_crop(img, r=10):
        return img[r:-r,r:-r]

    def image_local_std(img, radius=10):
        '''
        Get local z score
        '''
        # get local average of pixel intensity
        filt = skimage.morphology.disk(radius)
        filt = filt/np.sum(filt)
        # mean
        m1 = sp.signal.oaconvolve(img*img, filt, mode='same')
        # second moment
        m2 = sp.signal.oaconvolve(img, filt, mode='same')
        # local variance
        s = m2 - m1**2
        # normalize the pixel intensity locally
        fimg = (img-m1)/s
        # normalize values globally
        fimg = fimg/np.std(fimg)
        return fimg

    def image_get_edge(img, method='otsu', higher=False, erode=False, skeletonize=False):
        '''
        Compute droplet edge using brightfield pixel intensity
        '''
        if method=='kmeans':
            clst = sklearn.cluster.KMeans(n_clusters=2)
            d = img.reshape((img.size,1))
            clst.fit(d)
            L = clst.labels_.reshape(x.shape)
            idx = 0
            m = 0
            # edges are the darkest pixels
            for i in np.unique(L):
                v = np.mean(img[L==i])
                c = m > v
                if higher:
                    c = ~c
                if c:
                    idx = i
            edges = (L==idx)
        else:
            # edge via otsu threshold
            th = skimage.filters.threshold_otsu(img)
            edges = (img < th)
            if higher:
                edges = ~edges
        
        # do a few binary closing operations
        mask = edges > 0
        if erode:
            for i in range(0,100):
                mask = skimage.morphology.closing(mask > 0, selem=np.ones((3,3)))
        if skeletonize:
            mask = skimage.morphology.skeletonize(mask)
        return mask

    def image_get_labels(mask, img=None, properties=['label','area','perimeter','centroid']):
        '''
        Gets the labeled binary mask for the images
        mask = binary mask of droplets
        img = intensity values of pixels for region prop computation
        return labeled binary mask of droplets and region properties
        '''
        # apply bwlabel
        label, n_features = sp.ndimage.measurements.label(mask)
        # get some morphological properties
        if ('mean_intensity' in properties) and type(img)==None:
            print('Error!! No intensity image given.')
            return label, -1
        else:
            props = skimage.measure.regionprops_table(label, intensity_image=img, cache=True, properties=properties)
            return label, props

    def image_mask_overlay(img, mask, color):
        '''
        Convert image to color and overlay mask with given color
        return img array with pixel values for rgb in [0,1] range
        '''
        # convert image to 8bit color
        if len(img.shape)!=3:
            dx,dy = img.shape
            x = np.zeros((dx,dy,3))
            M = np.max(img)
            for i in range(0,3):
                x[:,:,i] = img/M
        else:
            x = img
        # apply mask of color to one of the channels
        c = matplotlib.colors.to_rgb(color)
        for i in range(0,3):
            x[mask,i] = c[i]
        return x

    def image_fluor_overlay(bf, rfu, colors, bf_weight=1):
        '''
        Add fluorescence intensity color to brightfield image
        bf = 2D array containing black white intensity values
        rfu = array of 2D arrays [rfu1, rfu2, rfu3, ...] containing rfu signal
        colors = array containing colors and weights such as [['green', 1.0], ['blue',0.5]]
        return img array with pixel values for rgb in [0,1] range
        '''
        # apply mask of color to one of the channels
        dx,dy = bf.shape
        img = np.zeros((dx,dy,3))
        # add each channel of colors
        for j in range(0,len(colors)):
            c = matplotlib.colors.to_rgb(colors[j][0])
            v = rfu[j]/np.max(rfu[j])
            for i in range(0,3):
                img[:,:,i] = img[:,:,i] + colors[j][1]*c[i]*v
        
        # add brightfield
        for i in range(0,3):
            img[:,:,i]+=bf_weight*bf/np.max(bf)
        # normalize each rgb from 0 to 1
        return img/np.max(img)

class FlowFrame:
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
        idx = np.random.permutation(len(df))[:self.params['subsamples']]
        return df.iloc[idx]

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

    def polygon_gate(self):
        '''
        Add
        '''

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
 
def main():
    parser = argparse.ArgumentParser(description='FCSlib: A python based toolkit for analyzing and processing flow cytometry and qPCR data')
    subparser = parser.add_subparsers(title='subcommands', dest='subcommand')
    # parse commands related to analysis
    aparser = subparser.add_parser('flow', help='suboptions for fcs')
    aparser.add_argument('-i', dest='infile', nargs='+', type=str, help='input files')
    aparser.add_argument('-n', dest='n_samples', type=int, help='number of samples to keep')
    aparser.add_argument('-trim', dest='trim', action='store_true', help='trim data in fcs file')
    # parse arguments
    args = parser.parse_args()
    if args.subcommand=='flow':
        
        df.to_csv(args.ofile, index=False)

if __name__ == "__main__":
    main()
