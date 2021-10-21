# Copyright (c) 2020 brainlife.io
#
# This file is a MNE python-based brainlife.io App
#
# Author: Guiomar Niso
# Indiana University

# Required libraries
# pip install mne-bids coloredlogs tqdm pandas scikit-learn json_tricks fire

# set up environment
#import mne-study-template
import os
import json
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Current path
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Load brainlife config.json
with open(__location__+'/config.json') as config_json:
    config = json.load(config_json)

# == LOAD DATA ==
# FIF
fname = config['fif']
raw = mne.io.read_raw_fif(fname)

# CTF
# fname = config['ctf']
# raw = mne.io.read_raw_ctf(fname)

# == GET CONFIG VALUES ==

fmin = config['fmin']
fmax = config['fmax']
average = config['average']

# Advanced parameters
tmin = config['tmin'] if config['tmin'] else None
tmax = config['tmax'] if config['tmax'] else None
n_fft = config['n_fft']
n_overlap = config['n_overlap']
n_per_seg = config['n_per_seg'] if config['n_per_seg'] else None
window = config['window']
reject_by_annotation = config['reject_by_annotation']
proj = config['proj']
n_jobs = 1

if config['picks']:
    #If its a list starting with square braket, convert to list of strings
    if config['picks'].find("[") == 0:
        picks = config['picks'].replace('[','').replace(']','').split(", ")
    else:
        picks = config['picks']   
else: 
    picks=None



# psd_welch.shape: Nchannels x Nfreqs


# Types of channels in the data
# e.g. ['ecg', 'eog', 'grad', 'mag', 'eeg','misc', 'stim']
ch_types=np.unique(raw.get_channel_types())


# == COMPUTE PSD ==
if picks==None:

    # FIGURE 1
    # Number of subplots
    bb=0
    for i in ['grad','mag','eeg']: 
        if i in ch_types: bb=bb+1
    plt.figure(1)
    fig, axs = plt.subplots(bb)
    fig.subplots_adjust(hspace =.5, wspace=.2)

    aa=0

    if 'eeg' in ch_types:
        raw_eeg = raw.copy().pick('eeg')
        ch_eeg = raw_eeg.ch_names
        '''psd_welch_eeg, freqs_eeg = mne.time_frequency.psd_multitaper(raw_eeg, 
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                            bandwidth=bandwidth, adaptive=adaptive, low_bias=low_bias, normalization=normalization, 
                            picks='eeg', proj=proj, n_jobs=n_jobs, verbose=None)'''
        psd_welch_eeg, freqs_eeg = mne.time_frequency.psd_welch(raw_eeg, 
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                            n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, window=window, 
                            reject_by_annotation=reject_by_annotation, average=average, 
                            picks='eeg', proj=proj,n_jobs=1, verbose=None)
        # Convert power to dB scale: V^2/hz -> uV^2/Hz
        psd_welch_eeg = 10*(np.log10(psd_welch_eeg.mean(axis=0)*1e6**2))

        # Save to CSV file (could be also TSV)
        df_psd = pd.DataFrame(psd_welch_eeg, index=ch_eeg, columns=freqs_eeg)
        df_psd.index.name='channels'
        df_psd.to_csv(os.path.join('out_dir','psd_eeg.csv')) #, sep = '\t', index=False)

        # Figure
        axs[aa].plot(freqs_eeg, psd_welch_eeg.transpose(), zorder=1) 
        axs[aa].set_xlim(xmin=0, xmax=max(freqs_eeg))
        axs[aa].set_xlabel('Frequency (Hz)')
        axs[aa].set_ylabel('uV^2/Hz [dB]')
        axs[aa].set_title('PSD - EEG')
        axs[aa].grid(linestyle=':')
        aa=aa+1
        
    if 'grad' in ch_types:
        raw_grad = raw.copy().pick('grad')
        ch_grad = raw_grad.ch_names
        '''psd_welch_grad, freqs_grad = mne.time_frequency.psd_multitaper(raw_grad, 
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                            bandwidth=bandwidth, adaptive=adaptive, low_bias=low_bias, normalization=normalization, 
                            picks='grad', proj=proj, n_jobs=n_jobs, verbose=None)'''
        psd_welch_grad, freqs_grad = mne.time_frequency.psd_welch(raw_grad, 
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                            n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, window=window, 
                            reject_by_annotation=reject_by_annotation, average=average, 
                            picks='grad', proj=proj,n_jobs=n_jobs, verbose=None)
        # Convert power to dB scale: (T/m)^2/hz -> (fT/cm)^2/Hz
        psd_welch_grad = 10*(np.log10(psd_welch_grad.mean(axis=0)*1e13**2))

        # Save to CSV file (could be also TSV)
        df_psd = pd.DataFrame(psd_welch_grad, index=ch_grad, columns=freqs_grad)
        df_psd.index.name='channels'
        df_psd.to_csv(os.path.join('out_dir','psd_grad.csv')) #, sep = '\t', index=False)

        # Figure
        axs[aa].plot(freqs_grad, psd_welch_grad.transpose(), zorder=1) 
        axs[aa].set_xlim(xmin=0, xmax=max(freqs_grad))
        axs[aa].set_xlabel('Frequency (Hz)')
        axs[aa].set_ylabel('(fT/cm)^2/Hz [dB]')
        axs[aa].set_title('PSD - Gradieometers')
        axs[aa].grid(linestyle=':')
        aa=aa+1

    if 'mag' in ch_types:
        raw_mag = raw.copy().pick('mag')
        ch_mag = raw_mag.ch_names
        '''psd_welch_grad, freqs_grad = mne.time_frequency.psd_multitaper(raw_grad, 
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                            bandwidth=bandwidth, adaptive=adaptive, low_bias=low_bias, normalization=normalization, 
                            picks='grad', proj=proj, n_jobs=n_jobs, verbose=None)'''
        psd_welch_mag, freqs_mag = mne.time_frequency.psd_welch(raw_mag, 
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                            n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, window=window, 
                            reject_by_annotation=reject_by_annotation, average=average, 
                            picks='mag', proj=proj,n_jobs=n_jobs, verbose=None)        
        # Convert power to dB scale: T^2/hz -> fT^2/Hz
        psd_welch_mag = 10*(np.log10(psd_welch_mag.mean(axis=0)*1e15**2))

        # Save to CSV file (could be also TSV)
        df_psd = pd.DataFrame(psd_welch_mag, index=ch_mag, columns=freqs_mag)
        df_psd.index.name='channels'
        df_psd.to_csv(os.path.join('out_dir','psd_mag.csv')) #, sep = '\t', index=False)

        # Figure
        axs[aa].plot(freqs_mag, psd_welch_mag.transpose(), zorder=1) 
        axs[aa].set_xlim(xmin=0, xmax=max(freqs_mag))
        axs[aa].set_xlabel('Frequency (Hz)')
        axs[aa].set_ylabel('fT^2/Hz [dB]')
        axs[aa].set_title('PSD - Magnetometers')
        axs[aa].grid(linestyle=':')
        aa=aa+1

    # Save fig
    plt.savefig(os.path.join('out_figs','psd_computed.png'))

# ==== PLOT FIGURES ====
# FIGURE 2
# Plot MNE PSD
plt.figure(2)
raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, 
            proj=proj, n_fft=n_fft, n_overlap=n_overlap, window=window, 
            ax=None, color='black', xscale='linear', area_mode='std', area_alpha=0.33, 
            dB=True, estimate='auto', show=True, n_jobs=n_jobs, average=False, 
            line_alpha=None, spatial_colors=True, sphere=None, verbose=None)
# Save fig
plt.savefig(os.path.join('out_figs','psd_mne.png'))

