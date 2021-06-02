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

# Current path
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Populate mne_config.py file with brainlife config.json
with open(__location__+'/config.json') as config_json:
    config = json.load(config_json)


fname = config['fif']

fmin = config['fmin']
fmax=config['fmax'] if config['fmax'] else 'Inf'
average = config['average']

# Advanced parameters
picks = config['picks']
tmin=config['tmin'] if config['tmin'] else 'None'
tmax=config['tmax'] if config['tmax'] else 'None'
n_fft = config['n_fft']
n_overlap = config['n_overlap']
n_per_seg=config['n_per_seg'] if config['n_per_seg'] else 'None'
window = config['window']
proj = config['proj']
reject_by_annotation = config['reject_by_annotation']
#n_jobs = config['n_jobs']
#verbose = config['verbose']


raw = mne.io.read_raw_fif(fname)

psds_welch, freqs = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, 
                             n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, window=window, picks=picks, proj=proj,
                             reject_by_annotation=reject_by_annotation, average=average, n_jobs=1, verbose=None)

# Convert power to dB scale.
psds_welch = 10 * np.log10(psds_welch)

# save the first seconds of MEG data in FIF file
np.save(os.path.join('out_dir','psd_welch'), psds_welch)
np.save(os.path.join('out_dir2','freqs'), freqs)

