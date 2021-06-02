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
fmax = config['fmax']
#tmin = config['tmin']
#tmax = config['tmax']
#n_fft = config['n_fft']
#n_overlap = config['n_overlap']


raw = mne.io.read_raw_fif(fname)

psds_welch, freqs = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, tmin=None, tmax=None, 
                             n_fft=256, n_overlap=0, n_per_seg=None, picks=None, proj=False, n_jobs=1, 
                             reject_by_annotation=True, average='mean', window='hamming', verbose=None)

# Convert power to dB scale.
psds_welch = 10 * np.log10(psds_welch)

# save the first seconds of MEG data in FIF file
np.save(os.path.join('out_dir','psd_welch'), psds_welch)
np.save(os.path.join('out_dir2','freqs'), freqs)

