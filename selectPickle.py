#!/usr/bin/env python
"""
Unifies all relevant information over behavioral performance of subjects into one list of dicts
saves results in a new pickle.
"""


import pickle
import os
import ioUtils as io
from IPython import embed as shell


# load data
file_location = '../data/pickles/'
file_list = sorted(os.listdir(file_location))

for subI,subj in enumerate(file_list):
    if subj[0]=='.' or subj[0]=='d':
        continue
    print 'processing subject {0}'.format(subj[:4])
    a = io.readPickle(os.path.join(file_location,subj))
    string_events = [[e.split(' ') for e in tevs if type(e) == str] for tevs in a['eventArray']]
    phase_events = [[[int(ee[1]),int(ee[3]),float(ee[6])] for ee in tevs if ee[2] == 'phase'and len(ee) == 7] for tevs in string_events]
    key_events = [[[int(ee[1]),ee[3],float(ee[5])] for ee in tevs if ee[2] == 'event' and len(ee) == 6] for tevs in string_events]

    behav_data = []
    for idx,event in enumerate(string_events):
        if len(key_events[idx]) != 1:
            continue
        behav_data.append({\
            'trial_no' : idx,\
            'block_no' : int(a['parameterArray'][idx]['block_no']),\
            'trial_in_block' : a['parameterArray'][idx]['trial_no'],\
            'blank_time' : a['parameterArray'][idx]['stim_off_time'],\
            'trial_type' : a['parameterArray'][idx]['trial_type'],\
            'response' : a['parameterArray'][idx]['answer'],\
            'stim_onset' : phase_events[idx][0][-1],\
            'stim_offset' : phase_events[idx][1][-1],\
            'keypress_time' : key_events[idx][0][-1]\
        })
    directory = 'data_pickles'
    folder = os.path.join(file_location,directory)
    if not os.path.exists(folder):
            os.makedirs(folder)
    print 'Finished subject {0}'.format(subj[:4])            
    io.writePickle(behav_data,os.path.join(folder,subj[:4]+'_data.pickle'))

