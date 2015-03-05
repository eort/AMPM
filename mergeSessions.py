#!/usr/bin/env python
"""
Because every session produced a separate pickle file with all behavioral results, trial parameters, etc., 
it'll come in handy to unify these sessions per subject. Later processing steps will require less looping and
annoyance.
In detail, this script reads a pickle, copies all content into a new container of same properties and
appends other sessions contents. After last file appended, new pickle file is closed and saved. 
"""

import pickle
import os
import ioUtils as io
from IPython import embed as shell

# provide path to raw pickles
file_location = '../data/raw_pickles/'
file_list = sorted(os.listdir(file_location))

# prepare data container and helper variables
current_subject = None
subject_container = {'parameterArray':[],'eventArray':[],'sessionArray':[]} 

#loop over session pickle files,add them to corresponding subject pickle and save file to disc
for idx,session_file in enumerate(file_list):
    if session_file[0]=='.':
        continue
    a = io.readPickle(os.path.join(file_location,session_file))
        
    if (session_file[:4]!=current_subject and current_subject != None):
        subject_file_name = os.path.join('../data/pickles/',current_subject+'.pickle')
        io.writePickle(subject_container,subject_file_name)
        print 'Finished processing subject {0}'.format(current_subject)
        print 'and start with processing subject {}'.format(session_file[:4])
        subject_container = {'parameterArray':[],'eventArray':[],'sessionArray':[]}
    
    for key in subject_container.keys():
        subject_container[key] += a[key]

    if idx == len(file_list)-1:
        subject_file_name = os.path.join('../data/pickles/',session_file[:4]+'.pickle')
        io.writePickle(subject_container,subject_file_name)
    
    current_subject = session_file[:4]

