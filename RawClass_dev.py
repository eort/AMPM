#!/usr/bin/env python

#trying to do analysis object-oriented-wise
import mne
import numpy as np 
from joblib import Parallel,delayed
import logging
import os
import numpy.polynomial.legendre as lg
import matplotlib.pyplot as plt
from IPython import embed as shell
import scipy.io as sio

def createDir(path, directory,task='create'):
    '''
    checks whether a directory in specified path is already existing. 
    If not directory is created if task says so, otherwise it is only checked
    '''

    folder = os.path.join(path,directory)
    if not os.path.exists(folder):
        if task =='create':
            os.makedirs(folder)
        elif task == 'check':
            return False
    return os.path.realpath(folder)  


def Legendre(n,x):
    '''
    According to Pn(x), where n is the nth polynomial from 0,1,..., order; see:http://people.sc.fsu.edu/~jburkardt/m_src/legendre_polynomial/legendre_polynomial.html
    x is data
    returns the respective polynomial
    For big n (>30), recursive definition takes too much time and resources
    '''
    if n == 0:
        return  np.ones(np.shape(x))
    elif n==1:
        return  x
    elif n>1:
        return  (2.0*n-1)/n * x * Legendre(n-1,x) - (n-1.0)/n * Legendre(n-2,x)

class Subject(object):
    def __init__(self, subject_file):

        self.ID = subject_file[:4]
        self.bad_channels = self.retrieveBads(self.ID)[1] # marked channels during recording
        self.replacements = self.retrieveBads(self.ID)[0] #broke electrode replacements during recording
        self.data_path = createDir('../data/fif/',self.ID) # creates a data path to fif/subject, where raw and events are saved to
        #init logger
        logging.basicConfig(filename='/home/shared/AM_EEG/results/logs/'+self.ID+'.log',level=logging.DEBUG, format='%(asctime)s %(message)s' )
        self.raw = self.readBDF(subject_file,eog = ['EXG1','EXG2','EXG3','EXG4'],misc = ['EXG5','EXG6','EXG7','EXG8']\
                ,montage="/home/shared/AM_EEG/data/biosemi128.hpts", preload=True) # create rawBDF instance
            
    def readBDF(self,bdf_file, eog=None,misc=None, stim_channel=-1, annot=None, annotmap=None,  montage=None, preload=False, verbose=None):
        '''
        Does pretty much the same as MNE's read_raw_edf. Difference is that instead of MNE's rawEDF, a custom-built rawBDF instance is called. 
        In doing so, more methods can be created that work on the raw instance. 
        '''
        bdf_path = '../data/bdf/'
        bdfname = os.path.join(bdf_path, bdf_file)
        return RawBDF(input_fname=bdfname, eog = eog, misc = misc,stim_channel=stim_channel, annot=annot, annotmap=annotmap, montage=montage, preload=preload,verbose=verbose)

    def retrieveBads(self,subID):
        '''
        replacers is a dictionary containing [replaced channels, bad channels] per subject
        Calling this function will return the entire list for a chosen subject. So, keep in mind to index what you need
        replaced channels saved in lists of tuples (replacing, replaced)
        '''

        replacers= {
            '01TK':[[('EXG8','C29'),('EXG7' , 'D29')], ['D12', 'D26']],
            '02BB':[[('EXG8','C29'),('EXG7' , 'D29')], ['D12','D26']],
            '03SM':[[('EXG8','C29'),('EXG7' , 'D29')], ['D12','D26']],
            '04RB':[[('EXG8','C29'),('EXG7' , 'D29')], ['D12','D26','C15','C29']],
            '05AB':[[('EXG8','C29'),('EXG7' , 'D29')], ['D12','D26']],
            '06KU':[[('EXG8','C29'),('EXG7' , 'D29')], ['D12','D26']],
            '07GW':[[('EXG8','C29')], ['B15']],
            '08EG':[[('EXG8','C29')], []],
            '09GG':[[('EXG8','C29')], []],
            '10SB':[[('EXG8','C29')], []],
            '11CK':[[('EXG8','C29')], ['D4','C29']],
            '12WW':[[('EXG8','C29')], ['A14','A27','A15','A26']],
            '13MZ':[[('EXG8','C29')], []],
            '14AC':[[('EXG8','C29'),('EXG7' , 'D2')], []],
            '15LA':[[('EXG8','C29'),('EXG7' , 'C30')], ['D9']],
            '16EB':[[()], ['D12','D26']],
            '17SK':[[('EXG7' , 'D4')], ['C10','C30','C31','C32']],
            '18CK':[[('EXG7' , 'D4')], []],
            '19LA':[[('EXG7' , 'D4')], []],
            '20KD':[[('EXG7' , 'D4')], []],
            '21LS':[[('EXG7' , 'D4')], ['B4', 'D23']]}
        return replacers[subID]

class RawBDF(mne.io.edf.edf.RawEDF):
    '''
    This class is a child originating from  MNE built-in RawEDF. All its features are loaded, but more methods can be written for it. 
    This instance is also an attribute of the Subject. 
    '''
    def __init__(self,input_fname,eog=None, misc=None , stim_channel=-1, annot=None, annotmap=None, montage=None, preload=False, verbose=None):
        super(RawBDF,self).__init__(input_fname, eog=eog, misc=misc, stim_channel=stim_channel, annot=annot, annotmap=annotmap, montage=montage, preload=preload,verbose=verbose)
        logging.info('rawBDF instance was created for subject {0}'.format(input_fname[-8:]))

    def rereference(self,refchans=None):
        print 'Re-referencing data to', refchans
        logging.info('Data was rereferenced to channels '+ ', '.join(refchans))
        (self, ref_data) = mne.io.set_eeg_reference(self, refchans, copy=False)
        for chI, channel in enumerate(refchans):
            self.info['chs'][self.ch_names.index(channel)]['kind'] = 502
        self.info['bads'] += refchans 

    def defineEOG(self,channels = list(xrange(128,132))):
        '''
        Changes the channel type of the passed channel list to EOG. 
        Assumes a recording system of 4 EOG electrodes
        Default settings are : EXG1,2,3,4 corresponding to VEOG1,2,HEOG1,2
        '''
        try:
            if len(channels) != 4:
                logging.warning('Usually there are exactly 4 EOG channels, you defined {0}'.format(len(channels)))
            elif type(channels)  not in (tuple, list):
                raise TypeError('Channels argument is supposed to be a list or tuple, not {0}'.format(type(channels)))

        except TypeError as e:
            print e,'Input has to be a list/tuple'
            logging.error(e)

        else:
            chan_translater = {0:'VEOG1',1:'VEOG2',2:'HEOG1',3:'HEOG2'}
            for chI, channel in enumerate(channels):
                self.info['chs'][channel]['kind'] = 202
                self.info['chs'][channel]['ch_name'] = chan_translater[chI]

            logging.info('EOG channels ({0},{1},{2},{3}) successfully defined'.format(channels[0],channels[1],channels[2],channels[3]))

        print 'EOG channels ({0},{1},{2},{3}) successfully defined as ({4},{5},{6},{7}) '.format(\
            channels[0],channels[1],channels[2],channels[3], chan_translater.values()[0],chan_translater.values()[1],chan_translater.values()[2],chan_translater.values()[3])

    def redefineType(self, channels, new_type):
        '''
        Change channel type of passed channels to new_type. Arguments should be passed as lists of strings
        '''
        if type(channels) not in  [list,tuple]:
            raise TypeError('Channels argument is supposed to be a list or tuple, not {0}. See doc'.format(type(channels)))
        if type(new_type) not in  [list,tuple]:
            raise TypeError('new_type argument is supposed to be a list or tuple, not {0}. See doc'.format(type(channels)))        

        for chI, channel in enumerate(channels):
            self.info['chs'][self.ch_names.index(channel)]['kind'] = new_type[chI]
        logging.info('Channels renamed')

    def renameChannel(self,channels, new_name):
        '''
        Change channel label to new_name
        '''
        if type(channels) not in  [list,tuple]:
            raise TypeError('Channels argument is supposed to be a list or tuple, not {0}. See doc'.format(type(channels)))
        if type(new_name) not in  [list,tuple]:
            raise TypeError('new_name argument is supposed to be a list or tuple, not {0}. See doc'.format(type(channels)))      

        for chI, channel in enumerate(channels):
            self.ch_names[self.ch_names.index(channel)] = new_name[chI]
            self.info['chs'][self.ch_names.index(channel)]['ch_name'] = new_name[chI]



    def replaceElecs(self, elec_pairs, mode = 'swap'):
        '''
        This function is supposed to swap the metadata of 2 (or more) electrodes. 
        Use the names of the electrodes to pass them (e.g. 'A1','EXG8',...). 
        Pass them in a list/tuple. Multiple operations in nested lists (e.g. [[A1,A2],[B1,B2]])

        mode: 'swap' means swapping everything between 2 electrodes
              'replace' means the identifying information from the elec1 are kept, 
              but anything else is used from elec2. After that elec2 is not used anymore
              and regarded as bad electrode.
        Returns the updated self instance
        '''
        # translate electrode names into indices 
        electrode_idx= {self.ch_names.index(elec1):self.ch_names.index(elec2) for elec1,elec2 in elec_pairs}
        # create placeholder container, storing the content of the replacing electrodes
        if mode =='swap':

            for elec1,elec2 in electrode_idx.items():
                foo_content = {k:v for k,v in self.info['chs'][elec1].items()}
                self.info['chs'][elec1] = {k:v for k,v in self.info['chs'][elec2].items()}
                self.info['chs'][elec2] = {k:v for k,v in foo_content.items()}
                logging.info('Electrode {0} and electrode {1} were successfully swapped.'.format(self.ch_names[elec1],self.ch_names[elec2]))
                print 'Electrode {0} and electrode {1} were successfully swapped.'.format(self.ch_names[elec1],self.ch_names[elec2])

            for nameI,name in enumerate(self.ch_names):
                self.ch_names[nameI] = self.info['chs'][nameI]['ch_name']

        elif mode == 'replace':
            try:
                raise NotImplementedError('This mode is not implemented yet (and probably won\'t be soon')
            except:
                print 'This mode is not implemented yet'


    def LaplacianTransform(self,picks = None, m = 4, legendre_order = 20, lamda = 10e-5, test = False):
        '''
        NOT WORKING YET
        Spatial smoothing on data, to increase spatial resolution of data. Method used: Spherical Spines (Perrin, 1987)

        #m related to smoothness of result, has to be posInt, default = 3
        #legendre_order of legendre polynomial: Spatial bandpass filter, not too high, not too low, default = 40
        #n_elec: number of electrodes, default = 128
        #lamda: another smoothing factor to smooth G matrix, default = 10e-5
        #picks: pass indices of good eeg channel (array like)
        what to return?
        '''
        logging.info('Laplacian Transformation is applied to the data.')
        #create array of order for the legendre polynomial
        # n = np.arange(1,order+1)

        #decide how many electrodes are used, if not specified, all are used that can be found in data
        if picks == None:
            picks = []
            for i in xrange(len(self.ch_names)):
                if self.info['chs'][i]['ch_name'][0] in ['A','B','C','D']:
                       picks.append(i)
        
        # initialize variables
        n_elec = picks.shape[0]
        x = np.zeros((n_elec))
        y = np.zeros((n_elec))
        z = np.zeros((n_elec))
        cosdist = np.zeros((n_elec, n_elec))
        G_empty = np.zeros((legendre_order,cosdist.shape[0], cosdist.shape[1]))
        H_empty = np.zeros((legendre_order,cosdist.shape[0], cosdist.shape[1]))
        data  = self._data[picks,:]

        #retrieve electrode location
        X = np.array( [elec['loc'][0] for elec in self.info['chs'] if elec['ch_name'][0] in ['A','B','C','D']])
        Y = np.array( [elec['loc'][1] for elec in self.info['chs'] if elec['ch_name'][0] in ['A','B','C','D']])
        Z = np.array( [elec['loc'][2] for elec in self.info['chs'] if elec['ch_name'][0] in ['A','B','C','D']])


        #consider only data of picks
        for clean_idx,raw_idx in enumerate(picks):
           x[clean_idx] = X[clean_idx]
           y[clean_idx] = Y[clean_idx]
           z[clean_idx] = Z[clean_idx]

        maxrad = np.linalg.norm(np.vstack([x,y,z]),axis = 0).max()    
        x,y,z = x/maxrad, y/maxrad, z/maxrad
        # 1) get cosdist (scaled distances between each pair of electrodes)
        
        x = sio.loadmat('X.mat')['X'][0].T
        z = sio.loadmat('Z.mat')['Z'][0].T
        y = sio.loadmat('Y.mat')['Y'][0].T
        maxrad = np.linalg.norm(np.vstack([x,y,z]),axis = 0).max()    
        x,y,z = x/maxrad, y/maxrad, z/maxrad
        

        #n_elec = 64
        cosdist = np.zeros((n_elec, n_elec))
        
        for i in xrange(n_elec):
            for j in range(i+1,n_elec):
                cosdist[i,j] = 1 - (((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2)/2)
        cosdist = cosdist + cosdist.T + np.eye(n_elec)

        legpoly = np.zeros((legendre_order,n_elec,n_elec));
        for ni in range(1,legendre_order+1):
            legpoly[ni-1,:,:] = Legendre(ni,cosdist)

        #twoN1_g = 2 * np.arange(1,legendre_order+1) + 1
        #twoN1_h = -2 * (np.arange(1,legendre_order+1) + 1)
        twoN1= 2 * np.arange(1,legendre_order+1) + 1
        gdenom = (np.arange(1,legendre_order+1) * np.arange(2,legendre_order+2)) ** m
        hdenom = (np.arange(1,legendre_order+1) * np.arange(2,legendre_order+2)) ** (m-1)

        G, H = np.zeros((n_elec,n_elec)), np.zeros((n_elec,n_elec))

        for i in range(n_elec):
            for j in range(n_elec):
                g=0; h=0;
                for ni in range(legendre_order):
                    g = g + (twoN1[ni]*legpoly[ni,i,j]) / gdenom[ni]
                    h = h - (twoN1[ni]*legpoly[ni,i,j]) / hdenom[ni]
 
                G[i,j] =  g/(4*np.pi)
                H[i,j] =  -1* h/(4*np.pi)    
        
        # 2) compute G and H matrix
        G = G + G.T; H = H + H.T
        G = G - np.eye(n_elec) * G[0,0]/2
        H = H - np.eye(n_elec) * H[0,0]/2
        

        # illustration of G and H
        #plt.figure(1)
        #plt.imshow(G)

        #3) compute Gs
        Gs = G + np.eye(n_elec) * lamda # only upper triangle ?

        logging.info('Start trying to find matrix inverse...')
        print 'Start trying to find matrix inverse...' 
        #get pseudo inverses
        inv_data = np.linalg.pinv(data)
        inv_Gs = np.sum(np.linalg.inv(Gs), axis = 0)

        logging.info('...and done')
        print '...and done' 

        #4) d matrix
        d = np.dot(inv_data,Gs)
        #5) C matrix
        C = d - (np.matrix((np.sum(d,axis = 1) / np.sum(inv_Gs))).T * np.matrix(inv_Gs))


        # 6) compute laplacian 
        lap = (C * np.matrix(H)).T

        #feed back into raw instance
        self._data[picks,:] = lap
        logging.info('Laplacian Transformation finished')
        return G,H, cosdist
   
    def runICA(self, n_components = 50, method = 'fastica', picks = None, decim = 3, enable_plots = False,save= None):
        """
        Runs an independet component analysis, type fastica (see MNE doc), enable_plots not yet implemented. supposed
        to allow/forbid plotting the results. At a later stage saving these plots should be implemented as well
        save to be specified as desired path where ICA solution should be stored
        """
        print 'start processing ICA'
        ica = mne.preprocessing.ICA(n_components=n_components, method='fastica')
        ica.fit(self, picks=picks, decim=3)
        n_max_eog = 1
        for eog_channel in ['VEOG1','VEOG2','HEOG1','HEOG2']:
            eog_inds, scores = ica.find_bads_eog(self, eog_channel)
            show_picks = np.abs(scores).argsort()[::-1][:5]
            #ica.plot_sources(self, show_picks, exclude=eog_inds, title='eog')
            ica.exclude += eog_inds[:n_max_eog]
            logging.info('Component {0} excluded with score of {1}'.format(eog_inds[:n_max_eog],scores[:n_max_eog]))
            print 'Component {0} excluded with score of {1}'.format(eog_inds[:n_max_eog],scores[:n_max_eog])
        ica.apply(self, copy=False)
        if save != None:
            ica.save(save)

