import RawClass_dev as cl
import mne
import os

#this script is only supposed to provide a raw file that is as clean as possible
def main(subj):
    s = cl.Subject(subj)
    s.raw.rereference(['EXG5','EXG6'])
    #add EOG labels to respective channels
    s.raw.renameChannel(['EXG1','EXG2','EXG3','EXG4'],['VEOG1','VEOG2','HEOG1','HEOG2'] )
    #if necessary swap replaced electrodes 
    if s.replacements != [()]:
        s.raw.replaceElecs(s.replacements)

    s.raw.info['bads'] = s.raw.info['bads'] + s.bad_channels + ['EXG7','EXG8'] 
    good_eeg = mne.pick_types(s.raw.info, meg = False, eeg = True, stim = False, eog = False, exclude = 'bads')

    # save pure raw
    if not cl.createDir(s.data_path, s.ID+'_basic_raw.fif', 'check' ):
        s.raw.save(os.path.join(s.data_path,s.ID+'_basic_raw.fif'))

    #filter
    s.raw.filter(l_freq = 1, h_freq = 100)

    #s.raw.plot(start = 24)

    s.raw.runICA(picks= good_eeg, save = os.path.join(s.data_path,'ica_solution.fif')
    
    # save postICA raw
    if not cl.createDir(s.data_path, s.ID+'_postICA_raw.fif', 'check' ):
        s.raw.save(os.path.join(s.data_path,s.ID+'_postICA_raw.fif'))

    #G,H, cosdist = s.raw.LaplacianTransform(picks = good_eeg, m=4,legendre_order = 20,lamda = 10e-5)

if __name__ == '__main__':
    #main('01TK.bdf')
    files = os.listdir('/home/shared/AM_EEG/data/bdf')
    Parallel(n_jobs = 12, verbose = 9)(delayed(main)(subj) for subj in files if subj[0] != '.')