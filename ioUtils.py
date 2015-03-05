"""
Helper functions to read and write various file formats.
"""
import os
import pickle

def writePickle(container,file_path,mode='wb'):
	'''
	Writes container to a file, specified by file_path.Mode is write-only.
	'''
	if mode not in ['w','wb','a','r+']:
		raise IOError("writePickle write-only! {0} mode not valid".format(mode))
	outfile = open(file_path,mode)        
	pickle.dump(container, outfile)
	outfile.close()

def readPickle(filename,mode='rb'):
    '''
    read pickles and returns content. Mode is read-only.
    '''
    if mode not in ['r','rb','r+']:
        raise IOError("readPickle read-only! {0} mode not valid".format(mode))
    with open(filename,mode) as f:
         a = pickle.load(f)
    f.close()
    return a