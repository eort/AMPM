"""
Helper functions to read and write various file formats.
"""
import os
import pickle

def savePickle(container,file_path):

	outfile = open(file_path,'wb')        
	pickle.dump(container, outfile)
	outfile.close()