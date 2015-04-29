#spatial smoothing helper file
import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt


def SmoothArray(array, window_size = 27):        
    sigma = 0.1
    mu  = 0.5
    window_size = 27
    kernel = np.exp(-(np.linspace(0,1,window_size,endpoint = True) - mu) **2 / (sigma**2*2))
    convolution = np.convolve(array,kernel, 'same')
    convolution /= max(abs(convolution))
    return convolution

def FourierTransform(InputArray):
    '''takes time signal as input and transforms it into power spectrum, to show at which frequency perceptual blocks oscilate'''
    sp = np.fft.fft(np.array(InputArray))
    freq = np.fft.fftfreq(np.array(InputArray).shape[-1])
    spAbs = np.absolute(sp) **2
    whichPlot = (freq > 0 )*(freq < 0.03)
    return (freq[whichPlot],spAbs[whichPlot])

def CalculateStability(ResponseArray,TimeArray, cutoff = 0):
    '''Input Responsearray should contain the smoothed responses for each condition per subject.
        TimeArray introduced the time when each response was given.
        Cutoff represents the value which has to be reached by the responses in order get counted as such. (Lost dichotomy by smoothing)
        
        Returns for each condition a list containing the duration of each stabilized percept and a list with an index for the number of which perceptual block it belongs to

        '''
    bias_list = []
    level = 0
    resp = 0
    start_percept =  0 
    percep_block_index = 0
    prelevel = 0
    for trial in xrange(len(ResponseArray)): 
        if trial == len(ResponseArray)-1:
            level = 2
        elif  ResponseArray[trial] > cutoff:
            level = 1
        elif ResponseArray[trial]< (-1 * cutoff):
            level = -1
        if level != prelevel:
            percep_block_index += 1
            resp = TimeArray[trial] - start_percept
            start_percept = TimeArray[trial]
            bias_list.append(resp)
        prelevel = level
    return bias_list


def expConvolve(arrays, window_size=27):
    '''
    Convolve input array with 2 exponential functions and normalizes them
    Returns two arrays: 1) with positive weights, 2) negative weights
    '''    

    
    array1 = np.array(arrays > 0, dtype = int)
    array2 = np.array(arrays < 0, dtype = int)

    kernel = expon.pdf(np.linspace(0,3, window_size), scale = 0.5)
    kernel /= kernel.sum()
    convolution1 = np.convolve(array1,kernel, 'same')
    convolution2 = np.convolve(array2,kernel, 'same')

    return convolution1, convolution2

def posInPercept(responses, response_times):
    '''
    converts the percept.block index of a timepoint in a session into its relative position (begin,middle,end) of one percep.block
    returns a list with 1,2,3,4s corresponding to begin (25%), 2x middle (50%), end(25%) of trials and corresponding response times
    '''
    pos = 0
    start = [0]
    stop =[] 
    quartile_session = []
    dur_quartile = []
    pos_in_block = []
    quartile_index = 1

    for i in xrange(len(responses)):
        if pos*responses[i]< 0:
            stop.append(response_times[i])
            start.append(response_times[i])
        if i == len(responses)-1:
            stop.append(response_times[i])
        if  responses[i]>0:
            pos = 1
        elif responses[i]<0:
            pos = -1
    
    for switchi in xrange(len(start)):
        dur_quartile.append((stop[switchi]-start[switchi])/4)

    for quarti in xrange(len(dur_quartile)):
        for i in xrange(1,5):
            quartile_session.append(start[quarti]+i*dur_quartile[quarti])
    
    for time in response_times:
        if time > quartile_session[quartile_index-1]:
            quartile_index += 1
        pos_in_block.append([quartile_index%4, time])

    return pos_in_block
