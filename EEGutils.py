#!/usr/bin/env python

'''script to analyze the data'''

import os 
import numpy as np
import mne
import pickle
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import joblib
import anlffr.helper.biosemi2mne
import subprocess
import pdb
from scipy.stats.mstats import mquantiles


def Biosemi2MNE(bdf, no_chans = 136, refchans=['EXG5','EXG6'],hptsname = "/home/shared/AM_EEG/data/biosemi128.hpts"):
	'''reads data with helper function (using read_raw_edf) and rereferencing it to mastoids (EXG5/EXG6)
		returns: rawBDF file and event file'''

	bdf_path = '/home/shared/AM_EEG/data/bdf/'
	bdf += '.bdf'
	bdfname = os.path.join(bdf_path, bdf)
	raw , events = anlffr.helper.biosemi2mne.importbdf(bdfname = bdfname, nchans=no_chans, refchans=refchans, hptsname=hptsname)
	return raw, events


def retrieveBads(subID):
	'''
	enables importing of bad channels and whether and how channels might be replaced
	'''

	replacers= {
        '01TK':[dict(EXG8='C29',EXG7 = 'D29'), ['D12', 'D26']],
        '02BB':[dict(EXG8='C29',EXG7 = 'D29'), ['D12','D26']],
        '03SM':[dict(EXG8='C29',EXG7 = 'D29'), ['D12','D26']],
        '04RB':[dict(EXG8='C29',EXG7 = 'D29'), ['D12','D26','C15','C29']],
        '05AB':[dict(EXG8='C29',EXG7 = 'D29'), ['D12','D26']],
        '06KU':[dict(EXG8='C29',EXG7 = 'D29'), ['D12','D26']],
        '07GW':[dict(EXG8='C29'), ['B15']],
        '08EG':[dict(EXG8='C29'), []],
        '09GG':[dict(EXG8='C29'), []],
        '10SB':[dict(EXG8='C29'), []],
        '11CK':[dict(EXG8='C29'), ['D4','C29']],
        '12WW':[dict(EXG8='C29'), ['A14','A27','A15','A26']],
        '13MZ':[dict(EXG8='C29'), []],
        '14AC':[dict(EXG8='C29',EXG7 = 'D2'), []],
        '15LA':[dict(EXG8='C29',EXG7 = 'C30'), ['D9']],
        '16EB':[dict(), ['D12','D26']],
        '17SK':[dict(EXG7 = 'D4'), ['C10','C30','C31','C32']],
        '18CK':[dict(EXG7 = 'D4'), []],
        '19LA':[dict(EXG7 = 'D4'), []],
        '20KD':[dict(EXG7 = 'D4'), []],
        '21LS':[dict(EXG7 = 'D4'), ['B4', 'D23']]}
	return replacers[subID]

def RenameChannels(raw,subID):
	'''used to define eog channels, and replace broken electrodes by externals'''
	# load electrode replacement dictionary
	replaced_electrodes = retrieveBads(subID)[0]

	# translate electrode names into indices 
	electrode_idx= {raw.ch_names.index(k):raw.ch_names.index(v) for k,v in replaced_electrodes.items()}
	
	# create placeholder container, storing the content of the replacing electrodes
	foo_content = []
	for pair in electrode_idx.keys():
		foo_content.append({k:v for k,v in raw.info['chs'][pair].items()})
		foo_content[-1]['kind'] = 502

	#replace electrodes
	helper = 0 
	for old,new in electrode_idx.items():
		for parameter in raw.info['chs'][old]:
			raw.info['chs'][old][parameter] = raw.info['chs'][new][parameter]
		raw.info['chs'][new] = foo_content[helper]
		helper+=1
	#define eog channels
	for i in xrange(128,132):
		raw.info['chs'][i]['kind'] = 202
	raw.info['chs'][128]['ch_name'] = 'VEOG1'
	raw.info['chs'][129]['ch_name'] = 'VEOG2'
	raw.info['chs'][130]['ch_name'] = 'HEOG1'
	raw.info['chs'][131]['ch_name'] = 'HEOG2'
	#exclude
	raw.info['chs'][132]['kind'] = 502
	raw.info['chs'][133]['kind'] = 502

	return raw


def findEOGevents(raw,eog_channels):
    '''
    takes all eog channels, finds all eog related events and marks with -99
    returns a dict with all events separetly for channel and one np.array with all unique events
    '''
    seperate_eog_events = {channel:mne.preprocessing.find_eog_events(raw,-99,ch_name = channel) for  channel in eog_channels}
    eog_timestamp = []
    for values  in seperate_eog_events.values():
        for event in values:
            eog_timestamp.append(event[0])
    print len(eog_timestamp)
    eog_events_identifier = np.unique(np.array(eog_timestamp))
    print len(eog_events_identifier)
    eog_events = []
    for values  in seperate_eog_events.values():
        for event in values:
            if event[0] in eog_events_identifier: 
                eog_events.append(event)
                eog_events_identifier = np.delete(eog_events_identifier, np.where(eog_events_identifier ==  event[0]))
    eog_events1 = np.array(eog_events)
    eog_events1 = eog_events1.astype(int)
    eog_events1.sort(axis = 0)
    print('%d blinks were found.'%len(eog_events))
    return seperate_eog_events, eog_events1

def makeEpochs(raw,events, picks,tmin,tmax, event_type = None, baseline = (None,0), reject = None, preload = True):
	'''makes epochs should be self-explanatory'''
	
	event_dict = dict(on_am = 1, on_cw = 2, on_ccw = 3, ccw_strong = 50, cw_strong = 60,ccw_weak = 70, cw_weak = 80, eog = -99, high = 11, medium = 12, low = 13, counterclockwise = 22, clockwise =21)
	baseline = (-.6,0)
	if event_type == None:
		event_id = event_dict
	else:
		event_id = [event_dict[event] for event in event_type]
	
	epochs = mne.Epochs(raw,events, event_id,  tmin,tmax,proj = True, picks = picks,baseline = baseline, reject = reject, preload = preload)
	return epochs


def ExcludeEogEvents(eeg_events,eog_events,tmin = 400,tmax = 1400):
    '''
    checks whether an eog event is within the critical interval of an eeg_event (stimulus onset)
    returns an event list that has seperate identifiers for trials without and with eeg event_type

    '''
    triggers = np.array(eeg_events)
    tmin*=0.512;tmax *= 0.512;
    for pos1,blink in enumerate(eog_events):
    	for pos2,trigger in enumerate(triggers):
    		if  (triggers[pos2][0]- tmin)< eog_events[pos1][0] <(triggers[pos2][0]+tmax):
    			triggers[pos2][2]=99
    			break
    return triggers

def plotErp(epochs,save_loc= None, title = 'Plot', show = False):
	'''
	plots the data for every stimulus type for all channels
	
	'''
	am_data = getEpochdata(epochs,'1')
	cw_data = getEpochdata(epochs,'2')
	ccw_data = getEpochdata(epochs,'3')
	fig = plt.figure(5)
	fig.suptitle(title)
	fig.add_subplot(3,1,1)
	plt.plot(1e3*epochs.times,np.mean(am_data,axis =0).T)
	plt.xlabel('apparent motion trials')
	fig.add_subplot(3,1,2)
	plt.plot(1e3*epochs.times,np.mean(cw_data,axis =0).T)
	plt.xlabel('Smooth clockwise motion')
	fig.add_subplot(3,1,3)
	plt.plot(1e3*epochs.times,np.mean(ccw_data,axis =0).T)
	plt.xlabel('Smooth counterclockwise motion')
	if save_loc != None:
		plt.savefig(save_loc+ title.replace(' ','')+'.pdf')
	if show:
		plt.show()
	plt.close()

def getEpochdata(epochs, condition):
	'''return data of chosen epochs'''

	return epochs[condition].get_data()

def loadPickles(subID):
	'''
	loads the pickle file to be able to access experimental parameters and prepares to use for trigger
	returns: np.array in same format as trigger list

	
	'''

	#os.chdir('/home/shared/AM_EEG/analysis')
	file_location = '/home/shared/AM_EEG/data/pickles/'
	file_list = os.listdir(file_location)
	files = [ i for i in file_list if i[:4] == subID]
	files.sort()
	print files
	parameter_list = []
	stim_translator = {0:1,1:2,-1:3}
	if subID!='14AC':
		resp_translator = {'apostrophe':70,'lshift':60,'b':70,'j':50,'comma':50,-999:-999,0:-999,1:50,-1 :60,0.5:70,-0.5:80,'m':50,'z': 60,'n' : 70,'x':80}
	else:
		resp_translator = {'apostrophe':70,'lshift':-999,'b':70,'j':50,'comma':50,-999:-999,0:-999,1:50,-1 :60,0.5:70,-0.5:80,'m':50,'z': 60,'n' : 70,'x':80}
	for fl in files:
		#pdb.set_trace()
		with open(file_location+fl,'r') as f:
			a = pickle.load(f)
		a['parameterArray'].pop(0)
		a['eventArray'].pop(0)	

		# extract pickle file events
		string_events = [[e.split(' ') for e in tevs if type(e) == str] for tevs in a['eventArray']]
		key_events = [[[int(ee[1]),ee[3],float(ee[5])] for ee in tevs if ee[2] == 'event' and len(ee) == 6] for tevs in string_events]
		phase_events = [[[int(ee[1]),int(ee[3]),float(ee[6])] for ee in tevs if ee[2] == 'phase'and len(ee) == 7] for tevs in string_events]

		#merge key/phase events with parameterArray
		for i in range(len(a['parameterArray'])):
			a['parameterArray'][i]['tot_stim_on_time'] = phase_events[i][0][-1]
			a['parameterArray'][i]['tot_stim_off_time'] = phase_events[i][1][-1]
			if len(key_events[i]) == 1:
				a['parameterArray'][i]['resp_time'] = key_events[i][-1][-1]
			elif i != len(a['parameterArray'])-1 and len(key_events[i+1]) == 2:
				a['parameterArray'][i]['resp_time'] = key_events[i+1][-2][-1]
				a['parameterArray'][i]['answer'] = key_events[i+1][-2][1]
				key_events[i+1].pop(-2)
			elif len(key_events[i]) == 2:
				a['parameterArray'][i]['resp_time'] = key_events[i][-2][-1]
			else:
				a['parameterArray'][i]['resp_time'] = -999
				a['parameterArray'][i]['answer'] = -999
		exp_block = [[p for p in a['parameterArray']]]
		exp_block.append(exp_block[0][0]['tot_stim_on_time'])

		for l in range(len(exp_block[0])):  
			# set every block beginning to t=0
			exp_block[0][l]['tot_stim_on_time'] -= exp_block[1]
			exp_block[0][l]['tot_stim_off_time'] -= exp_block[1]
			exp_block[0][l]['resp_time'] -= exp_block[1]
			exp_block[0][l]['block_no'] = int(fl[5])
		#modify layout of dictionary to match with trigger list
	
		for trial in exp_block[0]:
			parameters = [int(1000*trial['tot_stim_on_time']),int(0),int(stim_translator[trial['trial_type']])]
			parameter_list.append(parameters)
			parameters = [int(1000*trial['resp_time']),int(0),int(resp_translator[trial['answer']])]
			parameter_list.append(parameters)

	os.chdir('/home/shared/AM_EEG/analysis/')
	return np.array(parameter_list)

def CleanEvents(unfixed_events, event_type= None):
	'''
	takes raw events as input and adjust the trigger type that was sent. "resting trigger" as measure to infer what was probably the original trigger
	Event type specifies whether eeg triggers are fixed or the pickle file ones. Valid keys words are: 'triggers' and 'pickles'
	returns another array of events. Kick every trigger out where something weird is happening

	'''
	if event_type == 'triggers':

		# drop first trial 
		if unfixed_events[0][2] not in [1,2,3]:
			unfixed_events = np.delete(unfixed_events,0,axis = 0)

		# rename not properly named triggers that are sent properly aside of this
		for i in xrange(len(unfixed_events)):
			if unfixed_events[i][2] not in [1,2,3,50,60,70,80]:
				if unfixed_events[i][1] != 0:
					unfixed_events[i][2] -= unfixed_events[i][1]

	# remove every trial that has not the perfect shape: "stim-resp"
	kicks = []		
	comp = 0
	resp_dict = {0 : [1,2,3],1 : [50,60,70,80]}
	for triggerI, trigger in enumerate(unfixed_events):
		if trigger[2] not in resp_dict[comp]:
			if trigger[2] == -999:
				kicks.append(triggerI)
				kicks.append(triggerI-1)
				comp ^= 1
			elif trigger[2] in resp_dict[0]:
				kicks.append(triggerI-1)
			elif trigger[2] in resp_dict[1] or trigger[2] not in [1,2,3,50,60,70,80,-999]:
				kicks.append(triggerI)
		else: 
			comp ^=1

	fixed_events = np.delete(unfixed_events, kicks, axis = 0 )

	return fixed_events

def CreateEvents(unfixed_events, pickle_events, pickle_file, subj):
	'''
	The way this is done is by create new triggers. 11, 12 and 13 for high, medium and low adaptation (based on terzentiles)
	returns new array of events
	'''

	fixed_events = np.array(unfixed_events)
	behav_file = np.array(pickle_events)

	# convert pickle timepoints into indices:
	behav_file[:,0] = behav_file[:,0] * 0.512

	if subj == '04RB': #has to be dealt with separatly, because recording started 10 minutes later
		garb_tr = list(xrange(0,30))
		fixed_events = np.delete(fixed_events,garb_tr, axis= 0)
		garb_bh = list(xrange(0,470))
		behav_file = np.delete(behav_file,garb_bh, axis= 0)
		behav_file[:,0] = behav_file[:,0] - 322308#296595
	if subj =='21LS':
		scaler = behav_file[0][0]
		behav_file[:,0] = behav_file[:,0] - scaler
	else:
		scaler = 0
	#adjust beginning of pickle sessions to timepoints of trigger sessions
	
	enter = True #enter adaptation loop?
	block = 1 # which block in pickle file
	timess =  fixed_events[:,0]
	block_change = timess[0] # time to take care of trigger/pickle difference at block change

	for i,event in enumerate(behav_file):
		if (behav_file[i][0]==0  or behav_file[i][0]==-322308 or behav_file[i][0]== -scaler) and i != 0:
			if block ==1 and subj == '04RB':
				behav_file[i:,0] = behav_file[i:,0] + 322308
			elif block ==1 and subj == '21LS':
				behav_file[i:,0] = behav_file[i:,0] +scaler
			idx =  np.where(np.logical_and(timess>=behav_file[i-1][0]-8, timess<=behav_file[i-1][0]+8,))[0]
			while not( behav_file[i+1][0]-10 < abs(timess[idx[0]+1]-timess[idx[0]+2]) < behav_file[i+1][0]+10):
				idx[0]+=1
			block_change = fixed_events[idx[0]+1][0]
			block += 1
			enter = True
		if subj =='16EB' and i ==1594:
				block_change -=(2000*0.512)
		behav_file[i][0]+=block_change
		
		# and change trigger type corresponding to level of adaptation
		if enter:
			adaptation_list = pickle_file[subj][block]['adaptation_state'][0][1]
			rt_times  = np.array([i* 512 for i in pickle_file[subj][block]['adaptation_state'][0][0]])
			if subj == '04RB' and block ==1:
				rt_times = np.delete(rt_times,list(xrange(0,235)),axis = 0)
				adaptation_list = np.delete(adaptation_list,list(xrange(0,235)),axis = 0)

			terzentiles = mquantiles(adaptation_list, prob=[0.33,0.67])

			for t_idx, adap in enumerate(adaptation_list):
				ad_idx = np.where(np.logical_and(timess>= rt_times[t_idx]-8+block_change, timess<=rt_times[t_idx]+8+block_change))[0]
				if ad_idx.shape != (0,) and fixed_events[ad_idx[0]-1][2]==1:#take only AM trials
					#set adaptation state to low medium or high
					if adap<terzentiles[0]:
						fixed_events[ad_idx[0]-1][2] = 11 # high adaptation
					elif terzentiles[0]<= adap <= terzentiles[1]:
						fixed_events[ad_idx[0]-1][2] = 12 # medium adaptation
					elif  adap> terzentiles[1]:
						fixed_events[ad_idx[0]-1][2] = 13 # low adaptation	
			enter = False

	return fixed_events

def recodeResponse(events):
	'''
	Changes stimulus onset triggers depending on conterclockwise vs. clockwise
	'''

	for eventI,event in enumerate(events):
		if event[2] in [50,70] and events[eventI-1][2] == 1:
			events[eventI-1][2]=21
		elif event[2] in [60,80] and events[eventI-1][2] == 1:
			events[eventI-1][2]=22
	return events

def timeFreq(raw,epochs,condition,freqs, n_cycles = 3, correction = None):
	'''
	simple version for sake of testing effects. Not tailored to needs of data. Do this later
	'''
	data = getEpochdata(epochs, condition)
	Fs = raw.info['sfreq']
	times = 1e3*epochs.times
	decim = 5
	n_jobs = 20

	power,phase_lock = mne.time_frequency.induced_power(data, Fs = Fs, frequencies = freqs, decim = 5,n_cycles = n_cycles, n_jobs = n_jobs)
	#power_corr = power/ np.mean(power[:, :, -500 <  np.logical_and(-500 < times[::decim], times[::decim] < -200)], axis=2)[:, :, None]  
	power_corr = (power[:,:,:].T/power[:,:,100:133].mean(axis=-1).T).T

	return power_corr, phase_lock



def elecSelection(info, epochs,picks,  tmin, tmax, n_jobs):
	'''
	Takes some epochs and returns channels that are significantly activated. 
	Tmin and Tmax indicate interval, which has to be tested 
	returns: channel names
	'''

	data = epochs['1'].get_data()
	times = epochs.times*1e3

	temporal_mask = np.logical_and( tmin <= times, times <= tmax)
	data = np.squeeze(np.mean(data[:, :, temporal_mask], axis=2))

	n_permutations = 50000
	T0, p_values, H0 = mne.stats.permutation_t_test(data, n_permutations, tail = 1,n_jobs=n_jobs)

	significant_sensors = picks[p_values <= 0.01]
	significant_sensors_names = [info['ch_names'][k] for k in significant_sensors]

	
	print "Number of significant sensors : %d" % len(significant_sensors), "Sensors names : %s" % significant_sensors_names
	return significant_sensors_names

def powerPermutation(power1, power2, permutations = 100, threshold = 6, n_jobs = 15):
	'''quite standard procedure. Check documentations
	'''	
	T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test([power1, power2], n_permutations=100, threshold=threshold, tail=0, n_jobs = n_jobs)
	
	return T_obs, clusters, cluster_p_values, H0

def singleTrial(epochs,condition,freqs,baseline,n_cycles = None, correction = None, n_jobs = 15):
	'''
	simple version for sake of testing effects. Not tailored to needs of data. Do this later
	'''
	if type(condition) != list:
		data = getEpochdata(epochs, condition)
	else:
		data11 = getEpochdata(epochs, condition[0])
		data13 = getEpochdata(epochs, condition[1])
		data = data11 -data13

	Fs = 512.0
	times = 1e3 * epochs.times
	if correction != None:
		power=mne.time_frequency.single_trial_power(data, Fs = Fs, frequencies = freqs,baseline =baseline, baseline_mode = 'logratio', n_cycles = n_cycles,times = times, decim = 5, n_jobs =n_jobs)
		return power
	else:
		power=mne.time_frequency.single_trial_power(data, Fs = Fs, frequencies = freqs,baseline =baseline, n_cycles = n_cycles,times = times, decim = 5, n_jobs = n_jobs)
		power_corr = (power[:,:,:,:].T/power[:,:,:,100:133].mean(axis=-1).T).T
		return power_corr

def plotPower(power,times, condition,  t_min = 140, t_max =240, interpolation = True,  save_loc = None, title = 'Power'):
	'''
	Creates time frequency plots of input power arrays. These arrays have to be 2D (freqs X times)
	'''
	
	cond_translator = dict(diff='Difference between high and low adaptationstate', high = 'High adaptation state', low = 'Low adaptation state', total = 'All apparent motion epochs')
	freqs = np.exp(np.linspace(np.log(2),np.log(120), 24))
	#initialize figure only once
	vmax= np.max([np.max(power[p]) for p in power.keys()])
	vmin = np.min([np.min(power[p]) for p in power.keys()])
	fig, ax = plt.subplots(len(power.keys()),1)
	if type(ax) != np.ndarray:
		ax = [ax]
	# create grant title only if there are multiple plots
	fig.suptitle(title+' Per Condition', fontsize=18, fontweight = 'bold')
	for condI, con in enumerate(condition): 
		#choose between interpolated figure or smooth one
		if interpolation == None:		
			im = ax[condI].imshow(power[con],extent=[times[t_min], times[t_max], freqs[0], freqs[-1]],aspect='auto', origin='lower', vmin = vmin, vmax = vmax, interpolation = 'none')		
		else:
			im = ax[condI].imshow(power[con],extent=[times[t_min], times[t_max], freqs[0], freqs[-1]],aspect='auto', origin='lower', vmin = vmin, vmax = vmax)
			
		#add subplot values
		if len(power.keys())>1:
			ax[condI].set_title(cond_translator[con])
		#adjust axis settings
		ax[condI].set_yscale('log', subsy = [])
		ax[condI].set_yticks(freqs[1::3])
		ax[condI].set_yticklabels(np.around(freqs[1::3],1))
		ax[condI].tick_params(axis='y', which='minor',   left='off', right = 'off')
		ax[condI].tick_params(axis='x', which='both',  top='off')

	#axis labels	
	fig.text(0.5, 0.10, 'Time in ms', ha='center', va='center', fontsize=18, fontweight = 'bold')
	fig.text(0.06, 0.5, 'Frequencies: 2-120Hz', ha='center', va='center', rotation='vertical', fontsize=18, fontweight = 'bold')
	plt.tight_layout()	
	fig.subplots_adjust(bottom=0.2,left = .10, top = 0.87, right = 0.87)	
	cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
	fig.colorbar(im, cax=cax)
	

	if len(condition) !=1:
		plt.savefig(save_loc+title.replace(' ','')+'_per_condition'+'.pdf')
	else:
		plt.savefig(save_loc+con+'_'+title.replace(' ','')+'_per_condition'+'.pdf')
	plt.close()


def	SinglePowerPermutationTest(epochs_power, times, threshold = 2.5, tail = 0, n_permutations = 500,n_jobs = 10, plot = False, title = None, save_loc = None, show = False):
	'''
	Permutation test, permuting trials, to see significant activitations for every subject.
	returns T_obs (T values), clusters, cluster_p_values, H0
	Attention: Last 3 arguments only valid if plot True
	'''

	power = np.array(epochs_power)
	#chop off what isnt needed
	time_mask = (times > -100) & (times < 850)
	times = times[time_mask]
	power = power[:,:, time_mask].mean(axis = 1)

	T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(power, n_permutations=n_permutations, threshold=threshold, tail=tail)#, n_jobs = n_jobs)
	#reject_fdr, pval_fdr = mne.stats.fdr_correction(cluster_p_values, alpha=0.05, method='negcorr')
	print cluster_p_values##pval_fdr
	if not plot:
		return T_obs, clusters, cluster_p_values, H0
	else:
		if not show:
			plt.ioff()
		# Create new stats image with only significant clusters
		T_obs_plot = np.nan * np.ones_like(T_obs)
		for c, p_val in zip(clusters, cluster_p_values):#pval_fdr):
		    if p_val <= 0.05:
		        T_obs_plot[c] = T_obs[c]

		#vmax = np.max(np.abs(T_obs))
		#vmin = -vmax
		plt.imshow(T_obs, cmap=plt.cm.gray,extent=[times[0], times[-1], 2, 120], aspect='auto', origin='lower')#, vmin=vmin, vmax=vmax)
		plt.imshow(T_obs_plot, cmap=plt.cm.jet, extent=[times[0], times[-1], 2, 120],aspect='auto', origin='lower')#, vmin=vmin, vmax=vmax)
		plt.colorbar()
		plt.yscale('log')
		plt.xlabel('time (ms)')
		plt.ylabel('Frequency (2-120Hz)')
		plt.tight_layout()
		plt.title(title)
		if save_loc:
			plt.savefig(save_loc+title.replace(' ','')+'.pdf')
		plt.close()
		return T_obs, clusters, cluster_p_values, H0		


def makePowerPlots(files,condition = None, times = None,  t_min = 140, t_max =260,  interpolation = True, grand_mean = False, keep_arrays = False,save_loc = None):
    ''' 
    Loads power arrays from joblib pickle files and creates plots. If grand_mean == True, a total average is also calculated, if keep_arrays==True the power arrays are returned. 
    condition is string or array of strings, and can contain [total, high, low and diff]
    '''   

    results_path = '/home/shared/AM_EEG/results/eeg/'
    data_path = '/home/shared/AM_EEG/data/fif/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    #create arrays containing power
    power_pop = dict();total_power = dict()    
    for subjI,subj in enumerate(files):
        power_pop[subj] = dict(); 
    for subjI,subj in enumerate(files):	
        for cond in condition:
            print subj, cond
            
            open_file = data_path[:-4]+'power/'+subj+'_'+ cond +'_power'
            T_obs, clusters, cluster_p_values, H0	= mne.stats.permutation_cluster_1samp_test(joblib.load(open_file), n_permutations=500, threshold=2.5, tail=0)
            T_obs_plot = np.nan * np.ones_like(T_obs)
            for c, p_val in zip(clusters, cluster_p_values):#pval_fdr):
                if p_val <= 0.05:
	                T_obs_plot[c] = T_obs[c]
	        plt.imshow(T_obs, cmap=plt.cm.gray,extent=[times[0], times[-1], 2, 120], aspect='auto', origin='lower')#, vmin=vmin, vmax=vmax)
	        plt.imshow(T_obs_plot, cmap=plt.cm.jet, extent=[times[0], times[-1], 2, 120],aspect='auto', origin='lower')#, vmin=vmin, vmax=vmax)
	        plt.colorbar()
	        plt.show()

    total_power[cond] = np.zeros((21,24,119))
    time_mask = (t_min <np.array(xrange(power_pop[subj][cond].shape[3])) )& (np.array(xrange(power_pop[subj][cond].shape[3]))<t_max)
    #prepare plots
    trialAverage = dict();  #  electrodeAverage  = dict();    
    for subjI,subj in enumerate(power_pop.keys()):
        print subj
        for condi, cond in enumerate(condition):   
            trialAverage[cond] = power_pop[subj][cond][:,:,:,time_mask].mean(axis = 0)#[:,:,:,(times > t_min) & (times <t_max)]
            #electrodeAverage[cond]= trialAverage[cond].mean(axis = 0)
            total_power[cond][subjI] = trialAverage[cond].mean(axis = 0)
        #power plots per subject
        #plotPower(power = electrodeAverage,times = times,condition =condition , interpolation = True, save_loc = save_loc+subj+'/', title= 'Single Trial Power ['+subj+'] '+'_'.join(condition))   
    del trialAverage#, electrodeAverage  
    #total_power_array = dict();total_power_array1 = dict()
    # TF array averaged over subjects, trials and eletrodes
    #for cond in condition:
        #total_power_array1[cond] = np.array(total_power[cond]).mean(axis = 0)
        #total_power_array[cond]  =np.array(total_power[cond])
        #test_array = np.array(total_power[cond]).reshape((21,2856))
 
    #del total_power
   	#print test_array.shape
            #plot it
    	#plotPower(power = total_power_array1, times = times, condition = condition, interpolation = True, save_loc = save_loc, title= 'Single Trial Power Over All Subjects' )#+'_'.join(condition))  
    T_obs, clusters, cluster_p_values, H0	= mne.stats.permutation_cluster_1samp_test(total_power[cond], n_permutations=500, threshold=2.5, tail=0)
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):#pval_fdr):
            if p_val <= 0.05:
                T_obs_plot[c] = T_obs[c]
    plt.imshow(T_obs, cmap=plt.cm.gray,extent=[times[0], times[-1], 2, 120], aspect='auto', origin='lower')#, vmin=vmin, vmax=vmax)
    plt.imshow(T_obs_plot, cmap=plt.cm.jet, extent=[times[0], times[-1], 2, 120],aspect='auto', origin='lower')#, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()

    T = T0.reshape((24,119))
    p = p_values.reshape((24,119))
    print np.max(np.abs(T)), np.min(p)
    keep_arrays = False
    if keep_arrays== True:
        return power_pop, T,p
    del power_pop
"""

############################################################################
OUTDATED FUNCTIONS: NOT USED RIGHT NOW, BUT MAYBE NEEDED LATER
############################################################################

def readData(index):
	'''Reads .bdf files and returns into raw fiff format'''

	os.chdir('/home/shared/AM_EEG/data/bdf/')
	filename = os.listdir('/home/shared/AM_EEG/data/bdf')[index]
	if filename == '.AppleDouble':
		raise ValueError('THis is not a Bdf File. Pick another index')
	print filename
	raw_data = mne.fiff.edf.read_raw_edf(filename,stim_channel = -1,hpts = "biosemi128.hpts",preload = True)
	raw, ref_data = mne.fiff.raw.set_eeg_reference(raw_data, ref_channels =['EXG5','EXG6'])
	return raw, raw_data	

def RenameChannels(subID):

	'''
	path = '/home/shared/AM_EEG/data/'
	replacement = path+ 'rename_files/' +'alias.txt'
	filename  = path+ 'fif/' + subID + '_'+'raw.fif'
	subprocess.call(['mne_rename_channels','--fif',filename,'--alias',replacement])
	print 'Done'
	'''

def CorrectBlinkArtefacts(raw):
	'''Running an independent component analysis to  find pca components that resemble blinking artefacts
	returns index of components'''
	#ica = mne.preprocessing.ICA (n_components = 49,random_state = 0)
	ica = mne.preprocessing.ica.ICA(n_components = 49,n_pca_components=None,max_pca_components = 64, random_state=0)
	ica.decompose_raw(raw,picks= picks, decim = 3)
	
	# plot all found components
	start_plot, stop_plot = 200, 210.
	#ica.plot_sources_raw(raw,  start=start_plot, stop=stop_plot)

	# define comparison function
	#corr = lambda x, y: np.array([pearsonr(a, y.ravel()) for a in x])[:, 0]
	
	# find component which resembles blinks most
	eog_scores = ica.find_sources_raw(raw, target='EOG', score_func= 'pearsonr')
	eog_source_idx = np.abs(eog_scores).argmax()
	print eog_scores
	
	# plot shape of component
	ica.plot_sources_raw(raw, eog_source_idx, start = 200, stop = 210)

	#add component to exclude list
	ica.exclude += [eog_source_idx]
	raw_ica = ica.pick_sources_raw(raw, include  = None)


	#control plot. Pick a piece of data, in which you know to find a blink in the raw. and look whether you can still
	#find it in the corrected data
	start_compare, stop_compare = raw.time_as_index([100, 106])
	data, times = raw[picks, start_compare:stop_compare]
	data_clean, _ = raw_ica[picks, start_compare:stop_compare]
	

	return raw_ica 

def plotPower(power,times,  t_min = 140, t_max =300, interpolation = True, subplots = 1, save_loc = None, show= False, title = 'Power'):
	'''
	Creates time frequency plots of input power arrays. These arrays have to be 2D (freqs X times)
	'''
	freqs = np.exp(np.linspace(np.log(2),np.log(120), 24))
	if show == False:
		plt.ioff()
	if interpolation == None:
		plt.imshow(power[:,t_min:t_max],extent=[times[t_min], times[t_max], freqs[0], freqs[-1]],aspect='auto', origin='lower', interpolation = 'none')
	else:
		plt.imshow(power[:,t_min:t_max],extent=[times[t_min], times[t_max], freqs[0], freqs[-1]],aspect='auto', origin='lower')
	plt.title(title)
	plt.yscale('log')
	plt.colorbar()
	plt.ylabel('Frequencies: 2-120Hz')
	plt.xlabel('Time in ms')
	plt.tight_layout()
	if save_loc:
		plt.savefig(save_loc+title.replace(' ','')+'.pdf')
	plt.close()


"""