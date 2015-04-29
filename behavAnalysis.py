#!/usr/bin/env python
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import analUtils as au
import pylab as pl
from IPython import embed as shell


# load data
os.chdir('/home/shared/AM_EEG/data/pickles/')
file_location = '/home/shared/AM_EEG/data/pickles/'
file_list = os.listdir(file_location)
file_list.sort()
total_block_results =[] #contains M/SD of perceptual blocks for entire sample
results_per_subject = dict() #contains mean/sd of duration of perceptual blocks separately per subject
peak_freq = [];time = []
ids = None
#create dicts for results per subject
for j in file_list:
    if j[0]!='.':
        results_per_subject[j[:4]]=dict()
# helper list with dict.keys
plot_indices = results_per_subject.keys()
plot_indices.sort()
no = 50

response_distribution = []

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#####################################################################
LOOP OVER PICKLEFILES TO EXTRAXT DATA AND DO ANALYSES
    1) Prepare datafile to be easier interpretable
    2) Create some lists and do analysis
#####################################################################
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


for o,j in enumerate(file_list):
    if j[0]!='.':# and j[:4] == '01TK':
        with open(file_location+j,'r') as f:
            a = pickle.load(f)
            print j
        shell()
        path = '/home/shared/AM_EEG/results/behavioral/'+j[:4]+'/'
                
        if not os.path.exists(path):
           os.makedirs(path)
        # remove first trial with instructions
        a['parameterArray'].pop(0)
        a['eventArray'].pop(0)
        a['parameterArray'].pop(-1)
        a['eventArray'].pop(-1)

        # extract pickle file
        string_events = [[e.split(' ') for e in tevs if type(e) == str] for tevs in a['eventArray']]
        key_events = [[[int(ee[1]),ee[3],float(ee[5])] for ee in tevs if ee[2] == 'event' and len(ee) == 6] for tevs in string_events]
        phase_events = [[[int(ee[1]),int(ee[3]),float(ee[6])] for ee in tevs if ee[2] == 'phase'and len(ee) == 7] for tevs in string_events]

        #merge key/phase events with parameterArray

        resp_translator = {'m':1,'z': -1,'n' : 0.5,'x':-.5,'apostrophe' :0, 'lshift':0}

        for i in range(len(a['parameterArray'])):
            a['parameterArray'][i]['tot_stim_on_time'] = phase_events[i][0][-1]
            a['parameterArray'][i]['tot_stim_off_time'] = phase_events[i][1][-1]
            if len(key_events[i]) == 1:
                a['parameterArray'][i]['resp_time'] = key_events[i][-1][-1]
            elif i != len(a['parameterArray'])-1 and len(key_events[i+1]) == 2:
                a['parameterArray'][i]['resp_time'] = key_events[i+1][-2][-1]
                a['parameterArray'][i]['answer'] = resp_translator[key_events[i+1][-2][1]]
                key_events[i+1].pop(-2)
            elif len(key_events[i]) == 2:
                a['parameterArray'][i]['resp_time'] = key_events[i][-2][-1]
            else:
                a['parameterArray'][i]['resp_time'] = a['parameterArray'][i]['tot_stim_on_time']+(a['parameterArray'][i]['tot_stim_off_time']-a['parameterArray'][i]['tot_stim_on_time'] ) /2.0
                a['parameterArray'][i]['answer'] = 0

        #create responses and timecourse
        exp_block = [[p for p in a['parameterArray']]]
        exp_block.append(exp_block[0][0]['tot_stim_on_time'])

        for l in range(len(exp_block[0])):  
             # set every block beginning to t=0
            exp_block[0][l]['tot_stim_on_time'] -= exp_block[1]
            exp_block[0][l]['tot_stim_off_time'] -= exp_block[1]
            exp_block[0][l]['resp_time'] -= exp_block[1]
            exp_block[0][l]['block_no'] = int(j[5])
            response_distribution.append(exp_block[0][l]['answer'])
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        #####################################################################
        Create list and arrays and do anaylsis on them
        #####################################################################
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        original_responses = np.array([k['answer'] for k in exp_block[0]])
        direction_responses = np.array([-1 if res<0 else 1 if res>0 else 0 for res in original_responses])
        vagueness_responses = np.array( [-1 if 0<abs(res)<=0.5 else 1 if 0.5<abs(res)<=1 else 0 for res in original_responses])

        response_times = [k['resp_time'] for k in exp_block[0]]
        orig_exp_responses1,orig_exp_responses2 = au.expConvolve(original_responses)
        dir_exp_responses1,dir_exp_responses2 = au.expConvolve(direction_responses)# -1 for CCW, 1 for CW   
        vag_exp_responses1,vag_exp_responses2 = au.expConvolve(vagueness_responses)# -1 for vague, 1 for definite    

        # smooth data per condition and subject
        window_size= 27
        smoothed_orig_resp = au.SmoothArray(original_responses,window_size)
        smoothed_dir_resp = au.SmoothArray(direction_responses,window_size)
        smoothed_vag_resp = au.SmoothArray(vagueness_responses,window_size)

        #fourier transform to see frequency at which percpts cycle 
        orig_freq,orig_spAbs = au.FourierTransform(smoothed_orig_resp)
        dir_freq,dir_spAbs = au.FourierTransform(smoothed_dir_resp)

        #getting peak frequencies
        if j[:4]== ids:
            time.append(exp_block[0][0]['block_no'])
            dir_peak_freq.append(dir_freq[np.argmax(dir_spAbs)])
            orig_peak_freq.append(orig_freq[np.argmax(orig_spAbs)])
        else:
            time = [exp_block[0][0]['block_no']]
            dir_peak_freq = [dir_freq[np.argmax(dir_spAbs)]]
            orig_peak_freq = [orig_freq[np.argmax(orig_spAbs)]]
        

        #time space representation of frequencies
        orig_cycle = [1.0/i for i in orig_freq]  
        dir_cycle = [1.0/i for i in dir_freq]  

        # calculate duration of each perceptual block for every of the 3 ways to code
        cutoff = 0.0
        orig_bias_list = au.CalculateStability(smoothed_orig_resp,response_times,cutoff)
        dir_bias_list = au.CalculateStability(smoothed_dir_resp,response_times,cutoff)
        vag_bias_list = au.CalculateStability(smoothed_vag_resp,response_times,cutoff)


        #calculate adaption:
        dir_adaptation = [response_times, abs(dir_exp_responses1 - dir_exp_responses2)]
        vag_adaptation = [response_times, abs(vag_exp_responses1 - vag_exp_responses2)]

        #collect arrays and lists better be saved
        unfiltered_responses = [original_responses, direction_responses,vagueness_responses]
        gaussian_responses = [smoothed_orig_resp,smoothed_dir_resp,smoothed_vag_resp]
        exponential_responses = [original_responses,direction_responses,vagueness_responses,dir_exp_responses1,dir_exp_responses2,vag_exp_responses1,vag_exp_responses2,orig_exp_responses1,orig_exp_responses2]
        #adaptation_state = [original_posBlock,direction_posBlock,vagueness_posBlock]
        adaptation_state = [dir_adaptation, vag_adaptation]

        
        # finish lists with resulting data
        block_results= [[np.mean(orig_bias_list),np.std(orig_bias_list)],[np.mean(dir_bias_list),np.std(dir_bias_list)],np.mean(vag_bias_list),np.std(vag_bias_list)] # contains mean and SD for one subject for all conditions    
        total_block_results.append(block_results)

        results_per_subject[j[:4]][exp_block[0][0]['block_no']] = dict(unfiltered_responses = unfiltered_responses,  gaussian_responses = gaussian_responses,exponential_responses = exponential_responses, block_results = block_results, adaptation_state = adaptation_state)

        #gabriel = [i for i in response_distribution if i == 1 or i == -1]
        #asrael = [i for i in response_distribution if i == 0.5 or i == -0.5]
        #print 'gabi: ', 100* len(gabriel)/float(len(response_distribution))
        #print 'asi: ', 100 *len(asrael)/float(len(response_distribution))

        '''Shape of result dict:
            subject1: list(    session 1:   list: original responses coded in three different ways: original, direction sensitive and vagueness sensitive 
                                            list: gaussian filtered timecourse for every way of coding responses
                                            list: exponentially filtered timecourse for every way of coding responses (separately for - vs + values)
                                            list: block results containing mean and sd of duration of perceptual block for every way of coding responses
                                            list: adapation state contains indices where in one perceputal cycle a timepoint is located, so how adapted it is (quartiles) for every type of responses
                                                
                              session 2:
                              .
                              last session )

            subject2:    "
            .
            .
            .
            last subject:    "
            '''


        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        #####################################################################
        CREATE PLOTS
                1) create plots to later plot together with plots for all subjects
                2) create single session plot for one subject only 
        #####################################################################
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        '''''''''''''''''''''''
        group plot
        '''''''''''''''''''''''    
        #smoothed and original timecourse for original responses
        fig = pl.figure(1)
        pl.subplot(len(results_per_subject.keys()),4,0 + plot_indices.index(j[:4])*4 + exp_block[0][0]['block_no'] )
        x1,y1 = pl.plot(response_times,original_responses,response_times,smoothed_orig_resp)        
        pl.axis([0,response_times[-1]+20,-1.5,1.5])

        #smoothed and original timecourse direction sensitive responses
        fig = pl.figure(2)
        pl.subplot(len(results_per_subject.keys()),4,0 + plot_indices.index(j[:4])*4 + exp_block[0][0]['block_no'] )
        x2,y2 = pl.plot(response_times,original_responses,response_times,smoothed_dir_resp)        
        pl.axis([0,response_times[-1]+20,-1.5,1.5])


        #smoothed and original timecourse vagueness sensitive responses
        fig = pl.figure(3)
        pl.subplot(len(results_per_subject.keys()),4,0 + plot_indices.index(j[:4])*4 + exp_block[0][0]['block_no'] )
        x3,y3 = pl.plot(response_times,original_responses,response_times,smoothed_vag_resp)        
        pl.axis([0,response_times[-1]+20,-1.5,1.5])

        # alternation frequencies origninal responses
        fig = pl.figure(4)
        pl.subplot(len(results_per_subject.keys()),4,0 + plot_indices.index(j[:4])*4 + exp_block[0][0]['block_no'] )
        pl.plot(orig_freq,orig_spAbs)

        # alternation frequencies direction sensitive
        fig = pl.figure(5)
        pl.subplot(len(results_per_subject.keys()),4,0 + plot_indices.index(j[:4])*4 + exp_block[0][0]['block_no'] )
        pl.plot(dir_freq,dir_spAbs)

        # peak frequency orginal responses
        fig = pl.figure(6)  
        pl.plot(time,orig_peak_freq)
        plt.xlabel('Block',fontsize=18)
        plt.ylabel('Frequency in Hz',fontsize=18)
        plt.xticks(np.arange(1,5,1))
        pl.tight_layout()

        # peak frequency direction  responses
        fig = pl.figure(7)
        pl.plot(time,dir_peak_freq)
        plt.xlabel('Block',fontsize=18)
        plt.ylabel('Frequency in Hz',fontsize=18)
        plt.xticks(np.arange(1,5,1))
        pl.tight_layout()

        # plot time space version of alternation frequencies: How much time is needed for one entire cycle , original responses
        fig = pl.figure(8)
        pl.subplot(len(results_per_subject.keys()),4,0 + plot_indices.index(j[:4])*4 + exp_block[0][0]['block_no'] )
        pl.plot(orig_cycle,orig_spAbs)
        pl.xlim([0,1.0/0.002])

        # plot time space version of alternation frequencies: How much time is needed for one entire cycle, direction sensitive responses
        fig = pl.figure(9)
        pl.subplot(len(results_per_subject.keys()),4,0 + plot_indices.index(j[:4])*4 + exp_block[0][0]['block_no'] )
        pl.plot(dir_cycle,dir_spAbs)
        pl.xlabel('Time in seconds')
        pl.xlim([0,1.0/0.002])

        # adapatation and stuff-->exp fiilterd timecourse, all responses, as they are
        fig = pl.figure(10)
        pl.subplot(len(results_per_subject.keys()),4,0 + plot_indices.index(j[:4])*4 + exp_block[0][0]['block_no'] )
        aa,ab = pl.plot(response_times, orig_exp_responses1,response_times, orig_exp_responses2)
        pl.axis([0,response_times[-1]+20,-1.5,1.5])

        # adapatation and stuff-->exp fiilterd timecourse, responses vagueness degree pooled
        fig = pl.figure(11)
        pl.subplot(len(results_per_subject.keys()),4,0 + plot_indices.index(j[:4])*4 + exp_block[0][0]['block_no'] )
        ac,ad=pl.plot(response_times, dir_exp_responses1,response_times, dir_exp_responses2)
        pl.axis([0,response_times[-1]+20,-1.5,1.5])

        # adapatation and stuff-->exp fiilterd timecourse, responses direction pooled
        fig = pl.figure(12)
        pl.subplot(len(results_per_subject.keys()),4,0 + plot_indices.index(j[:4])*4 + exp_block[0][0]['block_no'] )
        ae ,af =pl.plot(response_times, vag_exp_responses1,response_times,vag_exp_responses2)
        pl.axis([0,response_times[-1]+20,-1.5,1.5])

        #combined plots for vagueness and direction
        fig = pl.figure(13)
        pl.subplot(len(results_per_subject.keys()),4,0 + plot_indices.index(j[:4])*4 + exp_block[0][0]['block_no'] )
        ag,ah,ai,aj = pl.plot(response_times, dir_exp_responses1,response_times, dir_exp_responses2,response_times, vag_exp_responses1,response_times,vag_exp_responses2)
        pl.axis([0,response_times[-1]+20,-1.5,1.5])
        
        #adaptation direction vs. adaptation vagueness
        fig = pl.figure(14)
        pl.subplot(len(results_per_subject.keys()),4,0 + plot_indices.index(j[:4])*4 + exp_block[0][0]['block_no'] )
        am,an = pl.plot(response_times,dir_adaptation[1],response_times,vag_adaptation[1])        
        pl.axis([0,response_times[-1]+20,-0.1,1.1])



        '''''''''''''''''''''''
        single plot
        '''''''''''''''''''''''
        if j[:4]!=ids:
            no += 1
            save = True
        #smoothed and original timecourse for original responses
        fig = pl.figure(no+40)
        fig.suptitle('Timecourse of Original Responses',fontsize=15, fontweight = 'bold')
        pl.subplot(4,1,exp_block[0][0]['block_no'] )
        org, sorg = pl.plot(response_times,original_responses,response_times,smoothed_orig_resp)
        pl.axis([0,response_times[-1]+20,-1.5,1.5])
        if exp_block[0][0]['block_no'] == 3:
            pl.xlabel('Time in seconds',fontsize=18)
            
            fig.legend((org,sorg),('Original responses', 'Smoothed responses'),bbox_to_anchor=(0.95, 0.07),loc='lower right', bbox_transform=pl.gcf().transFigure,borderaxespad=0.,prop={'size':15})
            pl.tight_layout()
            pl.subplots_adjust(top=0.90)
        pl.savefig(path+j[:4]+'_orig_timecourse.pdf')

        #smoothed and original timecourse direction sensitive responses
        fig = pl.figure(no+41)
        pl.subplot(4,1,exp_block[0][0]['block_no'] )
        pl.plot(response_times,original_responses,response_times,smoothed_dir_resp)
        pl.axis([0,response_times[-1]+20,-1.5,1.5])
        pl.tight_layout()
        pl.xlabel('Time in seconds')
        if save:
            pl.savefig(path+j[:4]+'_dir_timecourse.pdf')

        #smoothed and original timecourse vagueness sensitive responses
        fig = pl.figure(no+42)  
        pl.subplot(4,1,exp_block[0][0]['block_no'] )
        pl.plot(response_times,original_responses,response_times,smoothed_vag_resp)
        pl.axis([0,response_times[-1]+20,-1.5,1.5])
        pl.tight_layout()
        pl.xlabel('Time in seconds')
        if save:
            pl.savefig(path+j[:4]+'_vag_timecourse.pdf')

        # alternation frequencies origninal responses
        fig = pl.figure(no+43)  
        pl.subplot(4,1,exp_block[0][0]['block_no'] )
        pl.plot(orig_freq,orig_spAbs)
        pl.xlabel('Frequency in Hz')
        pl.ylabel('Power')
        pl.tight_layout()
        if save:
            pl.savefig(path+j[:4]+'_orig_frequencies.pdf')

        # alternation frequencies direction sensitive
        fig = pl.figure(no+44)  
        pl.subplot(4,1,exp_block[0][0]['block_no'] )
        pl.plot(dir_freq,dir_spAbs)
        pl.xlabel('Time in seconds',fontsize=18)
        pl.tight_layout()
        if save:
            pl.savefig(path+j[:4]+'_dir_frequencies.pdf')
        
        # plot time space version of alternation frequencies: How much time is needed for one entire cycle , original responses
        fig = pl.figure(no+45)  
        pl.subplot(4,1,exp_block[0][0]['block_no'] )
        pl.plot(orig_freq,orig_spAbs)
        pl.tight_layout()
        if save:
            pl.savefig(path+j[:4]+'_orig_CycleDuration.pdf')

        # plot time space version of alternation frequencies: How much time is needed for one entire cycle, direction sensitive responses
        fig = pl.figure(no+46)  
        pl.subplot(4,1,exp_block[0][0]['block_no'] )
        pl.plot(dir_freq,dir_spAbs)
        pl.xlabel('Time in seconds')
        pl.tight_layout()
        if save:
            pl.savefig(path+j[:4]+'_dir_CycleDuration.pdf')   

        # adapatation and stuff-->exp fiilterd timecourse, all responses, as they are
        fig = pl.figure(no+47)  
        pl.subplot(4,1,exp_block[0][0]['block_no'] )
        pl.plot(response_times, orig_exp_responses1,response_times, orig_exp_responses2)
        pl.xlabel('Time in seconds')
        pl.tight_layout()
        if save:
            pl.savefig(path+j[:4]+'orig_exp_filtered_timecourse.pdf')

        # adapatation and stuff-->exp fiilterd timecourse, responses vagueness degree pooled
        fig = pl.figure(no+48)  
        pl.subplot(4,1,exp_block[0][0]['block_no'] )
        pl.plot(response_times, dir_exp_responses1,response_times, dir_exp_responses2)
        pl.xlabel('Time in seconds')
        pl.tight_layout()
        if save:
            pl.savefig(path+j[:4]+'_direction_sensitive_exp_filtered_timecourse.pdf')
        
        # adapatation and stuff-->exp fiilterd timecourse, responses direction pooled
        fig = pl.figure(no+49)  
        pl.subplot(4,1,exp_block[0][0]['block_no'] )
        pl.plot(response_times, vag_exp_responses1,response_times,vag_exp_responses2)
        pl.xlabel('Time in seconds')
        pl.tight_layout()
        if save:
            pl.savefig(path+j[:4]+'_vaguenes_sensitive_exp_filtered_timecourse.pdf')
      
        #combined plots for vagueness and direction
        fig = pl.figure(no+50)  
        pl.subplot(4,1,exp_block[0][0]['block_no'] )
        pl.plot(response_times, dir_exp_responses1,response_times, dir_exp_responses2,response_times, vag_exp_responses1,response_times,vag_exp_responses2)
        pl.xlabel('Time in seconds')
        pl.tight_layout()
        if save:
            pl.savefig(path+j[:4]+'_combined_exp_filtered_timecourse.pdf')

        fig = pl.figure(no+51)
        pl.subplot(4,1,exp_block[0][0]['block_no'] )
        am,an = pl.plot(response_times,dir_adaptation[1],response_times,vag_adaptation[1])
        fig.legend((am,an),('Direction Sensitive', 'Vagueness Sensitive'), 'upper right',prop={'size':15})     
        pl.axis([0,response_times[-1]+20,-0.1,1.1])
        pl.xlabel('Time in seconds',fontsize=18)
        pl.ylabel('Adaptation (low-high)',fontsize=18)
        pl.tight_layout()
        if save:
            pl.savefig(path+j[:4]+'_adapatationCorrelation.pdf')

        if j[:6]== '01TK_1':
            fig = pl.figure(j[:6])
            am,an = pl.plot(response_times,dir_adaptation[1],response_times,vag_adaptation[1])
            fig.legend((am,an),('Direction Sensitive', 'Vagueness Sensitive'),bbox_to_anchor=(0.95, 0.01), loc='lower right', bbox_transform=pl.gcf().transFigure,prop={'size':15})     
            #fig.legend((org,sorg),('Original responses', 'Smoothed responses'),bbox_to_anchor=(0.95, 0.07),loc='lower right', bbox_transform=pl.gcf().transFigure,borderaxespad=0.,prop={'size':15})
            pl.axis([0,response_times[-1]+20,-0.1,1.1])
            pl.xlabel('Time in seconds',fontsize=18)
            pl.ylabel('Adaptation (high-low)',fontsize=18)
            pl.tight_layout()
            pl.subplots_adjust(top=0.86, bottom = 0.15)
            pl.title('Correlation of Perceptual Stabilization Timecourse and Strength of Percept',fontsize=15, fontweight = 'bold')

            pl.savefig(path[:-5]+j[:4]+'_example_adaptationCorrelation.pdf')
        
        if o == len(file_list)-1:
            ids = False
        else:
            ids = j[:4]
            save = False
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#####################################################################
PLOT AND SAVE RESULTS
#####################################################################
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print 'saving group plots'
#lists with mean and SD of perceptive blocks
outfile = open('/home/shared/AM_EEG/results/behavioral/results_per_subject.pickle','wb')
pickle.dump(results_per_subject,outfile) 
outfile.close()

#timecourse
fig = pl.figure(1)
fig.suptitle('Timecourse with original responses')
fig.legend((x1,y1), ('Real timecourse', ' original Smoothed timecourse'), 'upper right',prop={'size':15})
#pl.figure(1).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/orig_timecourse_all.pdf')
plt.close(fig)

fig = pl.figure(2)
fig.suptitle('Timecourse with original responses')
fig.legend((x2,y2), ('Real timecourse', 'direction Smoothed timecourse'), 'upper right',prop={'size':15})
#pl.figure(2).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/dir_timecourse_all.pdf')
plt.close(fig)

fig = pl.figure(3)
fig.suptitle('Timecourse with original responses')
fig.legend((x3,y3), ('Real timecourse', 'Vague timecourse (neg = vague, pos = definite'), 'upper right',prop={'size':15})
#pl.figure(3).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/vag_timecourse_all.pdf')
plt.close(fig)

#frequencies -orignal
fig = pl.figure(4)
fig.suptitle('Frequencies of perceptual alternations in Hz')
#pl.figure(4).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/orig_frequencies.pdf')
plt.close(fig)

#frequencies -direction
fig = pl.figure(5)
fig.suptitle('Frequencies of perceptual alternations in Hz')
#pl.figure(5).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/dir_frequencies.pdf')
plt.close(fig)

# check whether percept stability changes over time (sessions) for original
fig = pl.figure(6)
#fig.suptitle('For each block: Which freqs has highest power')
pl.figure(6).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/orig_peak_freq.pdf')
plt.close(fig)

# check whether percept stability changes over time (sessions) for direction
fig = pl.figure(7)
#fig.suptitle('For each block: Which freqs has highest power')
pl.figure(7).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/dir_peak_freq.pdf')
plt.close(fig)

# same as frequencies but in timespace orignal
fig = pl.figure(8)
fig.suptitle('Duration of  a perceptual cycle in s')
#pl.figure(8).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/orig_CycleDuration.pdf')
plt.close(fig)

# same as frequencies but in timespace direction
fig = pl.figure(9)
fig.suptitle('Duration of  a perceptual cycle in s')
#pl.figure(9).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/dir_CycleDuration.pdf')
plt.close(fig)


#exponentially filtered timecourse with original responses
fig = pl.figure(10)
fig.suptitle('Exponetially filtered timecourse with original responses')
fig.legend((aa,ab), ('One direction', 'Another direction'), 'upper right',prop={'size':15})
#pl.figure(10).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/exp_filtered_timecourse.pdf')
plt.close(fig)

#exponentially filtered timecourse with only direction sensitive responses
fig = pl.figure(11)
fig.suptitle('Exponetially filtered timecourse with direction sensitive responses')
fig.legend((ac,ad),('One direction', 'Another direction'), 'upper right',prop={'size':15})
#pl.figure(11).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/direction_sensitive_exp_filtered_timecourse.pdf')
plt.close(fig)

#exponentially filtered timecourse with only vagueness sensitive responses
fig = pl.figure(12)
fig.suptitle('Exponetially filtered timecourse with vagueness sensitive responses')
fig.legend((ae,af),  ('Vague', 'Definite'), 'upper right',prop={'size':15})
#pl.figure(12).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/vaguenes_sensitive_exp_filtered_timecourse.pdf')
plt.close(fig)

# combination of latter two plots in order to see what is going on during reversals
fig = pl.figure(13)
fig.suptitle('Combined Exponetially filtered timecourse for vagueness/direction sensitive responses')
fig.legend((ag,ah,ai,aj), ('One direction', 'Another direction','Vague', 'Definite'), 'upper right',prop={'size':15})
#pl.figure(13).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/combined_exp_filtered_timecourse.pdf')
plt.close(fig)

fig = pl.figure(14)
fig.suptitle('Timecourse of direction/vagueness sensitive adaptation')
fig.legend((am,an), ('Direction Sensitive', 'Vagueness Sensitive'), 'upper right',prop={'size':15})
#pl.figure(13).show()
pl.savefig('/home/shared/AM_EEG/results/behavioral/adaptationCorrelation.pdf')
plt.close(fig)