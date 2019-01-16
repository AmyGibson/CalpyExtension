import os, sys, copy, numpy
from myUtilities import *
from calpy.calpy import utilities
from calpy.calpy import plots
from calpy.calpy import students
import matplotlib.pyplot as plt
from incident_finder import *

sys.path.append('..')

speaker_fname = ['PWD003_INTERVIEWER', 'PWD003_PWD'];
min_pause = 0.3 #0.15
start_time = 9 #in min
duration = 5 #in min

sound_array, sampling_rate = process_audio(speaker_fname, min_pause, start_time, duration)


codeList = ['all silence',
            'A talking', 'B talking',
            'A inner pause', 'B inner pause',
            'A uptake B', 'B uptake A',
            'A overlap B', 'B overlap A',
            'A overtake B', 'B overtake A']

results = sounding_pattern_two_parties(sound_array[0], sound_array[1], codeList)

incidents_dict = dict()

for i in range(len(codeList)):
    incidents_length, incident_index = get_incident_length(results, i)
    if incidents_length is not []:
        incidents_dict[codeList[i]] = (incidents_length, incident_index)

incidents_dict = covert_indicient_dict(incidents_dict)

csv_file_name = speaker_fname[0] + '_minpause_'+str(min_pause) +'s_incidents_two_parties.csv'
write_incidents_to_csv(csv_file_name, incidents_dict, sampling_rate, start_time)












