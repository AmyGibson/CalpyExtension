import sys, numpy
from incident_finder import *
from plot_incidents import *

sys.path.append('..')

speaker_fname = ['PWD003_CARER', 'PWD003_INTERVIEWER', 'PWD003_PWD'];
min_pause = 0.3 #0.15
start_time = 10 #in min
duration = 1 #in min

sound_array, sampling_rate = process_audio(speaker_fname, min_pause, start_time, duration)


codeList = ['all silence',
            'A talking', 'B talking', 'C talking',
            'A inner pause', 'B inner pause', 'C inner pause',
            'A uptake B', 'A uptake C',
            'B uptake A', 'B uptake C',
            'C uptake A', 'C uptake B',
            'A overlap B', 'A overlap C',
            'B overlap A', 'B overlap C',
            'C overlap A', 'C overlap B',
            'A overlap BC', 'B overlap AC', 'C overlap AB',
            'A overtake B', 'A overtake C',
            'B overtake A', 'B overtake C',
            'C overtake A', 'C overtake B',
            'A overtake BC', 'B overtake AC', 'C overtake AB']

results, threeoverlap, overtakes = sounding_pattern(sound_array[0], sound_array[1], sound_array[2], codeList)

incidents_dict = dict()

for i in range(codeList.index('A overlap BC')): #before A overlapBC
    incidents_length, incident_index = get_incident_length(results, i)
    if incidents_length is not []:
        incidents_dict[codeList[i]] = (incidents_length, incident_index)

for i in [codeList.index('A overlap BC'), codeList.index('B overlap AC'), codeList.index('C overlap AB')]:
    incidents_length, incident_index = get_incident_length(threeoverlap, i)
    if incidents_length is not []:
        incidents_dict[codeList[i]] = (incidents_length, incident_index)

for i in range(6):
    incidents_length, incident_index = get_incident_length(overtakes[i], i+codeList.index('A overtake B'))
    if incidents_length is not []:
        incidents_dict[codeList[i+codeList.index('A overtake B')]] = (incidents_length, incident_index)

incidents_dict = covert_indicient_dict(incidents_dict)

csv_file_name = speaker_fname[0] + '_minpause_'+str(min_pause) +'s_incidents_adjusted.csv'
write_incidents_to_csv(csv_file_name, incidents_dict, sampling_rate, start_time)





plot_incidents(incidents_dict, start_time, duration, sampling_rate, codeList)






