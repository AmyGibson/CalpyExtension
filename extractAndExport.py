import os, sys, copy, numpy
from myUtilities import *
from calpy.calpy import utilities
from calpy.calpy import plots
from calpy.calpy import students
import matplotlib.pyplot as plt

sys.path.append('..')

partial_fname = '016-1'
full_filename = 'audio/'+ partial_fname+'.wav'

sound_file_name = os.path.join(os.path.dirname(__file__), full_filename)# I have a folder called audio within this current folder
sampling_rate, full_len_sound = utilities.read_wavfile(sound_file_name)


min_pause = 0.15

original_signal = copy.deepcopy(full_len_sound) #just to compare before and after the processing

full_len_sound = process_signal_amplify_sound(full_len_sound) ##use as you see fit, see doctype for more details
pauses = convert_signal_to_pauses(full_len_sound, sampling_rate, min_silence_duration=min_pause)


### these are for graph labeling
labels = []
for i in range(int((full_len_sound.shape[0]/(sampling_rate*5)))+1):
    labels.append(str(i*5))
loc = numpy.arange(0,full_len_sound.shape[0],sampling_rate*5)


pause_length, pause_start_index = get_pause_length_modified(pauses) ## my version not the one from calpy
sound_length, sound_start_index = get_sound_length(pauses)

### below are to get rid of some short artifacts, i find it can be helpful in some cases, and
### at worse just not doing anything so quite harmless
compressed_pause_len, compressed_start_index = compress_pause_break(pause_length, pause_start_index, 0.01, sampling_rate)
compressed_pause = convert_to_pause(compressed_pause_len, compressed_start_index, len(pauses))
compressed_sound_length, compressed_sound_start_index = get_sound_length(compressed_pause)

### work in progress
#field_pause_len, field_pause_start_index = find_field_pause(pause_length, pause_start_index, sampling_rate, \
#                                                            max_field_pause_len=0.2, min_surrounding_len=0.5)
#field_pause = convert_to_pause(field_pause_len, field_pause_start_index, len(pauses))





#csv_file_name = partial_fname + '_minpause_'+str(min_pause) +'s_pauses_compressed.csv'
#write_to_csv(csv_file_name, 'Pause', compressed_pause_len, compressed_start_index, sampling_rate)

#csv_file_name = partial_fname + '_minpause_'+str(min_pause) +'s_sounds_compressed.csv'
#write_to_csv(csv_file_name, 'Pause', compressed_sound_length, compressed_sound_start_index, sampling_rate)

pause_length = compressed_pause_len
pause_start_index = compressed_start_index
sound_length = compressed_sound_length
sound_start_index = compressed_sound_start_index



csv_file_name = partial_fname + '_minpause_'+str(min_pause) +'s_pauses.csv'
write_to_csv(csv_file_name, 'Pause', pause_length, pause_start_index, sampling_rate)


csv_file_name = partial_fname + '_minpause_'+str(min_pause) +'s_sounds.csv'
write_to_csv(csv_file_name, 'Sound', sound_length, sound_start_index, sampling_rate)


#csv_file_name = partial_fname + '_minpause_'+str(min_pause) +'s_field_pauses.csv'
#write_to_csv(csv_file_name, 'Pause', field_pause_len, field_pause_start_index, sampling_rate)




plt.figure(figsize=(20, 10), dpi=150,)
plt.subplot(211)
plt.plot(original_signal) #unprossesed
plt.xticks(loc,labels)
plt.xlabel('sec')
plt.subplot(212)
plt.plot(compressed_pause) #1 is pauses
plt.xticks(loc,labels)
plt.xlabel('sec, 1s are pauses')
plt.savefig('{}.png'.format(partial_fname + '_minpause_'+str(min_pause) +'s_pauses'))

plt.show()


