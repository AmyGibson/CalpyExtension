import math, numpy, csv, copy
from calpy.calpy import dsp
from calpy.calpy import utilities
import matplotlib.pyplot as plt

def index_to_sec(ind, sampling_rate):
    return float(ind)/sampling_rate


def sec_to_min_str(sec):
    """
    :param sec: (float) second
    :return (str): in min:sec:ms format
    """
    minute = math.floor(sec/60)
    remainder_sec = math.floor(sec - minute * 60)
    remainder_ms = round((sec - minute * 60 - remainder_sec) * 1000)
    return str(minute) + ":" + str(remainder_sec) + ":" + '%03d' % remainder_ms


def duration_to_ms(dur, sampling_rate):
    return dur/(sampling_rate/1000.0)


def get_pause_length_modified(pauses):
    """Compute the length of pause.
        Args:
            pauses (numpy array, bool): True indicates occurrence of pause.

        Returns:
            res (numpy array): The length of consecutive pauses.
            startIndex(numpy array): where each pause starts.
    """
    res = []
    startIndex = []
    cnt = 0
    i = 0
    for pause in pauses:
        if pause:
            if cnt == 0:
                startIndex.append(i)
            cnt += 1
        elif cnt:
            res.append(cnt)
            cnt = 0
        i+=1
    if cnt:
        res.append(cnt)
    return numpy.array(res), numpy.array(startIndex)


def get_sound_length(pauses):
    """Compute the length of sound.
        Args:
            pauses (numpy array, bool): True indicates occurrence of pause.

        Returns:
            res (numpy array): The length of consecutive pauses.
            startIndex(numpy array): where each pause starts.
    """
    res = []
    startIndex = []
    cnt = 0
    i = 0
    for pause in pauses:
        if not pause:
            if cnt == 0:
                startIndex.append(i)
            cnt += 1
        elif cnt:
            res.append(cnt)
            cnt = 0
        i+=1
    if cnt:
        res.append(cnt)
    return numpy.array(res), numpy.array(startIndex)




def compress_pause_break(pause_length, start_index, min_pause_break, sampling_rate):
    """

    :param pause_length: (numpy array) The length of consecutive pauses, number of cells.
    :param start_index: (numpy array) the cell index where each pause starts.
    :param min_pause_break: (float) in sec, any pause break smaller are to be obmitted
    :param sampling_rate: (int) in hz
    :return:
        compressed_pause_len: (numpy array) The length of consecutive pauses, number of cells, without pause breaks.
        compressed_start_index: (numpy array) the cell index where each pause starts, without pause breaks.
    """
    compressed_pause_len = []
    compressed_start_index = []

    min_pause_break_no_cell = int (min_pause_break * sampling_rate)

    #for i in range(len(pause_length)-1):
    i = 0
    while i < len(pause_length)-1:
        combine_finished = False
        #print('i: ' + str(i))
        cur_i = i
        total_plen = pause_length[cur_i]
        init_start = start_index[i] #this should be the one that is appended
        cur_end_pause = start_index[cur_i] + pause_length[cur_i] - 1 #also counting the starting index
        till_the_end = False
        while not combine_finished:
            #print('combine_finished: ' + str(combine_finished))
            #print('cur_i: '+str(cur_i))
            start_next_pause = start_index[cur_i + 1]
            #print('start_next_pause: ' + str(start_next_pause))

            sound_start = cur_end_pause + 1
            sound_end = start_next_pause - 1
            sound_dur = sound_end - sound_start + 1

            #print('sound_start: ' + str(sound_start))
            #print('sound_end: ' + str(sound_end))
            #print('sound_dur: ' + str(sound_dur))

            if sound_dur < min_pause_break_no_cell: # combine the two pauses
                total_plen += sound_dur + pause_length[cur_i+1]
                cur_end_pause = init_start + total_plen - 1
                #print('total_plen: ' + str(total_plen))
                #print('cur_end_pause: ' + str(cur_end_pause))
                if cur_i + 1 < len(pause_length)-1:
                    cur_i += 1
                    continue
                else:
                    till_the_end = True
                    combine_finished = True
            else:
                combine_finished = True

            if combine_finished:
                #print('inside combine_finished: ' + str(combine_finished))
                i = cur_i + 1
                #print('inside i: ' + str(i))
                compressed_pause_len.append(total_plen)
                compressed_start_index.append(init_start)
               # break
    if not till_the_end:
        compressed_pause_len.append(pause_length[i])
        compressed_start_index.append(start_index[i])
    return compressed_pause_len, compressed_start_index

def convert_to_pause(pause_len, start_index, total_len):
    """
    converting the result of get_pause_length back to a 1D array of 1/0

    :param pause_len: (numpy array) duration of each pauses (in number of sample/cell, not in time)
    :param start_index: (numpy array) where each pause starts (in form of index, not in time)
    :param total_len: (int) the total length of the signal, len(signal)
    :return: pauses: (numpy array) of 1 x total_len, 1 indicate pauses

    """
    pauses = numpy.zeros(total_len)
    for i in range(len(start_index)):
        pauses[start_index[i]:start_index[i]+pause_len[i]] = 1

    return pauses

def write_to_csv(csv_file_name, title, feature_len, feature_start, sampling_rate):
    """

    :param csv_file_name: (string) please include .csv
    :param title: (string) for the title of the csv cells
    :param feature_len: (numpy array) duration of each feature(sound/pause etc) (in number of sample/cell, not in time)
    :param feature_start: (numpy array) where each feature starts (in form of index, not in time)
    :param sampling_rate: (int)
    :return: null
    """
    with open(csv_file_name, mode='w', newline='\n', encoding='utf-8') as pause_csv_file:
        pause_writer = csv.writer(pause_csv_file, delimiter=',')
        pause_writer.writerow([title +' starts', 'Duration (ms)', title + ' ends'])
        for i in range(len(feature_len)):
            plen = feature_len[i]
            pstart = feature_start[i]
            start_time_in_s = index_to_sec(pstart, sampling_rate)
            p_duration_in_ms = duration_to_ms(plen, sampling_rate)
            end_time_in_s = start_time_in_s + p_duration_in_ms / 1000
            pause_writer.writerow([sec_to_min_str(start_time_in_s), \
                                   '%.2f' % p_duration_in_ms, \
                                   sec_to_min_str(end_time_in_s)])





def process_signal_amplify_sound(signal):
    """
    !!! warning !!! this modifies the data, use as you see fit
    based on the audios i have and based on how calpy idendtify sounds/silences
    I come up with this modification with the below step:
    1. cut off the 1% abnormally loud sounds (good from some audio with interviewer)
    2. raise the signal to some power, effects: loud noise get louder and soft sounds
        gets relatively softer, however how much to raise it to the power depends on
        how loud the audio is, i have a very rough rule derived by trial and error so
        I am not at all certain this will works for all audios
        this method is inspired by the original calpy code, in _sound_or_silence()
        the original author use a similar method
    3. normalisation. it was done in the original pause_profile code so i kept it

    probably dont need to return as python pass it by refernce
    just a habbit from other programming language

    :param signal: (numpy.array(float)): Audio signal.
    :return: signal: numpy.array(float)): Audio signal.


    """
    #plt.figure()
    #plt.subplot(311)
    #plt.plot(signal)

    max_signal = numpy.percentile(abs(signal), 99)  # get rid of 1% abnormalty
    print(max_signal)

    signal[signal > max_signal] = max_signal
    signal[signal < -max_signal] = -max_signal

    #plt.subplot(312)
    #plt.plot(signal)

    pwr = 2
    if max_signal < 300:
        pwr = 8
    elif max_signal < 500:
        pwr = 6
    elif max_signal < 1000:
        pwr = 4
    elif max_signal < 2000:
        pwr = 4

    #pwr = 2
    print('pwr ' + str(pwr))
    signal[signal > 0] = signal[signal > 0] ** pwr
    signal[signal < 0] = -(signal[signal < 0] ** pwr)

    #plt.subplot(313)
    #plt.plot(signal)

    signal = signal / max(abs(signal))  # normalisation


    return signal



def convert_signal_to_pauses(signal, sampling_rate, min_silence_duration=0.01):
    """Find pauses in audio. modified from the calpy library function of pause_profile
        just removing the last step

        Args:
            signal (:obj:`numpy.array(float)`): Audio signal.
            sampling_rate (int): Sampling frequency in Hz.
            min_silence_duration (float, optional): The minimum duration in seconds to be considered pause. Default to 0.01.

        Returns:
            numpy.array(float): 0-1 1D numpy integer array with 1s marking PAUSES.
    """

    signal = dsp.audio_features._silence_or_sounding(signal)
    signal = numpy.array(signal)

    N = len(signal)
    ans = numpy.zeros(N)

    i, count, start, end = 0, 0, 0, 0

    T = min_silence_duration * sampling_rate
    for i in range(N):
        if signal[i] == 0:
            if i == N-1 and signal[i-1] == 0 and count >= T:
                ans[start:end] = 1
            elif count==0:
                start, end, count = i, i+1, count+1
            else:
                end, count = end+1, count+1
        elif i > 0 and signal[i-1] == 0:
            if count >= T:
                ans[start:end] = 1
            count = 0

    return ans


#ans2 = utilities.compress_pause_to_time(ans, sampling_rate, time_step=time_step, frame_window=frame_window)


#### work in progress
def find_field_pause(pause_length, start_index, sampling_rate, max_field_pause_len=0.2, min_surrounding_len=0.5):
    """

    :param pause_length: (numpy array) The length of consecutive pauses, number of cells.
    :param start_index: (numpy array) the cell index where each pause starts.
    :param max_field_pause_len: (float) in sec,
    :param min_surrounding_length: (float) in sec
    :param sampling_rate: (int) in hz
    :return:
        field_pause_len: (numpy array) The length of field pauses, number of cells.
        field_pause_index: (numpy array) the cell index where each field starts.
    """
    field_pause_len = []
    field_pause_index = []

    max_pause_break_no_cell = int(max_field_pause_len * sampling_rate)
    min_surrounding_no_cell = int(min_surrounding_len * sampling_rate)

    #for i in range(len(pause_length)-1):
    i = 0
    #while i < len(pause_length)-1:
    for i in range(len(pause_length)-1):
        combine_finished = False

        cur_i = i
        total_plen = pause_length[cur_i]
        init_start = start_index[i] #this should be the one that is appended
        cur_end_pause = start_index[cur_i] + pause_length[cur_i] - 1 #also counting the starting index
        till_the_end = False

        start_next_pause = start_index[cur_i + 1]
        sound_start = cur_end_pause + 1
        sound_end = start_next_pause - 1
        sound_dur = sound_end - sound_start + 1

        cur_pause_dur = pause_length[i]
        next_pause_dur = pause_length[i + 1]

        if sound_dur < max_pause_break_no_cell and \
                cur_pause_dur > min_surrounding_no_cell and \
                next_pause_dur > min_surrounding_no_cell:  # combine the two pauses
            field_pause_len.append(sound_dur)
            field_pause_index.append(sound_start)
    return field_pause_len, field_pause_index

    """
        while not combine_finished:

            start_next_pause = start_index[cur_i + 1]

            sound_start = cur_end_pause + 1
            sound_end = start_next_pause - 1
            sound_dur = sound_end - sound_start + 1

            cur_pause_dur = pause_length[i]
            next_pause_dur = pause_length[i+1]

            if sound_dur < max_pause_break_no_cell and \
                    cur_pause_dur > min_surrounding_no_cell and \
                    next_pause_dur > min_surrounding_no_cell: # combine the two pauses

            
            
                total_plen += sound_dur + pause_length[cur_i+1]
                cur_end_pause = init_start + total_plen - 1

                if cur_i + 1 < len(pause_length)-1:
                    cur_i += 1
                    continue
                else:
                   till_the_end = True
                    combine_finished = True
            else:
                combine_finished = True

            if combine_finished:                
                i = cur_i + 1
                compressed_pause_len.append(total_plen)
                compressed_start_index.append(init_start)
                break
    """
    """
    if not till_the_end:
        compressed_pause_len.append(pause_length[i])
        compressed_start_index.append(start_index[i])
    return compressed_pause_len, compressed_start_index
    """


