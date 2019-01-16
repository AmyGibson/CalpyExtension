import numpy, csv, os
import matplotlib.pyplot as plt
from myUtilities import *

def process_audio(speaker_fname, min_pause, start_time=0, duration=0):
    """
        :param speaker_fname: (array of string) the audio file name, in the order for A, B, C
        :param min_pause: (float): The minimum duration in seconds to be considered pause.
        :param start_time: (float) in minutes, when the session of the audio to be analysed starts.
                            default to 0, starts from the beginning
        :param duration: (float) in minutes, how long should the session of the audio to be analysed for.
                            default to 0, the whole audio
        :return:
            sound_array: (array of array(float)) An array with each cell holds the sounding incidents for the input
                        audio files. each cell is a 0-1 1D numpy integer array with 1s marking sounds.
            sampling_rate: (int) in hz, assuming all files have the same sampling rate.
    """

    sound_array = []
    for i in range(len(speaker_fname)):
        full_filename = 'audio/threeSpeakers/' + speaker_fname[i] + '.wav'
        sound_file_name = os.path.join(os.path.dirname(__file__), full_filename)
        sampling_rate, original_signal = utilities.read_wavfile(sound_file_name)

        # calculate the section needed
        start_index = int(start_time * 60 * sampling_rate)

        if duration == 0:
            end_index = len(original_signal)
        else:
            end_index = int((start_time+duration) * 60 * sampling_rate)

        #original_signal = original_signal[start_index:end_index]

        original_signal_copy = copy.deepcopy(original_signal)  # just to compare before and after the processing

        # attempt to clean up the sound files, use as you see fit, see doctype for more details
        processed_sound = process_signal_amplify_sound_3speakers(original_signal)
        # make sure to extract the session after processing the whole audio for more stable results, as the processing
        # takes the volumne level of the whole audio into account
        processed_sound = processed_sound[start_index:end_index]

        pauses = convert_signal_to_pauses(processed_sound, sampling_rate, min_silence_duration=min_pause)
        # to find the incidents we only need the sounds array
        sound_array.append(numpy.logical_not(pauses))

        ### below are just for debugging, to visualise how the audio has been processed
        if False and i == 0:
            plt.figure()
            plt.subplot(311)
            plt.plot(original_signal_copy[start_index:end_index])  # unprossesed

            plt.subplot(312)
            plt.plot(processed_sound)  # prossesed

            plt.subplot(313)
            plt.plot(sound_array[0])  # 1 is pauses

            plt.show()
            break

    return sound_array, sampling_rate


def innerpause(sound_A, sound_B, sound_C):
    """
        finding inner pause among a 3-party conversation
        to be an inner pause, AipB and AipC have to be both true

        :param sound_A: numpy.array(float): 0-1 1D numpy integer array with 1s marking sounds.
        :param sound_B: numpy.array(float): 0-1 1D numpy integer array with 1s marking sounds.
        :param sound_C: numpy.array(float): 0-1 1D numpy integer array with 1s marking sounds.

        :return:
            res: numpy.array(bool): 0-1 1D numpy boolean array with trues marking A is having an inner pause.
    """

    conditions = ranges_satisfying_condition(sound_A, sound_B, *name_to_edge_condition["AipB"])
    AipB = numpy.zeros(shape=sound_A.shape)
    for L, R in conditions:
        AipB[L:R + 1] = 1

    conditions = ranges_satisfying_condition(sound_A, sound_C, *name_to_edge_condition["AipB"])
    AipC = numpy.zeros(shape=sound_A.shape)
    for L, R in conditions:
        AipC[L:R + 1] = 1

    res = numpy.zeros(shape=sound_A.shape)
    res[numpy.where(numpy.array(AipB, dtype=bool) & numpy.array(AipC, dtype=bool))] = 1
    res = numpy.asarray(res, dtype=bool)
    return res


def sounding_pattern_two_parties(sound_A, sound_B, codeList):
    """
        find the incidents of talking, uptakes, inner pauses, over takes, and overlap in a 2-party conversation
    :param sound_A: numpy.array(float): with 1 indicates sounding.
    :param sound_B: numpy.array(float): with 1 indicated sounding.
    :param codeList: array(string): a list of incident names to be identified

    :return:
        results: numpy.array(float): with each cell marked with the index of the incidents according to codeList
    """
    assert sound_A.shape == sound_B.shape

    results = numpy.zeros(shape=sound_A.shape)

    # talking comes first coz it can be both talking and overlap/overtake, and the later should be marked over talking
    results[numpy.where(sound_A)] = codeList.index('A talking')
    results[numpy.where(sound_B)] = codeList.index('B talking')

    # A uptake B
    conditions = ranges_satisfying_condition(sound_A, sound_B, *name_to_edge_condition["AupB"])
    for L, R in conditions:
        results[L:R + 1] = codeList.index('A uptake B')

    # B uptake A
    conditions = ranges_satisfying_condition(sound_B, sound_A, *name_to_edge_condition["AupB"])
    for L, R in conditions:
        results[L:R + 1] = codeList.index('B uptake A')

    # A inner pause
    conditions = ranges_satisfying_condition(sound_A, sound_B, *name_to_edge_condition["AipB"])
    for L, R in conditions:
        results[L:R + 1] = codeList.index('A inner pause')

    # B inner pause
    conditions = ranges_satisfying_condition(sound_B, sound_A, *name_to_edge_condition["AipB"])
    for L, R in conditions:
        results[L:R + 1] = codeList.index('B inner pause')

    # A overlap B
    conditions = ranges_satisfying_condition(sound_A, sound_B, *name_to_edge_condition["AfoB"])
    for L, R in conditions:
        results[L:R + 1] = codeList.index('A overlap B')

    # B overlap A
    conditions = ranges_satisfying_condition(sound_B, sound_A, *name_to_edge_condition["AfoB"])
    for L, R in conditions:
        results[L:R + 1] = codeList.index('B overlap A')

    # A overtake B
    conditions = ranges_satisfying_condition(sound_A, sound_B, *name_to_edge_condition["AsoB"])
    for L, R in conditions:
        results[L:R + 1] = codeList.index('A overtake B')

    # B overtake A
    conditions = ranges_satisfying_condition(sound_B, sound_A, *name_to_edge_condition["AsoB"])
    for L, R in conditions:
        results[L:R + 1] = codeList.index('B overtake A')

    return results


def sounding_pattern(sound_A, sound_B, sound_C, codeList):
    """
        find the incidents of talking, uptakes, inner pauses, over takes, and overlap in a 3-party conversation

        3-party overlaps are marked separately because to be A overlap BC, A is overlapping B and A is overlapping C, and
        B and C may also overlap each other at the same time. 3-party overlaps can only be marked when all overlaps
        have been processed.

        overtakes are calculated are marked separately is due to the complication of 3-party overtakes like A overtakes
        BC. There was not enough time for the definition of these situation to be resolved so the 3-party overtakes
        have been omitted in this implementation. At this stage, the overtakes can probably be merged back with the
        results. However I am leaving the separated implementation here in case this needs to be expanded in the future

    :param sound_A: numpy.array(float): with 1 indicates sounding.
    :param sound_B: numpy.array(float): with 1 indicated sounding.
    :param sound_C: numpy.array(float): with 1 indicated sounding.
    :param codeList: array(string): a list of incident names to be identified

    :return:
        results: numpy.array(float): with each cell marked with the index of the incidents of talking, inner pause,
                    uptake and 2-party overlap, according to codeList
        threeoverlap: numpy.array(float): with each cell marked with the index of the incidents of 3-party overlap,
                        according to codeList
        overtakes: numpy.array(float): with each cell marked with the index of the incidents of overtakes,
                        according to codeList
    """
    assert sound_A.shape == sound_B.shape == sound_C.shape

    results = numpy.zeros(shape=sound_A.shape)

    # talking comes first coz it can be both talking and overlap/overtake, and the later should be marked over talking
    results[numpy.where(sound_A)] = codeList.index('A talking')
    results[numpy.where(sound_B)] = codeList.index('B talking')
    results[numpy.where(sound_C)] = codeList.index('C talking')

    # A uptake B
    conditions = ranges_satisfying_condition(sound_A, sound_B, *name_to_edge_condition["AupB"])
    for L, R in conditions:
        if sum(sound_C[L:R+1]) == 0:
            results[L:R + 1] = codeList.index('A uptake B')

    # A uptake C
    conditions = ranges_satisfying_condition(sound_A, sound_C, *name_to_edge_condition["AupB"])
    for L, R in conditions:
        if sum(sound_B[L:R + 1]) == 0:
            results[L:R + 1] = codeList.index('A uptake C')

    # B uptake A
    conditions = ranges_satisfying_condition(sound_B, sound_A, *name_to_edge_condition["AupB"])
    for L, R in conditions:
        if sum(sound_C[L:R + 1]) == 0:
            results[L:R + 1] = codeList.index('B uptake A')

    # B uptake C
    conditions = ranges_satisfying_condition(sound_B, sound_C, * name_to_edge_condition["AupB"])
    for L, R in conditions:
        if sum(sound_A[L:R + 1]) == 0:
            results[L:R + 1] = codeList.index('B uptake C')

    # C uptake A
    conditions = ranges_satisfying_condition(sound_C, sound_A, * name_to_edge_condition["AupB"])
    for L, R in conditions:
        if sum(sound_B[L:R + 1]) == 0:
            results[L:R + 1] = codeList.index('C uptake A')

    # C uptake B
    conditions = ranges_satisfying_condition(sound_C, sound_B, * name_to_edge_condition["AupB"])
    for L, R in conditions:
        if sum(sound_A[L:R + 1]) == 0:
            results[L:R + 1] = codeList.index('C uptake B')

    # find the 3-party inner pause
    ip = innerpause(sound_A, sound_B, sound_C)
    results[numpy.where(ip)] = codeList.index('A inner pause')

    ip = innerpause(sound_B, sound_A, sound_C)
    results[numpy.where(ip)] = codeList.index('B inner pause')

    ip = innerpause(sound_C, sound_A, sound_B)
    results[numpy.where(ip)] = codeList.index('C inner pause')

    # calculating the overlaps
    # to be classified as a 3-party overlap, A has to be talking while B and C are both talking at the same time
    # Each case is tested against two parties first then considers the third party,
    # eg: is A overlapping B, if so, is C also talking at the same time (more than half of the overlapping during is
    # marked with sound)
    threeoverlap = numpy.zeros(shape=sound_A.shape)

    #A overlap B
    conditions = ranges_satisfying_condition(sound_A, sound_B, *name_to_edge_condition["AfoB"])
    for L, R in conditions:
        if sum(sound_C[L:R + 1]) > len(sound_C[L:R + 1])/2:
            threeoverlap[L:R + 1] = codeList.index('A overlap BC')
        else:
            results[L:R + 1] = codeList.index('A overlap B')

    #A overlap C
    conditions = ranges_satisfying_condition(sound_A, sound_C, *name_to_edge_condition["AfoB"])
    for L, R in conditions:
        if sum(sound_B[L:R + 1]) > len(sound_B[L:R + 1])/2:
            threeoverlap[L:R + 1] = codeList.index('A overlap BC')
        else:
            results[L:R + 1] = codeList.index('A overlap C')

    #B overlap A
    conditions = ranges_satisfying_condition(sound_B, sound_A, *name_to_edge_condition["AfoB"])
    for L, R in conditions:
        if sum(sound_C[L:R + 1]) > len(sound_C[L:R + 1])/2:
            threeoverlap[L:R + 1] = codeList.index('B overlap AC')
        else:
            results[L:R + 1] = codeList.index('B overlap A')

    #B overlap C
    conditions = ranges_satisfying_condition(sound_B, sound_C, *name_to_edge_condition["AfoB"])
    for L, R in conditions:
        if sum(sound_A[L:R + 1]) > len(sound_A[L:R + 1])/2:
            threeoverlap[L:R + 1] = codeList.index('B overlap AC')
        else:
            results[L:R + 1] = codeList.index('B overlap C')

    #C overlap A
    conditions = ranges_satisfying_condition(sound_C, sound_A, *name_to_edge_condition["AfoB"])
    for L, R in conditions:
        if sum(sound_B[L:R + 1]) > len(sound_B[L:R + 1])/2:
            threeoverlap[L:R + 1] = codeList.index('C overlap AB')
        else:
            results[L:R + 1] = codeList.index('C overlap A')

    #C overlap B
    conditions = ranges_satisfying_condition(sound_C, sound_B, *name_to_edge_condition["AfoB"])
    for L, R in conditions:
        if sum(sound_A[L:R + 1]) > len(sound_A[L:R + 1])/2:
            threeoverlap[L:R + 1] = codeList.index('C overlap AB')
        else:
            results[L:R + 1] = codeList.index('C overlap B')

    #[A ovterake B, A overtake C, B overtake A, B overtake C, C overtake A, C overtake B]
    # see general comment on why overtakes are stored separately
    overtakes = numpy.zeros(shape=(6, sound_A.shape[0]))

    #A overtake B
    conditions = ranges_satisfying_condition(sound_A, sound_B, *name_to_edge_condition["AsoB"])
    for L, R in conditions:
        overtakes[0, L:R + 1] = codeList.index('A overtake B')

    #A overtake C
    conditions = ranges_satisfying_condition(sound_A, sound_C, *name_to_edge_condition["AsoB"])
    for L, R in conditions:
        overtakes[1, L:R + 1] = codeList.index('A overtake C')

    #B overtake A
    conditions = ranges_satisfying_condition(sound_B, sound_A, *name_to_edge_condition["AsoB"])
    for L, R in conditions:
        overtakes[2, L:R + 1] = codeList.index('B overtake A')


    #B overtake C
    conditions = ranges_satisfying_condition(sound_B, sound_C, *name_to_edge_condition["AsoB"])
    for L, R in conditions:
        overtakes[3, L:R + 1] = codeList.index('B overtake C')

    #C overtake A
    conditions = ranges_satisfying_condition(sound_C, sound_A, *name_to_edge_condition["AsoB"])
    for L, R in conditions:
        overtakes[4, L:R + 1] = codeList.index('C overtake A')

    #C overtake B
    conditions = ranges_satisfying_condition(sound_C, sound_B, *name_to_edge_condition["AsoB"])
    for L, R in conditions:
        overtakes[5, L:R + 1] = codeList.index('C overtake B')

    return results, threeoverlap, overtakes


def process_signal_amplify_sound_3speakers(signal):
    """
    !!! warning !!! this modifies the data, use as you see fit
    The goal of modification is to amplify the speaking of the source party and slience the speaking of the other
    parties in the background
    based on the audios i have and based on how calpy idendtify sounds/silences
    I come up with this modification with the below step:

    1. cut off some max volume so the processing can focus on the middle to low range volume
    2. raise the signal to some power, effects: loud noise get louder and soft sounds
        gets relatively softer. this and the max volume depend on the actual audio file. I have some trial and error
        rules but are not generalised to all audio files
    3. normalisation. it was done in the original pause_profile code so i kept it

    probably dont need to return as python pass it by reference, just a habit from other programming language

    :param signal: (numpy.array(float)): Audio signal.
    :return: signal: numpy.array(float)): Processed audio signal.

    """

    # need to do more studies/test with more samples to determine the values
    max_signal = numpy.percentile(abs(signal), 96)  # get rid of 1% abnormalty 96
    print(str(max_signal) + ' at 96')
    if max_signal < 3000:
        max_signal = numpy.percentile(abs(signal), 98)
        print(str(max_signal) + ' at 98')

    signal[signal > max_signal] = max_signal
    signal[signal < -max_signal] = -max_signal

    pwr = 8 #magic number sorry, this number is not critical

    print('pwr ' + str(pwr))
    signal[signal > 0] = signal[signal > 0] ** pwr
    signal[signal < 0] = -(signal[signal < 0] ** pwr)

    signal = signal / max(abs(signal))  # normalisation

    return signal



def get_incident_length(src_incident, incident_index):
    """
    Compute the length of incidents in the number cells.

    :param src_incident: numpy.array(float): with numbers indicate certain incidents have occurred.
    :param incident_index: int: the index of the incidents to be identified

    :return: res (numpy array): The length of the incidents marked with incident_index.
             startIndex(numpy array): where each incident starts.
    """
    incidents = numpy.zeros(src_incident.shape)
    incidents[numpy.where(src_incident == incident_index)] = 1
    incidents = numpy.asanyarray(incidents, dtype=bool)

    res = []
    startIndex = []
    cnt = 0
    i = 0
    for incidents in incidents:
        if incidents:
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



def covert_indicient_dict(src_dict):
    """
    the src_dist is organised as the key being the incident name and the value is (incident_len[], start_index[])
    needs to convert a dictionary with key being the start index and value is (len, incident name)
    :param src_dict: dictionary (Key, Value): Key is incident name and Value is a tuple of (len[], index[])
    :return: res_dict: dictionary (Key, Value): Key is the start index for each incident that had happened
                        and Value is a tuple of (len, incident name) of that episode
    """
    res_dict = dict()
    for k, v in src_dict.items():
        len_array, index_array = v
        for i in range(len(len_array)):
            res_dict[index_array[i]] = (len_array[i], k)

    return res_dict


def write_incidents_to_csv(csv_file_name, incidents_dict, sampling_rate, start_time=0):
    """
    :param csv_file_name: (string) please include .csv
    :param incidents_dict: (dictionary) key: start index (int)
                value: (incident length in number of sample/cell, not in time (int), incident name(string))
    :param sampling_rate: (int)
    :param start_time: (float) in minutes, when the session of the audio to be analysed starts.
                            default to 0, starts from the beginning
    :return: null
    """
    start_index = int(start_time * 60 * sampling_rate)
    with open(csv_file_name, mode='w', newline='\n', encoding='utf-8') as pause_csv_file:
        pause_writer = csv.writer(pause_csv_file, delimiter=',')
        pause_writer.writerow(['Starts', 'Duration (ms)', 'Ends', 'Incident'])
        for key in sorted(incidents_dict):
            incident_len, incident_name = incidents_dict[key]
            start_time_in_s = index_to_sec(key + start_index, sampling_rate)
            p_duration_in_ms = duration_to_ms(incident_len, sampling_rate)
            end_time_in_s = start_time_in_s + p_duration_in_ms / 1000
            pause_writer.writerow([sec_to_min_str(start_time_in_s), \
                                   '%.2f' % p_duration_in_ms, \
                                   sec_to_min_str(end_time_in_s),\
                                   incident_name])


###################################
#
# Below are just code copied form the calpy library,
# they are here so in case the library was not installed properly
# or not accessible the code still work, and in case we need to modify these code
#
###################################


def integer_to_edge_condition(k):
    """Given a pattern id k, return left, middle, and right conditions

    Args:
        k (int): pattern id

    Returns:
        (Lcond, Mcond, Rcond) a tuple of three functions.
    """
    assert k < 63, "Input k = {} must be < 63.".format(k)

    bdigs = [int(bit) for bit in '{:06b}'.format(k)]

    Lcond = lambda x, y: x == bdigs[0] and y == bdigs[1]
    Rcond = lambda x, y: x == bdigs[4] and y == bdigs[5]

    # xs and ys here MUST be numpy arrays or the logical check will fail
    # Mcond = lambda xs, ys : numpy.all(xs==bdigs[2]) and numpy.all(ys==bdigs[3])

    Mcond = lambda xs, ys: numpy.logical_and(numpy.logical_xor(not bdigs[2], xs), numpy.logical_xor(not bdigs[3], ys))

    return (Lcond, Mcond, Rcond)


def ranges_satisfying_condition(A, B, Lcond, Mcond, Rcond):
    """Compute the left and right boundries of a particular pause pattern.

    Args:
        A (numpy.array(int)): 0-1 1D numpy integer array with 1s marking pause of speaker A.
        B (numpy.array(int)): 0-1 1D numpy integer array with 1s marking pause of speaker B.
        Lcond (function): bool, bool -> bool
        Mcond (function): numpy.array(bool), numpy.array(bool) -> numpy.array(bool)
        Rcond (function): bool, bool -> bool

    Returns:
        [(L, R)] a list of tuples that contains the left and right indeces in pause profiles that satifies Lcond and Mcond and Rcond
    """
    N = A.shape[0]
    middles = consecutive(numpy.where(Mcond(A, B) == True)[0])
    return [(L, R) for (L, R) in middles if
            L > 0 and R + 1 < N and Lcond(A[L - 1], B[L - 1]) and Rcond(A[R + 1], B[R + 1])]


# up = uptake
# so = successful overtake
# fo = failed overtake = overlap
# ip = inner pause
name_to_edge_condition = dict({
    "AupB": integer_to_edge_condition(18),
    "BupA": integer_to_edge_condition(33),
    "AsoB": integer_to_edge_condition(30),
    "AfoB": integer_to_edge_condition(29),
    "BsoA": integer_to_edge_condition(45),
    "BfoA": integer_to_edge_condition(46),
    "AipB": integer_to_edge_condition(34),
    "BipA": integer_to_edge_condition(17)
})




def consecutive(data, stepsize=1):
    """Find left and right indices of consecutive elements.

    Args:
        data (numpy.array(int)): 1D numpy.array

    Returns:
        [(L, R)] a list of tuples that contains the left and right indeces in data that breaks continuity.
    """
    if len(data)==0:
        return data

    runs = numpy.split(data, numpy.where(numpy.diff(data) != stepsize)[0]+1)
    return [ (run[0],run[-1]) for run in runs ]


