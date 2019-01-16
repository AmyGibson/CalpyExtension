import numpy, datetime, matplotlib
import matplotlib.pyplot as plt
from myUtilities import *
from matplotlib.colors import LinearSegmentedColormap
from random import sample





# heatmap = numpy.load('heatmap.npy')
# #print (sum(map(sum, heatmap)))
# plt.figure()
# fig, ax = plt.subplots()
#
# plt.imshow(heatmap, aspect='auto')
# plt.show()

def plot_incidents(incident_dict, start_time, duration, sample_rate, codeList):


    duration_per_row = duration*60 #second

    num_subplot_rows = round(duration*60 / duration_per_row)

    row_width = 10
    row_height = 1
    xtickevery = 10 * sample_rate


    overall_start_index = start_time * 60 * sample_rate




    for plot_row in range(num_subplot_rows):

        heatmap = numpy.zeros(shape=(len(codeList), int(duration_per_row * sample_rate)))
        last_index = (plot_row + 1) * int(duration_per_row * sample_rate)

        for k, v in incident_dict.items():
            if k < (plot_row) * int(duration_per_row * sample_rate) or \
                k >  last_index:
                continue;

            #k = index, v = (len, incident name)
            start_index = k - plot_row * int(duration_per_row * sample_rate)
            end_index = start_index + v[0]
            heatmap[codeList.index(v[1]), start_index:end_index] = codeList.index(v[1])
            if (v[1] == 'A overlap B' or v[1] == 'A overlap C' or v[1] == 'A overtake B' or v[1] == 'A overtake C' or \
                    v[1] == 'A overlap BC' or v[1] == 'C overlap A' or v[1] == 'B overlap A' or v[1] == 'B overlap AC' \
                    or v[1] == 'C overlap AB'):
                heatmap[codeList.index('A talking'), start_index:end_index] = codeList.index('A talking')
            if (v[1] == 'B overlap A' or v[1] == 'B overlap C' or v[1] == 'B overtake A' or v[1] == 'B overtake C' or \
                    v[1] == 'B overlap AC' or v[1] == 'A overlap B' or v[1] == 'C overlap B' or v[1] == 'A overlap BC' \
                    or v[1] == 'C overlap AB'):
                heatmap[codeList.index('B talking'), start_index:end_index] = codeList.index('B talking')
            if (v[1] == 'C overlap A' or v[1] == 'C overlap B' or v[1] == 'C overtake A' or v[1] == 'C overtake B' or \
                    v[1] == 'C overlap AB' or v[1] == 'A overlap C' or v[1] == 'B overlap C' or v[1] == 'B overlap AC' \
                    or v[1] == 'A overlap BC'):
                heatmap[codeList.index('C talking'), start_index:end_index] = codeList.index('C talking')

    numpy.save('heatmap.npy', heatmap)
    #heatmap = numpy.load('heatmap.npy')

    #ax = plt.subplot(num_subplot_rows, 1, plot_row + 1)

    fig, ax = plt.subplots()
    # Set y coordinate axis
    ax.set_yticks(range(len(codeList)))
    ax.set_yticklabels(codeList)

    plot_row = 0
    x_range = range(0, int(duration_per_row * sample_rate), xtickevery)
    ax.set_xticks(x_range)
    # # Convert seconds to HR:MIN:SEC format
    #
    to_time_format = lambda x: str(
         sec_to_min_str(index_to_sec(x + plot_row * duration_per_row * sample_rate + overall_start_index, sample_rate)))
    ax.set_xticklabels([to_time_format(x) for x in x_range])

    #my_cmap = matplotlib.cm.get_cmap('tab10')
    #my_cmap = sample(my_cmap, len(codeList))
    #my_cmap.set_under('w')

    my_cmap = matplotlib.cm.get_cmap('tab10').colors
    my_cmap = [list(row) for row in my_cmap]
    print(len(my_cmap))
    my_cmap = my_cmap + my_cmap
    print(len(my_cmap))
    my_cmap = my_cmap + my_cmap
    print(len(my_cmap))
    my_cmap = sample(my_cmap, len(codeList))
    # my_cmap.set_under('w')
    my_cmap = [(1, 1, 1)] + my_cmap

    plt.imshow(heatmap, cmap=LinearSegmentedColormap.from_list('mycmap', [*my_cmap], N=len(codeList)), aspect='auto',
               vmin=.001)



    #plt.imshow(heatmap, cmap=my_cmap, aspect='auto', vmin=.001)

    #plt.imshow(heatmap, cmap=LinearSegmentedColormap.from_list('mycmap', [*colours], N=len(colours) ), aspect='auto')

    # plt.margins(1)
    #plt.tight_layout()
    #plt.savefig('{}.png'.format('threespeaker'))
    plt.show()


