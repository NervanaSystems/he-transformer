import numpy as np

file_dir = './exp_runtimes/'
suffix = '.txt'


def get_runtime_from_file(filename):
    try:
        lines = [line.strip() for line in open(filename).readlines()]

        time_parse = 'Total time'

        for line in lines:
            if time_parse in line:
                split_line = line.split(' ')
                time_ms = split_line[-3]
                return int(time_ms) / 1000.
    except Exception as e:
        print("Error: Could not parse file")
        return


def get_runtimes(config_string):
    times = []
    for i in range(1, 11):
        TT_file = file_dir + 'exp_' + str(i) + '_' + config_string + suffix
        time_ms = get_runtime_from_file(TT_file)

        if time_ms:
            times.append(time_ms)

    print(config_string, times)
    assert (len(times) == 10)
    print(
        str(int(np.round(np.mean(times), 0))) + '$\pm$',
        str(int(np.round(np.std(times), 0))) + '\\\\')


get_runtimes('TT')
get_runtimes('TF')
get_runtimes('FT')
get_runtimes('FF')
