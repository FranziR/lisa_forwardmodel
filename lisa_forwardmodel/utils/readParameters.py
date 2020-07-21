'''
Date: 20.07.2020
Author: Franziska Riegger
Revision Date:
Revision Author:
'''

import os
import sys

from math import pi, sin, cos


def read_parameter_fun(filepath, startline='#', delimiter=' = '):
    """
    Function reads in a text file and stores content in a dictionary.
    Note the mandatory structure of the .txt file:
        key = value

    :param filepath: (str)  path from where to read .txt file
    :param startline: (str) character that marks a new line
    :param delimiter: (str) delimiter between key and value

    :returns:
        data (dict):        data, read in from .txt file and stored as dictionary
    """
    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()

    with open(filepath) as fp:
        data = {}
        line = fp.readline()  # reads line
        cnt = 1
        while line:  # loops through all read in line
            if not (line[0] == startline or line[0] == '\n'):
                # if neither a comment (#) or an emtpy line, try to split line at delimiter
                data_temp = line.split(delimiter)
                try:
                    data[data_temp[0]] = eval(data_temp[-1].split('\n')[0])
                except NameError:
                    data[data_temp[0]] = data_temp[-1].split('\n')[0]
                cnt += 1
            line = fp.readline()
    fp.close()
    try:
        assert data['no_parameters'] == len(data.keys()) - 2, 'File {} contains {} instead of {} parameters!'.format(
            filepath, str(len(data.keys()) - 2), str(data['no_parameters']))
        return data
    except:
        return data


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)

    LISA_parameter_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'input_data',
                                       'LISA_parameters.txt')
    LISA_parameter = read_parameter_fun(LISA_parameter_file)

    GW_parameter_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'input_data',
                                     'GW_parameters.txt')
    GW_parameter = read_parameter_fun(GW_parameter_file)

    SIM_parameter_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'input_data',
                                      'Simulation_parameters.txt')
    SIM_parameter = read_parameter_fun(SIM_parameter_file)
    print(SIM_parameter)
