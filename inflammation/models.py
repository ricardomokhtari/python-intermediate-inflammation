"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2d array) where each row contains
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np
from functools import reduce

def attach_names(data, names):
    """attach names to data
    :param data: 2D data
    :param names: list of names
    :returns: list of dicts

    """
    output = []

    for data_row, name in zip(data, names):
        output.append({'name': name,
                       'data': data_row})

    return output

def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    :returns: numpy array
    """
    return np.loadtxt(fname=filename, delimiter=',')

def patient_normalise(data):
    """
    Normalise patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.

    Negative values are rounded to 0.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data should be a numpy array')
    if len(data.shape) != 2:
        raise ValueError('data should be a 2D numpy array')
    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')
    max_data = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max_data[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised

def daily_above_threshold(data, patient_num, threshold):
    above_threshold = map(lambda x: x > threshold, data[patient_num])
    return reduce(lambda a, b: a + 1 if b else a, above_threshold, 0)

def daily_mean(data):
    """Calculate the daily mean of a 2d inflammation data array.

    :param data: 2d numpy array
    :returns: mean of data on axis 0
    """
    return np.mean(data, axis=0)

def daily_std(data):
    """
    Calculate the daily std of a 2d inflammation data array.

    :param data: 2d numpy array
    :returns: std of data on axis 0
    """
    return np.std(data, axis=0)

def daily_max(data):
    """Calculate the daily max of a 2d inflammation data array.

    :param data: 2d numpy array
    :returns: daily max of data on axis 0
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2d inflammation data array.

    :param data: 2d numpy array
    :returns: min of data on axis 0
    """
    return np.min(data, axis=0)

