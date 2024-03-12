'''
Package that contains functions to process data
'''

import pandas as pd
from project_waste_sorter.params import *


def preprocess_labels(labels_list):
    ''' Converts the names of the labels in integer (category classes)'''
    labels_series = pd.Series(labels_list)
    categories_series = labels_series.map(CATEGORIES_MAP)
    return categories_series
