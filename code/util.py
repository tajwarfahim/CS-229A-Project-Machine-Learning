import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import math
from read_data_file import *

def show_plot(array, x_title = None, y_title = None):
    x_axis = range(len(array))
    y_axis = array

    plt.plot(x_axis, y_axis)
    if x_title != None:
        plt.xlabel(x_title)
    if y_title != None:
        plt.ylabel(y_title)

    plt.show()

def visualize_dataset(split_dataset, datapoints_per_class, index_1, index_2):
    groups = create_groups(split_dataset, datapoints_per_class, index_1, index_2)
    if len(split_dataset) == 3:
        colors = ['green', 'blue', 'red']
    else:
        colors = ['green', 'red']

    fig = plt.figure(figsize=(8, 8))

    for i in range(len(groups)):
        x_1, x_2 = groups[i]
        color_for_this_group = colors[i]

        plt.scatter(x_1, x_2, marker = '*', color = color_for_this_group)

    plt.show()


def create_groups(split_dataset, datapoints_per_class, index_1, index_2):
    groups = []
    for i in range(len(split_dataset)):
        X_i, _ = split_dataset[i]
        x_1 = X_i[0 : datapoints_per_class, index_1]
        x_2 = X_i[0 : datapoints_per_class, index_2]

        groups.append((x_1, x_2))

    return tuple(groups)
