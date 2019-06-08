# author: Fahim Tajwar
# various functions, useful for other implementations
# cleanup was not done

import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import math
from read_data_file import *
import sklearn.metrics as metrics
import collections


def plot_x_vs_y(x_axis, y_axis, x_title = None, y_title = None):
    plt.plot(x_axis, y_axis)
    if x_title != None:
        plt.xlabel(x_title)
    if y_title != None:
        plt.ylabel(y_title)

    plt.show()

def show_plot(array, x_title = None, y_title = None):
    x_axis = range(len(array))
    y_axis = array

    plt.plot(x_axis, y_axis)
    if x_title != None:
        plt.xlabel(x_title)
    if y_title != None:
        plt.ylabel(y_title)

    plt.show()

# help taken from : https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.legend.html
def show_train_and_test_error(train_errors, test_errors, sample_sizes):
    fig, ax = plt.subplots()
    ax.plot(sample_sizes, train_errors, 'k--', label='Train error')
    ax.plot(sample_sizes, test_errors, 'k:', label='Test Error')

    legend = ax.legend(loc='lower left', shadow=True, fontsize='small')

    # Put a nicer background color on the legend.

    plt.xlabel("Training Dataset Size")
    plt.ylabel("Error")
    plt.show()


def visualize_dataset(split_dataset, datapoints_per_class, index_1, index_2, x_title = None, y_title = None, labels = None):
    groups = create_groups(split_dataset, datapoints_per_class, index_1, index_2)
    if len(split_dataset) == 3:
        colors = ['green', 'blue', 'red']
    else:
        colors = ['green', 'red']

    fig = plt.figure(figsize=(8, 8))

    for i in range(len(groups)):
        x_1, x_2 = groups[i]
        color_for_this_group = colors[i]

        if labels != None:
            plt.scatter(x_1, x_2, marker = '*', color = color_for_this_group, label = labels[i])
        else:
            plt.scatter(x_1, x_2, marker = '*', color = color_for_this_group)

    if x_title != None:
        plt.xlabel(x_title)
    if y_title != None:
        plt.ylabel(y_title)

    plt.show()


def create_groups(split_dataset, datapoints_per_class, index_1, index_2):
    groups = []
    for i in range(len(split_dataset)):
        X_i, _ = split_dataset[i]
        x_1 = X_i[0 : datapoints_per_class, index_1]
        x_2 = X_i[0 : datapoints_per_class, index_2]

        groups.append((x_1, x_2))

    return tuple(groups)

def plot_accuracy_per_class(map):
    plt.bar(map.keys(), map.values(), color='g')
    plt.xlabel("Class")
    plt.ylabel("Accuracy for that class")

def plot_bar_graph_from_map(map, x_label, y_label, label_for_each_class):
    plt.bar(range(len(map)), list(map.values()), align='center')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(range(len(map)), label_for_each_class)
    plt.show()

def draw_roc_curve(probability_vector, y, given_figsize = (8, 8), given_filename = None):
    fpr, tpr, threshold = metrics.roc_curve(y, probability_vector)
    roc_auc = metrics.auc(fpr, tpr)

    fig = plt.figure(figsize = given_figsize)
    ax = fig.add_subplot(111)
    plt.title('Receiver Operating Characteristic')
    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    if given_filename != None:
        plt.savefig(given_filename)
    else:
        plt.show()

def show_vital_statistics(predictions, target):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(predictions.shape[0]):
        if target[i] == 0:
            if predictions[i] == 0:
                true_negative += 1
            else:
                false_positive += 1

        else:
            if predictions[i] == 1:
                true_positive += 1
            else:
                false_negative += 1

    print("True Positive Rate: ", float(true_positive) / (true_positive + false_negative))
    print("True Negative Rate: ", float(true_negative) / (true_negative + false_positive))
    print("False Positive Rate: ", 1.0 - float(true_negative) / (true_negative + false_positive))
    print("False Negative Rate: ", 1.0 - float(true_positive) / (true_positive + false_negative))

    precision = float(true_positive) / (true_positive + false_positive)
    print("Precision : ", precision)

    recall = float(true_positive)/ (true_positive + false_negative)
    print("Recall : ", float(true_positive)/ (true_positive + false_negative))

    F1_score = 2.0 / (1.0 / precision + 1.0 / recall)
    print("F1 Score : ", F1_score)

    accuracy = float(true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    print("Accuracy : ", accuracy)

def get_F1_score(predictions, target):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(predictions.shape[0]):
        if target[i] == 0:
            if predictions[i] == 0:
                true_negative += 1
            else:
                false_positive += 1

        else:
            if predictions[i] == 1:
                true_positive += 1
            else:
                false_negative += 1


    precision = float(true_positive) / (true_positive + false_positive)
    recall = float(true_positive)/ (true_positive + false_negative)
    F1_score = 2.0 / (1.0 / precision + 1.0 / recall)

    return F1_score


def get_per_class_accuracy(predictions, target):
    total_map = collections.defaultdict(int)
    correct_map = collections.defaultdict(int)

    for i in range(predictions.shape[0]):
        total_map[target[i]] += 1
        if predictions[i] == target[i]:
            correct_map[target[i]] += 1

    accuracy_map = {}
    for i in total_map:
        accuracy_map[i] = float(correct_map[i]) / total_map[i]

    return accuracy_map

def get_random_sample(X, y, sample_size):
    X = np.array(X)
    y = np.array(y)
    N = X.shape[0]

    sample_indices = np.random.choice(N, sample_size, replace = False)
    return X[sample_indices], y[sample_indices]


def find_duplicates(X, y):
    X, y = np.array(X), np.array(y)
    class_map = collections.defaultdict(set)

    for i in range(X.shape[0]):
        X_i = tuple(list(X[i]))
        class_map[X_i].add(y[i])

    ambiguous = {}
    for i in class_map:
        if len(class_map[i]) > 1:
            ambiguous[i] = len(class_map[i])

    return ambiguous

def look_for_impossible_classification(X, y):
    ambiguous = find_duplicates(X, y)
    if len(ambiguous) > 0:
        print("Perfect classification based on this set of features is impossible")
        print("Total ambiguous data points : ", len(ambiguous))
        print(ambiguous)
    else:
        print("Building a better classifier should be possible")
