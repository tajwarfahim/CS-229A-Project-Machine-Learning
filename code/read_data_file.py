import pandas as pd
import numpy as np
import math as math
from builtins import *

# helper functions
def create_input_vector(dataFrame, row_index, num_cols):
    vector = []
    for col_index in range(num_cols - 1):
        vector.append(dataFrame.iloc[row_index, col_index])

    return vector

def find_min_of_map(map):
    min_index = None
    min_value = float('+inf')

    for i in map:
        if map[i] < min_value:
            min_value = map[i]
            min_index = i
    return min, min_value

def get_panda_dataframe(filename, file_type):
    dataFrame = None
    if file_type == "csv":
        dataFrame = pd.read_csv(filename)
    else:
        dataFrame = pd.read_excel(filename)

    return dataFrame

def read_input_features(filename, file_type):
    dataFrame = get_panda_dataframe(filename, file_type)

    X = []
    num_rows, num_cols = dataFrame.shape

    for i in range(num_rows):
        input_vector = create_input_vector(dataFrame, i, num_cols)
        X.append(input_vector)

    numpy_X = np.array(X)
    assert(numpy_X.shape == (num_rows, num_cols - 1))

    return numpy_X

def read_labels(filename, file_type):
    dataFrame = get_panda_dataframe(filename, file_type)

    y = []
    num_rows, num_cols = dataFrame.shape

    for i in range(num_rows):
        y.append(dataFrame.iloc[i, num_cols - 1])

    numpy_y = np.transpose(np.array(y))
    numpy_y -= 1

    assert(numpy_y.shape == (num_rows,))

    return numpy_y

def find_class_distribution(y):
    num_classes = np.max(y) + 1
    map = {}
    for i in range(num_classes):
        map[i] = np.sum(y == i)

    return map

def separate_dataset(X, y, num_classes):
    list = []
    for i in range(num_classes):
        X_i = []
        y_i = []

        for row in range(X.shape[0]):
            if y[row] == i:
                X_i.append(X[row])
                y_i.append(i)

        list.append((np.array(X_i), np.array(y_i)))

    return list


def get_random_sample(X, y, sample_size):
    rand_index = np.random.choice(X.shape[0], sample_size, replace = False)
    X_sample = X[rand_index]
    y_sample = y[rand_index]
    return X_sample, y_sample


def get_even_dataset(separated, num_datapoint_per_class):
    list = []
    for i in range(len(separated)):
        X_i, y_i = separated[i]
        X_sample, y_sample = get_random_sample(X_i, y_i, num_datapoint_per_class)
        list.append((X_sample, y_sample))

    return list

def compound_dataset(dataset):
    X = []
    y = []
    for i in range(len(dataset)):
        X_i, y_i = dataset[i]
        for row in range(X_i.shape[0]):
            X.append(X_i[row])
            y.append(y_i[row])

    return np.array(X), np.array(y)


def randomize_dataset(dataset):
    X, y = compound_dataset(dataset)
    num_datapoints = X.shape[0]
    random_index = np.random.choice(num_datapoints, num_datapoints, replace = False)
    return X[random_index], y[random_index]


def split_dataset(total_dataset, training_data_fraction, validation_data_fraction, test_data_fraction):
    X, y = total_dataset.get_X(), total_dataset.get_y()
    total_data_points = X.shape[0]

    random_index = random_index = np.random.choice(total_data_points, total_data_points, replace = False)

    num_train = math.floor(total_data_points * training_data_fraction)
    train_index = random_index[0 : num_train]
    training_X = X[train_index]
    training_y = y[train_index]
    training_dataset = Dataset(training_X, training_y)

    num_validation = math.floor(total_data_points * validation_data_fraction)
    validation_index = random_index[num_train : num_train + num_validation]
    validation_X = X[validation_index]
    validation_y = y[validation_index]
    validation_dataset = Dataset(validation_X, validation_y)

    num_test = math.floor(total_data_points * test_data_fraction)
    test_index = random_index[num_train + num_validation : num_train + num_validation + num_test]
    test_X = X[test_index]
    test_y = y[test_index]
    test_dataset = Dataset(test_X, test_y)

    return training_dataset, validation_dataset, test_dataset


class Dataset_Reader:
    def __init__(self, filename, file_type, X = None, Y = None):
        self.X = read_input_features(filename, file_type)
        self.y = read_labels(filename, file_type)

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

class Dataset:
    def __init__(self, X_data, y_data):
        self.X = X_data
        self.y = y_data
        self.class_distribution = find_class_distribution(self.y)

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def get_batch(self, batch_size):
        num_datapoints = self.X.shape[0]
        assert(num_datapoints >= batch_size)
        random_index = np.random.choice(num_datapoints, batch_size, replace = False)

        new_X = self.X[random_index]
        new_y = self.y[random_index]
        return Dataset(new_X, new_y)

    def get_number_of_datapoints(self):
        return self.X.shape[0]

    def get_number_of_features(self):
        return self.X.shape[1]

    def get_class_distribution(self):
        return self.class_distribution

class Dataset_Divider:
    def __init__(self, dataset_reader, total_data_points = "all", training_data_fraction = 0.6, validation_data_fraction = 0.2, test_data_fraction = 0.2):
        X = dataset_reader.get_X()
        y = dataset_reader.get_y()

        num_points = X.shape[0]

        if total_data_points != "all":
            rand_index = np.random.choice(num_points, total_data_points, replace = False)
            X = X[rand_index]
            y = y[rand_index]

        self.total_dataset = Dataset(X, y)
        self.training_dataset, self.validation_dataset, self.testing_dataset = split_dataset(self.total_dataset, training_data_fraction, validation_data_fraction, test_data_fraction)

    def get_training_dataset(self):
        return self.training_dataset

    def get_validation_dataset(self):
        return self.validation_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_total_dataset(self):
        return self.total_dataset

class Balanced_Class_Dataset_Divider:
    def __init__(self, dataset_reader, num_data = "all", training_data_fraction = 0.6, validation_data_fraction = 0.2, test_data_fraction = 0.2):
        dataset = Dataset(dataset_reader.get_X(), dataset_reader.get_y())
        class_distribution = dataset.get_class_distribution()
        min_index, min_number_of_datapoints = find_min_of_map(class_distribution)
        num_classes = np.max(dataset.get_y()) + 1

        self.separated = separate_dataset(dataset.get_X(), dataset.get_y(), num_classes)
        even_dataset = get_even_dataset(self.separated, min_number_of_datapoints)
        X, y = randomize_dataset(even_dataset)

        if num_data != "all":
            rand_index = np.random.choice(X.shape[0], num_data, replace = False)
            X = X[rand_index]
            y = y[rand_index]
        else:
            num_data = X.shape[0]

        self.total_dataset = Dataset(X, y)
        self.training_dataset, self.validation_dataset, self.test_dataset = split_dataset(self.total_dataset, training_data_fraction, validation_data_fraction, test_data_fraction)

    def get_training_dataset(self):
        return self.training_dataset

    def get_validation_dataset(self):
        return self.validation_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_total_dataset(self):
        return self.total_dataset

    def get_separated_dataset(self):
        return self.separated
