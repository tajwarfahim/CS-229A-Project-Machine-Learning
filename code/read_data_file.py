import pandas as pd
import numpy as np
import math as math
from builtins import *

def create_input_vector(dataFrame, row_index, num_cols):
    vector = []
    for col_index in range(num_cols - 1):
        vector.append(dataFrame.iloc[row_index, col_index])

    return vector

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

        random_index = None
        if total_data_points == "all":
            random_index = np.random.choice(num_points, num_points, replace = False)
            total_data_points = num_points
        else:
            random_index = np.random.choice(total_data_points, total_data_points, replace = False)

        num_train = math.floor(total_data_points * training_data_fraction)
        train_index = random_index[0 : num_train]
        training_X = X[train_index]
        training_y = y[train_index]
        self.training_dataset = Dataset(training_X, training_y)

        num_validation = math.floor(total_data_points * validation_data_fraction)
        validation_index = random_index[num_train : num_train + num_validation]
        validation_X = X[validation_index]
        validation_y = y[validation_index]
        self.validation_dataset = Dataset(validation_X, validation_y)

        num_test = math.floor(total_data_points * test_data_fraction)
        test_index = random_index[num_train + num_validation : num_train + num_validation + num_test]
        test_X = X[test_index]
        test_y = y[test_index]
        self.test_dataset = Dataset(test_X, test_y)

    def get_training_dataset(self):
        return self.training_dataset

    def get_validation_dataset(self):
        return self.validation_dataset

    def get_test_dataset(self):
        return self.test_dataset
