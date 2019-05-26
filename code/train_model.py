import numpy as np
import math as math
from builtins import *
from read_data_file import *
from multinomial_logistic_regression import *

def get_training_batch(dataset, batch_size):
    if batch_size == "all":
        return dataset
    else:
        return dataset.get_batch(batch_size)

class Linear_Model:
    def __init__(self, model_type, dataset, weight_scale = 0.001):
        self.model_type = model_type
        self.dataset = dataset
        self.num_features = dataset.get_number_of_features()
        self.num_classes = np.max(dataset.get_y()) + 1
        self.num_datapoints = dataset.get_number_of_datapoints()
        self.W = weight_scale * np.random.randn(self.num_features, self.num_classes)

        if model_type == "logistic":
            self.loss_function = softmax_loss_and_gradient
        else:
            self.loss_function = None

    def train(self, learning_rate = 1e-3, reg = 1e-2,
            num_epochs = 100, batch_size = 200, verbose = False, num_print = 10,
            num_average = 10, decay = 1, start_new = True):

        if start_new:
            self.W = weight_scale * np.random.randn(self.num_features, self.num_classes)
            
        running_loss_history = []
        average_loss_history = []
        for it in range(num_epochs):
            batch = get_training_batch(self.dataset, batch_size)
            total_loss, gradient = self.loss_function(batch, self.W, reg)
            self.W -= learning_rate * gradient
            learning_rate *= decay
            running_loss_history.append(total_loss)

            if verbose and it % num_print == 0:
                print('iteration %d / %d , loss : %f' % (it, num_epochs, total_loss))

            if it % num_average == 0:
                average_loss_history.append(np.mean(np.array(running_loss_history)))
                running_loss_history = []

        return average_loss_history

    def predict(self, X):
        score_matrix = np.matmul(X, self.W)
        y_pred = np.argmax(score_matrix, axis = 1)

        return y_pred
