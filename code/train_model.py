import numpy as np
import math as math
from builtins import *
from read_data_file import *
from multinomial_logistic_regression import *
from svm import *

def get_training_batch(dataset, batch_size):
    if batch_size == "all":
        return dataset
    else:
        return dataset.get_batch(batch_size)

def get_class_accuracy(particular_class, X, y, W):
    score_matrix = np.matmul(X, W)
    y_pred = np.argmax(score_matrix, axis = 1)
    total = 0
    total_correct = 0

    for i in range(X.shape[0]):
        if y[i] == particular_class:
            total += 1
            if y_pred[i] == particular_class:
                total_correct += 1

    class_accuracy = (1.0 * total_correct) / total
    return class_accuracy

class Linear_Model:
    def __init__(self, model_type, dataset, weight_scale = 0.001):
        self.model_type = model_type
        self.dataset = dataset
        self.num_features = dataset.get_number_of_features()
        self.num_classes = np.max(dataset.get_y()) + 1
        self.num_datapoints = dataset.get_number_of_datapoints()
        self.weight_scale = weight_scale
        self.W = self.weight_scale * np.random.randn(self.num_features, self.num_classes)

        if model_type == "logistic":
            self.loss_function = softmax_loss_and_gradient
        else:
            self.loss_function = svm_loss_and_gradient

    def train(self, learning_rate = 1e-3, reg = 1e-2,
            num_epochs = 100, batch_size = 200, verbose = False, num_print = 10,
            num_average = 10, decay = 1, start_new = True):

        if start_new:
            self.W = self.weight_scale * np.random.randn(self.num_features, self.num_classes)

        running_loss_history = []
        average_loss_history = []
        training_accuracies = []
        for it in range(num_epochs):
            training_accuracy = self.get_training_accuracy()

            batch = get_training_batch(self.dataset, batch_size)
            total_loss, gradient = self.loss_function(batch, self.W, reg)
            self.W -= learning_rate * gradient
            learning_rate *= decay
            running_loss_history.append(total_loss)

            if verbose and it % num_print == 0:
                print('iteration %d / %d , loss : %f, training accuracy : %f' % (it, num_epochs, total_loss, training_accuracy))

            if it % num_average == 0:
                average_loss_history.append(np.mean(np.array(running_loss_history)))
                running_loss_history = []
                training_accuracies.append(training_accuracy)

        return average_loss_history, training_accuracies, self.get_per_class_accuracy(self.dataset.get_X(), self.dataset.get_y())

    def predict(self, X_test):
        score_matrix = np.matmul(X_test, self.W)
        y_pred = np.argmax(score_matrix, axis = 1)
        return y_pred

    def get_accuracy(self, X_test, y_test):
        y_pred = self.predict(X_test)
        num_correct = np.sum(y_pred == y_test)
        return float(num_correct) / X_test.shape[0]

    def get_training_accuracy(self):
        return self.get_accuracy(self.dataset.get_X(), self.dataset.get_y())

    def get_per_class_accuracy(self, X_test, y_test):
        class_distribution = self.dataset.get_class_distribution()
        class_accuracies = {}
        for particular_class in class_distribution:
            class_accuracies[particular_class] = get_class_accuracy(particular_class, X_test, y_test, self.W)

        return class_accuracies
