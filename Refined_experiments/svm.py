# author: Fahim Tajwar

import numpy as np
import math as math
from util import *
import sklearn as sklearn
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import seaborn as sns

class LinearSVMModel:
    def __init__(self, C_value = 1, given_class_weight = 'balanced', given_fit_intercept = True,
                multi_class_decision = 'ovr'):

        self.model = LinearSVC(class_weight = given_class_weight, fit_intercept = given_fit_intercept,
                                        multi_class = multi_class_decision, C = C_value, max_iter = 10000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_accuracy(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def show_confusion_matrix(self, X_test, y_test):
        predictions = self.predict(X_test)
        cm = sklearn.metrics.confusion_matrix(y_test, predictions)
        plt.figure(figsize=(9,9))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Accuracy Score: {0}'.format(self.get_accuracy(X_test, y_test))
        plt.title(all_sample_title, size = 15)

    def get_decision_function(self, X_test):
        return self.model.decision_function(X_test)

    def show_roc_curve(self, X_test, y_test):
        decision_function = self.get_decision_function(X_test)
        draw_roc_curve(decision_function, y_test)

    def get_model_statistics(self, X_test, y_test):
        show_vital_statistics(self.predict(X_test), y_test)

    def get_model_F1_score(self, X_test, y_test):
        return get_F1_score(self.predict(X_test), y_test)

    def get_accuracy_map_per_class(self, X_test, y_test):
        return get_per_class_accuracy(self.predict(X_test), y_test)


def validation_linear_SVM(C_values, X_train, y_train, X_val, y_val, X_test, y_test, class_weight = 'balanced', fit_intercept = True,
                multi_class = 'ovr'):

    assert(len(C_values) > 0)

    f1_scores = []
    for C in C_values:
        model = LinearSVMModel(C_value = C, given_class_weight = class_weight, given_fit_intercept = fit_intercept, multi_class_decision = multi_class)

        model.train(X_train, y_train)
        f1_score = model.get_model_F1_score(X_val, y_val)
        f1_scores.append(f1_score)

    f1_scores = np.array(f1_scores)
    best_f1_score = np.max(f1_scores)
    best_C = C_values[np.argmax(f1_scores)]
    print(best_C)
    print(best_f1_score)

    best_model = LinearSVMModel(C_value = best_C, given_class_weight = class_weight, given_fit_intercept = fit_intercept, multi_class_decision = multi_class)
    best_model.train(X_train, y_train)

    print(best_model.get_accuracy(X_test, y_test))
    best_model.show_confusion_matrix(X_test, y_test)
    best_model.show_roc_curve(X_test, y_test)
    best_model.get_model_statistics(X_test, y_test)


class SVMGaussian:
    def __init__(self, C_value = 1, given_class_weight = 'balanced'):
        self.model = SVC(kernel = 'rbf', class_weight = given_class_weight, C = C_value, max_iter = 10000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_accuracy(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def show_confusion_matrix(self, X_test, y_test):
        predictions = self.predict(X_test)
        cm = sklearn.metrics.confusion_matrix(y_test, predictions)
        plt.figure(figsize=(9,9))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Accuracy Score: {0}'.format(self.get_accuracy(X_test, y_test))
        plt.title(all_sample_title, size = 15)
        plt.show()

    def get_decision_function(self, X_test):
        return self.model.decision_function(X_test)

    def show_roc_curve(self, X_test, y_test):
        decision_function = self.get_decision_function(X_test)
        draw_roc_curve(decision_function, y_test)

    def get_model_statistics(self, X_test, y_test):
        show_vital_statistics(self.predict(X_test), y_test)

    def get_model_F1_score(self, X_test, y_test):
        return get_F1_score(self.predict(X_test), y_test)

    def get_accuracy_map_per_class(self, X_test, y_test):
        return get_per_class_accuracy(self.predict(X_test), y_test)


def validation_Gaussian_SVM(C_values, X_train, y_train, X_val, y_val, X_test, y_test, class_weight = 'balanced', multi_class  = False):

    assert(len(C_values) > 0)

    f1_scores = []
    for C in C_values:
        model = SVMGaussian(C_value = C, given_class_weight = class_weight)

        model.train(X_train, y_train)
        f1_score = model.get_model_F1_score(X_val, y_val)
        f1_scores.append(f1_score)

    f1_scores = np.array(f1_scores)
    best_f1_score = np.max(f1_scores)
    best_C = C_values[np.argmax(f1_scores)]
    print(best_C)
    print(best_f1_score)

    best_model = SVMGaussian(C_value = best_C, given_class_weight = class_weight)
    best_model.train(X_train, y_train)

    print(best_model.get_accuracy(X_test, y_test))
    best_model.show_confusion_matrix(X_test, y_test)

    if multi_class == False:
        best_model.show_roc_curve(X_test, y_test)
        best_model.get_model_statistics(X_test, y_test)


def analyze_bias_variance_issue_linear_SVM(X_train, y_train, X_test, y_test):
    train_errors = []
    test_errors = []
    sample_sizes = []
    num_trial = 5

    N = X_train.shape[0]

    for sample_size in range(1000, N, 100):
        sample_sizes.append(sample_size)
        train_error_sum = 0.0
        test_error_sum = 0.0

        for trial in range(num_trial):
            X_train_sample, y_train_sample = get_random_sample(X_train, y_train, sample_size)
            model = LinearSVMModel()
            model.train(X_train_sample, y_train_sample)

            train_error = 1.0 - model.get_accuracy(X_train_sample, y_train_sample)
            test_error = 1.0 - model.get_accuracy(X_test, y_test)

            train_error_sum += train_error
            test_error_sum += test_error

        average_train_error = float(train_error_sum) / num_trial
        average_test_error = float(test_error_sum) / num_trial

        train_errors.append(average_train_error)
        test_errors.append(average_test_error)

    show_train_and_test_error(train_errors, test_errors, sample_sizes)


def analyze_bias_variance_issue_gaussian_svm(X_train, y_train, X_test, y_test):
    train_errors = []
    test_errors = []
    sample_sizes = []
    num_trial = 5

    N = X_train.shape[0]

    for sample_size in range(1000, N, 100):
        sample_sizes.append(sample_size)
        train_error_sum = 0.0
        test_error_sum = 0.0

        for trial in range(num_trial):
            X_train_sample, y_train_sample = get_random_sample(X_train, y_train, sample_size)
            model = SVMGaussian()
            model.train(X_train_sample, y_train_sample)

            train_error = 1.0 - model.get_accuracy(X_train_sample, y_train_sample)
            test_error = 1.0 - model.get_accuracy(X_test, y_test)

            train_error_sum += train_error
            test_error_sum += test_error

        average_train_error = float(train_error_sum) / num_trial
        average_test_error = float(test_error_sum) / num_trial

        train_errors.append(average_train_error)
        test_errors.append(average_test_error)

    show_train_and_test_error(train_errors, test_errors, sample_sizes)
