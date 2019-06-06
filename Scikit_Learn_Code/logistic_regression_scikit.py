# author: Fahim Tajwar
# help taken from : https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

import numpy as np
import math as math
from util import *
import sklearn as sklearn
from sklearn.linear_model import LogisticRegression
import seaborn as sns

class LogRegModel:
    def __init__(self, given_class_weight = 'balanced', given_fit_intercept = True, multi_class_decision = 'multinomial', given_solver = 'newton-cg'):
        self.model = LogisticRegression(class_weight = given_class_weight, fit_intercept = given_fit_intercept,
                                        solver = given_solver, multi_class = multi_class_decision)

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

    def get_probability_vector(self, X_test):
        return self.model.predict_proba(X_test)

    def show_roc_curve(self, X_test, y_test):
        probability_vector = self.get_probability_vector(X_test)
        draw_roc_curve(probability_vector[:, 1], y_test)

    def get_model_statistics(self, X_test, y_test):
        show_vital_statistics(self.predict(X_test), y_test)
