# author: Fahim Tajwar
# help taken from : https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

import numpy as np
import math as math
from util import *
import sklearn as sklearn
from sklearn.linear_model import LogisticRegression
import seaborn as sns

class LogRegModel:
    def __init__(self, C_value = 1, given_class_weight = 'balanced', given_fit_intercept = True,
                multi_class_decision = 'multinomial', given_solver = 'newton-cg'):

        self.model = LogisticRegression(class_weight = given_class_weight, fit_intercept = given_fit_intercept,
                                        solver = given_solver, multi_class = multi_class_decision, C = C_value)

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

    def get_model_F1_score(self, X_test, y_test):
        return get_F1_score(self.predict(X_test), y_test)

    def get_accuracy_map_per_class(self, X_test, y_test):
        return get_per_class_accuracy(self.predict(X_test), y_test)


def validation_logistic(C_values, X_train, y_train, X_val, y_val, X_test, y_test, class_weight = 'balanced', fit_intercept = True,
                multi_class = 'multinomial', solver = 'newton-cg'):

    assert(len(C_values) > 0)

    f1_scores = []
    for C in C_values:
        model = LogRegModel(C_value = C, given_class_weight = class_weight, given_fit_intercept = fit_intercept, multi_class_decision = multi_class,
                            given_solver = solver)

        model.train(X_train, y_train)
        f1_score = model.get_model_F1_score(X_val, y_val)
        f1_scores.append(f1_score)

    f1_scores = np.array(f1_scores)
    best_f1_score = np.max(f1_scores)
    best_C = C_values[np.argmax(f1_scores)]
    print(best_C)
    print(best_f1_score)

    best_model = LogRegModel(C_value = best_C, given_class_weight = class_weight, given_fit_intercept = fit_intercept, multi_class_decision = multi_class,
                        given_solver = solver)
    best_model.train(X_train, y_train)

    print(best_model.get_accuracy(X_test, y_test))
    best_model.show_confusion_matrix(X_test, y_test)
    best_model.show_roc_curve(X_test, y_test)
    best_model.get_model_statistics(X_test, y_test)
