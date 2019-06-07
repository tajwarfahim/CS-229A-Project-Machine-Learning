# author: Fahim Tajwar

import numpy as np
import math
from read_data_file import *

def svm_loss_and_gradient(dataset, W, reg):
    X  = dataset.get_X()
    y  = dataset.get_y()
    num_train = X.shape[0]
    scores_matrix = np.matmul(X, W)

    correct_class_scores = scores_matrix[np.arange(num_train), y].reshape(-1, 1)
    loss_matrix = np.maximum(0, scores_matrix - correct_class_scores + 1)
    loss_matrix[np.arange(num_train), y] = 0

    svm_loss = np.sum(loss_matrix)

    mask_matrix = loss_matrix
    mask_matrix[loss_matrix > 0] = 1
    row_sum  = np.sum(mask_matrix, axis = 1)
    mask_matrix[np.arange(num_train), y] = -row_sum

    gradient = np.matmul(np.transpose(X), mask_matrix)

    # averaging over number of data points
    gradient /= num_train
    svm_loss /= num_train

    # adding regularization
    gradient += reg * W
    svm_loss += 0.5 * reg * np.sum(np.square(W))

    return svm_loss, gradient
