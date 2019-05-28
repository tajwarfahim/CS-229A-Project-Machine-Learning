import numpy as np
import math
from read_data_file import *

def softmax_loss_and_gradient(dataset, W, reg):
    X, y = dataset.get_X(), dataset.get_y()
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores_matrix = np.matmul(X, W)

    # taking care of numerical instability
    scores_matrix -= np.max(scores_matrix)

    exp_matrix = np.exp(scores_matrix)
    ratio_matrix = exp_matrix / (np.sum(exp_matrix, axis = 1).reshape(-1, 1))

    softmax_loss = -np.sum(np.log(ratio_matrix[np.arange(num_train), y]))

    ratio_matrix[np.arange(num_train), y] -= 1
    gradient = np.matmul(np.transpose(X), ratio_matrix)

    # averaging
    softmax_loss /= num_train
    gradient /= num_train

    # regularization
    softmax_loss += 0.5 * reg * np.sum(np.square(W))
    gradient += reg * W

    return softmax_loss, gradient
