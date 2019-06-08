# author: Fahim Tajwar

# import necessary modules
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
import sklearn
import numpy as np
import math as math
import seaborn as sns
from util import *


# helper functions
def get_predictions_and_target(model, test_loader, device = 'cpu'):
    predictions = []
    target = []

    for X, y in test_loader:
        target.append(y[0].item())
        X = X[0]
        X = Variable(X).to(device)
        output = model(X).to(device).detach().numpy()
        predicted = np.argmax(output)
        predictions.append(predicted)

    return np.array(predictions), np.array(target)

def get_score(predictions, target):
    total = 0
    correct = 0

    for i in range(predictions.shape[0]):
        total += 1
        if predictions[i] == target[i]:
            correct += 1

    return float(correct) / total

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0,1/np.sqrt(y))
        m.bias.data.fill_(0)

# custom dataset class for our neural net in pytorch
class Pytorch_Dataset(data.Dataset):
    # initializations
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        return self.X_data.shape[0]

    def __getitem__(self, index):
        X = self.X_data[index]
        y = self.y_data[index]
        return X, y

# Neural Network Models
class Net1(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layer_size = 3):
        super(Net1, self).__init__()

        fc_1 = nn.Linear(input_size, hidden_layer_size)
        fc_2 = nn.Linear(hidden_layer_size, num_classes)

        self.model = nn.Sequential(
            fc_1,
            nn.ReLU(),
            fc_2
        )

        self.model.apply(weights_init_normal)

    def forward(self, X):
        return self.model(X)


# our model trainer class, abstract class to train a model
class Model:
    def __init__(self, model, training_set, batch_size, learning_rate, weight = None, imbalanced_class = False, num_epochs = 25, verbose = True):
        self.model = model
        self.model.to(torch.double)
        self.training_set = training_set
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.imbalanced_class = imbalanced_class
        self.weight = weight

    def train(self, device = "cpu", print_it = False):
        model, learning_rate, num_epochs, verbose = self.model, self.learning_rate, self.num_epochs, self.verbose
        train_loader = torch.utils.data.DataLoader(dataset = self.training_set, batch_size = self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        if self.imbalanced_class:
            criterion = nn.CrossEntropyLoss(self.weight)

        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
        loss_history = []

        for epoch in range(num_epochs):
            running_loss = 0
            num_iters = 0
            for i, (X, y) in enumerate(train_loader):
                X = Variable(X).to(device)
                y = Variable(y).to(device)

                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.data
                num_iters += 1

            average_loss = float(running_loss) / num_iters
            if print_it:
                print ('Epoch: [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, average_loss))

            loss_history.append(average_loss)

        print("Training done!")
        self.model = model
        if verbose:
            x_axis = range(num_epochs)
            y_axis = loss_history
            plot_x_vs_y(x_axis, y_axis, x_title = "Training epoch", y_title= "Average Training Loss")


    def get_trained_model(self):
        return self.model

    def get_accuracy(self, test_loader, device = 'cpu'):
        predictions, target = get_predictions_and_target(self.model, test_loader, device)
        return get_score(predictions, target)

    def show_confusion_matrix(self, test_loader, device = 'cpu'):
        predictions, y_test = get_predictions_and_target(self.model, test_loader, device)
        cm = sklearn.metrics.confusion_matrix(y_test, predictions)
        plt.figure(figsize=(9,9))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Accuracy Score: {0}'.format(self.get_accuracy(test_loader, device))
        plt.title(all_sample_title, size = 15)

    def get_model_F1_score(self, test_loader, device = 'cpu'):
        predictions, y_test = get_predictions_and_target(self.model, test_loader, device)
        return get_F1_score(predictions, y_test)

    def get_accuracy_map_per_class(self, test_loader, device = 'cpu'):
        predictions, y_test = get_predictions_and_target(self.model, test_loader, device)
        return get_per_class_accuracy(predictions, y_test)

    def get_model_statistics(self, test_loader, device = 'cpu'):
        predictions, y_test = get_predictions_and_target(self.model, test_loader, device)
        show_vital_statistics(predictions, y_test)


# our abstract class to do the cross validation for us

class Hyperparameter_Tuner:
    def __init__(self, training_set, validation_loader, batch_size, input_size, num_classes, learning_rates, hidden_layer_sizes,
                weight = None, imbalanced_class = False, num_epochs = 100, device = "cpu"):

        assert(len(learning_rates) > 0)
        assert(len(hidden_layer_sizes) > 0)

        best_model = None
        best_validation_accuracy = float('-inf')
        best_hyperparameters = None

        for lr in learning_rates:
            for hd in hidden_layer_sizes:
                neural_network = Net1(input_size, num_classes, hd)
                model = Model(neural_network, batch_size, lr, weight = weight,
                            imbalanced_class = imbalanced_class, num_epochs = num_epochs, verbose = False)
                model.train()
                validation_accuracy = model.test(validation_loader, verbose = False)

                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_model = model
                    best_hyperparameters = (lr, hd)

        self.best_model = best_model
        self.best_hyperparameters = best_hyperparameters

    def get_best_model(self):
        return self.best_model

    def get_best_hyperparameters(self):
        return self.best_hyperparameters
