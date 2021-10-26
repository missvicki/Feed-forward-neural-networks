"""Binary classifier using PyTorch.

Patrick Wang, Vicki Nomwesigwa 2021
"""
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from gen_data import gen_simple, gen_xor


class FFNN(ABC):
    """Feed-forward neural network."""

    def __init__(self, learning_rate=1e-2, device='cpu'):
        """Initialize."""
        self.device = device
        self.learning_rate = learning_rate

        if self.device == 'cuda':
            self.net.cuda()

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), 
            lr=self.learning_rate,
        )

    def predict(self, X_t):
        """Predict."""
        return self.net(X_t)

    def update_network(self, y_hat, Y_t):
        """Update weights."""
        self.optimizer.zero_grad()
        loss = self.loss_func(y_hat, Y_t)
        loss.backward()
        self.optimizer.step()
        self.training_loss.append(loss.item())

    def calculate_accuracy(self, y_hat_class, Y):
        """Calculate accuracy."""
        return np.sum(Y.reshape(-1, 1) == y_hat_class) / len(Y)

    def train(self, X, Y, n_iters=1000):
        """Train network."""
        self.training_loss = []
        self.training_accuracy = []

        X_t = torch.FloatTensor(X).to(device=self.device)
        Y = Y.reshape(-1, 1)
        Y_t = torch.FloatTensor(Y).to(device=self.device)

        for _ in range(n_iters):
            y_hat = self.predict(X_t)
            self.update_network(y_hat, Y_t)
            y_hat_class = np.where(y_hat < 0.5, 0, 1)
            accuracy = self.calculate_accuracy(y_hat_class, Y)
            self.training_accuracy.append(accuracy)

    def plot_training_progress(self):
        """Plot training progress."""
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(self.training_loss)
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training Loss')

        ax[1].plot(self.training_accuracy)
        ax[1].set_ylabel('Classification Accuracy')
        ax[1].set_title('Training Accuracy')

        plt.tight_layout()
        plt.show()

    def plot_testing_results(self, X_test, Y_test):
        """Plot testing results."""
        X_t = torch.FloatTensor(X_test).to(device=self.device)
        y_hat_test = self.predict(X_t)
        y_hat_test_class = np.where(y_hat_test < 0.5, 0, 1)

        # Plot the decision boundary
        # Determine grid range in x and y directions
        x_min, x_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
        y_min, y_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1

        # Set grid spacing parameter
        spacing = min(x_max - x_min, y_max - y_min) / 100

        # Create grid
        XX, YY = np.meshgrid(
            np.arange(x_min, x_max, spacing),
            np.arange(y_min, y_max, spacing)
        )

        # Concatenate data to match input
        data = np.hstack((
            XX.ravel().reshape(-1, 1),
            YY.ravel().reshape(-1, 1),
        ))

        # Pass data to predict method
        data_t = torch.FloatTensor(data).to(device=self.device)
        db_prob = self.predict(data_t)

        clf = np.where(db_prob < 0.5, 0, 1)

        Z = clf.reshape(XX.shape)

        print("Test Accuracy {:.2f}%".format(
            self.calculate_accuracy(y_hat_test_class, Y_test) * 100)
        )

        plt.figure(figsize=(12, 8))
        plt.contourf(XX, YY, Z, cmap=plt.cm.RdYlBu, alpha=0.5)
        plt.scatter(
            X_test[:, 0], X_test[:, 1],
            c=Y_test,
            cmap=plt.cm.RdYlBu,
        )
        plt.show()

    def plot_training_results(self, X_train, Y_train):
        """Plot testing results."""
        X_t = torch.FloatTensor(X_train).to(device=self.device)
        y_hat_train = self.predict(X_t)
        y_hat_train_class = np.where(y_hat_train < 0.5, 0, 1)

        # Plot the decision boundary
        # Determine grid range in x and y directions
        x_min, x_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
        y_min, y_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1

        # Set grid spacing parameter
        spacing = min(x_max - x_min, y_max - y_min) / 100

        # Create grid
        XX, YY = np.meshgrid(
            np.arange(x_min, x_max, spacing),
            np.arange(y_min, y_max, spacing)
        )

        # Concatenate data to match input
        data = np.hstack((
            XX.ravel().reshape(-1, 1),
            YY.ravel().reshape(-1, 1),
        ))

        # Pass data to predict method
        data_t = torch.FloatTensor(data).to(device=self.device)
        db_prob = self.predict(data_t)

        clf = np.where(db_prob < 0.5, 0, 1)

        Z = clf.reshape(XX.shape)

        print("Train Accuracy {:.2f}%".format(
            self.calculate_accuracy(y_hat_train_class, Y_train) * 100)
        )

        plt.figure(figsize=(12, 8))
        plt.contourf(XX, YY, Z, cmap=plt.cm.RdYlBu, alpha=0.5)
        plt.scatter(
            X_train[:, 0], X_train[:, 1],
            c=Y_train,
            cmap=plt.cm.RdYlBu,
        )
        plt.show()


class BinaryLinear(FFNN):
    """Linear FFNN for binary classification."""
    def __init__(self, n_input, **kwargs):
        """Initialize."""
        self.n_input_dim = n_input
        self.n_output = 1

        self.net = nn.Sequential(
            nn.Linear(self.n_input_dim, 3),
            nn.ReLU(),
            nn.Linear(3, self.n_output),
            nn.Sigmoid(),
        )

        self.loss_func = nn.BCELoss()

        super().__init__(**kwargs)


def main():
    """Run experiment."""
    n_dims = 2
    X, Y = gen_xor(400)

    # Split into test and training data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.25,
    )

    net = BinaryLinear(n_dims)
    net.train(X_train, Y_train)
    net.plot_training_progress()
    net.plot_training_results(X_train, Y_train)
    net.plot_testing_results(X_test, Y_test)

    w1l1 = net.net[0].weight.detach().numpy()
    b1l1 = net.net[0].bias.detach().numpy()
    w2l2 = net.net[2].weight.detach().numpy()
    b2l2 = net.net[2].bias.detach().numpy()

    print(w1l1, b1l1,w2l2, b2l2)

    predictions = net.predict(torch.FloatTensor(X_test)).detach().numpy()
    print(predictions)


if __name__ == "__main__":
    main()
