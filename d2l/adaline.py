import numpy as np


class AdalineGD:
    """
    Attrs:
        w_: 1d-array
            Weights after fitting
        b_: scalar
            Bias unit after fitting

        errors_: list
            Number of mis-classification in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        ADAptive LInear NEuron classifier (Adaline)
        Args:
            eta: Learning rate (from 0.0 to 1.0)
            n_iter: passes over the training dataset
            random_state: Random number generator seed for random weight initialization
        """
        self.w_ = None
        self.b_ = None
        self.losses_ = None

        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data.

        :param X: array-like, shape=[n_examples, n_features]
            Training vectors, where n_examples is the number of examples and n_features
        :param y: array-lilke, shape=[n_examples]
            target values.
        :return: self
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # output =
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, X):
        """
        calculate the net input
        """
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """
        compute linear activation
        """
        return X

    def predict(self, X):
        """
        return class label after unit step
        """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
