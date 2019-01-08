import numpy as np
from utils import utils
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MyPerceptron(object):
    """
    Perceptron classifier
    """

    def __init__(self, eta0=0.01, max_iter=50, random_state=1):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.random_state = random_state
        self.w_ = None
        self.errors_ = None

    def net_input(self, X):
        """
        y_hat = w * X + b
        :param X:
        :return:
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def fit(self, X, y):
        """

        :param X: shape = (n_samples, n_features)
        :param y: shape = (n_samples)
        :return:
        """
        rgen = np.random.RandomState(self.random_state)
        # generate random weigths from normal distribution
        size = X.shape[1] + 1 # number of features + bias
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=size)
        self.errors_ = []

        for _ in range(self.max_iter):
            errors = 0
            print(self.w_)
            for xi, target in zip(X, y):
                update = self.eta0 * (target - self.predict(xi))
                # update params
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self


if __name__ == "__main__":
    X, y = utils.get_iris_data()
    # leave only first two classes
    X, y = X[:100, :], y[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train_sc, X_test_sc = utils.scale_data(X_train, X_test)
    # ppn = Perceptron(eta0=0.1, max_iter=3, random_state=1)
    ppn = MyPerceptron(eta0=0.1, max_iter=3, random_state=1)
    ppn.fit(X_train_sc, y_train)
    y_pred = ppn.predict(X_test_sc)
    print(y_pred)
    print(y_test)
    print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

