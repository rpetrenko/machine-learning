import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
#np.random.seed(seed=13)


def scale_data(x_train, x_test):
    sc = StandardScaler()
    sc.fit(x_train)
    return sc.transform(x_train), sc.transform(x_test)


def get_iris_data():
    iris = datasets.load_iris()
    return iris.data, iris.target


def generate_random_xy(f, n, scale_x=1):
    x = scale_x * np.random.rand(n, 1) - scale_x/2.
    y = f(x) + np.random.randn(n, 1)
    return x, y
