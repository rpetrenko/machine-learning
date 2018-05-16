import numpy as np
np.random.seed(seed=13)
def generate_random_xy(f, n, scale_x=1):
    x = scale_x * np.random.rand(n, 1) - scale_x/2.
    y = f(x) + np.random.randn(n, 1)
    return x, y
