import numpy as np


class LinearRegression:
    def __init__(self, w=0, b=0, alpha=0.001):
        self.w = w
        self.b = b
        self.alpha = alpha

    def error(self, x, y):
        m, n = x.shape
        y = y.reshape(m, n)
        f_wb = np.dot(x, self.w) + self.b
        j_wb = 1 / (2 * m) * np.sum((f_wb - y) ** 2)
        return j_wb

    def fit(self, x, y, n_iters=1000):
        m, n = x.shape
        y = y.reshape(m, n)
        cost = []
        for i in range(1, n_iters+1):
            f_wb = np.dot(x, self.w) + self.b
            self.b = self.b - self.alpha * (1 / m * np.sum(f_wb - y))
            self.w = self.w - self.alpha * (1 / m * np.sum(np.dot(x.T, (f_wb - y))))
            error = self.error(x, y)
            cost.append(error)
            if i % 20 == 0 or i==n_iters:
                print(f"Iterators {i} Error: {error:.2f}")
        print(f"Final w: {self.w:.2f} b: {self.b:.2f}")
        return [i for i in range(n_iters)], cost

    def predict(self, x):
        yhat = np.dot(x, self.w) + self.b
        return yhat
