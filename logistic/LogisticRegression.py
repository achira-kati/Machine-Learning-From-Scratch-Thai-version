import numpy as np


class LogisticRegression:
    def __init__(self, w=0, b=0):
        self.w = w
        self.b = b

    def error(self, x, y):
        m = len(x)
        loss = 0
        f_wb = 1 / (1 + np.exp(-1 * (np.dot(x, self.w) + self.b)))
        loss = 1 / m * np.sum(-y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb))
        return loss

    def fit(self, x, y, alpha=0.003, n_iter=200000):
        m, n = x.shape
        self.w = np.zeros(n)
        for _ in range(1, n_iter + 1):
            f_wb = 1 / (1 + np.exp(-1 * (np.dot(x, self.w) + self.b)))
            self.b = self.b - alpha * (1 / m * np.sum(f_wb - y))
            for j in range(n):
                self.w[j] = self.w[j] - alpha * (
                    1 / m * np.sum(np.dot((f_wb - y), x[:, j]))
                )

            error = self.error(x, y)
            if _ % 100000 == 0 or _ == n_iter:
                print(f"Iterators {_}: Error {error:.2f}")
        print(f"Final w: {self.w} b: {self.b}")
        return self.w, self.b

    def predict(self, x):
        m = len(x)
        p = np.zeros(m)
        f_wb = 1 / (1 + np.exp(-1 * (np.dot(x, self.w) + self.b)))
        for i in range(m):
            if f_wb[i] >= 0.5:
                p[i] = 1
            else:
                p[i] = 0
        return p
