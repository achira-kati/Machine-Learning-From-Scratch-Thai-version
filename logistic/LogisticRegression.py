import numpy as np


class LogisticRegression:
	def __init__(self):
		self.W = 0
		self.b = 0
	
	def error(self, X, y):
		m = len(X)
		f_wb = 1 / (1 + np.exp(-1 * (np.dot(self.W, X.T) + self.b)))
		epsilon = 1e-8
		loss = (-y * np.log(f_wb + epsilon) )- (1 - y) * np.log(1 - f_wb + epsilon)
		return 1/m * np.sum(loss)
	def fit(self, X, y, alpha, n_iters):
		m, n = X.shape
		self.W = np.zeros(n)
		total_error = []
		for i in range(1, n_iters+1):
			f_wb = 1 / (1 + np.exp(-1 * (np.dot(self.W, X.T) + self.b)))
			self.b = self.b - alpha * (1/m * np.sum(f_wb - y))

			for j in range(n):
				self.W[j] = self.W[j] - alpha * (1/m * np.sum(np.dot((f_wb - y), X[:, j])))
			error = self.error(X, y)
			total_error.append(error)
			if i%100==0 or i==n_iters:
				print(f'Iteration: {i} Error: {error}')
		return total_error, [i for i in range(n_iters)], self.W, self.b
	
	def predict(self, X):
		m = len(X)
		p = np.zeros(m)
		f_wb = 1 / (1 + np.exp(-1 * (np.dot(self.W, X.T) + self.b)))
		for i in range(m):
			if f_wb[i] >= 0.5:
				p[i] = 1
			else:
				p[i] = 0
		return p
	
	def get_accuracy(self, predict, y):
		return np.sum(predict == y) / len(y)