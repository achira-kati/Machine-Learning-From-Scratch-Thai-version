import numpy as np


class LinearRegression:
	def __init__(self):
		self.W = 0
		self.b = 0
	
	def error(self, X, Y):
		m = len(X)
		f_wb = np.dot(self.W, X.T) + self.b
		return 1/(2*m) * np.sum((f_wb - Y)**2) 
	
	def fit(self, X, Y, alpha, n_iter):
		m = len(X)
		for i in range(1, n_iter+1):
			f_wb = np.dot(self.W, X.T) + self.b
			self.b = self.b - alpha*(1/m*np.sum(f_wb - Y))
			self.W = self.W - alpha*(1/m*np.sum(np.dot((f_wb - Y), X)))
			error = self.error(X, Y)
			if i%100==0 or i==n_iter:
				print(f'Iteration: {i} Error: {error}')

	def predict(self, X):
		f_wb = np.dot(self.W, X.T) + self.b
		return f_wb.T