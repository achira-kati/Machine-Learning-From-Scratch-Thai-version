import matplotlib.pyplot as plt
import numpy as np


def ReLU(Z):
	return np.maximum(0, Z)

def softmax(z):
    max_z = np.max(z, axis=0)
    z_adjusted = z - max_z
    exp_values = np.exp(z_adjusted)
    sum_exp_values = np.sum(exp_values, axis=0)
    probabilities = exp_values / sum_exp_values
    return probabilities

def ReLU_deriv(Z):
	return Z > 0

def one_hot(Y):
	m = len(Y)
	one_hot_Y = np.zeros((10, m))
	for j in range(m):
		one_hot_Y[Y[j]][j] = 1 
	return one_hot_Y

def get_predictions(A2):
	return np.argmax(A2, 0)

class Nn:
	def __init__(self):
		self.W1 = np.random.rand(15, 400) - 0.5
		self.b1 = np.random.rand(15, 1) - 0.5
		self.W2 = np.random.rand(10, 15) - 0.5
		self.b2 = np.random.rand(10, 1) - 0.5
	
	def forward_prop(self, X):
		Z1 = np.dot(self.W1, X) + self.b1
		A1 = ReLU(Z1)
		Z2 = np.dot(self.W2, A1) + self.b2
		A2 = softmax(Z2)
		return Z1, A1, Z2, A2
	
	def backward_prop(self, Z1, A1, Z2, A2, X, Y, m, alpha):
		one_hot_Y = one_hot(Y)
		dZ2 = A2 - one_hot_Y
		dW2 = 1 / m * dZ2.dot(A1.T)
		db2 = 1 / m * np.sum(dZ2)
		dZ1 = np.dot(self.W2.T, dZ2) * ReLU_deriv(Z1)
		dW1 = 1 / m * dZ1.dot(X.T)
		db1 = 1 / m * np.sum(dZ1)

		self.W1 = self.W1 - alpha * dW1
		self.b1 = self.b1 - alpha * db1    
		self.W2 = self.W2 - alpha * dW2  
		self.b2 = self.b2 - alpha * db2    

	def get_accuracy(self, predictions, Y):
		print(predictions, Y)
		return np.sum(predictions == Y) / Y.size
	
	def gradient_descent(self, X, Y, alpha, iterations, m):
		for i in range(1, iterations+1):
			Z1, A1, Z2, A2 = self.forward_prop(X)
			self.backward_prop(Z1, A1, Z2, A2, X, Y, m, alpha)
			if i % 100 == 0 or i == iterations:
				print("Iteration: ", i)
				predictions = get_predictions(A2)
				print(f'Test Accuracy: {self.get_accuracy(predictions, Y):.2f}')

	def predict(self, X):
		_, _, _, A2 = self.forward_prop(X)
		predict = get_predictions(A2)
		return predict
	
	def test_train(self, index, X, Y):
		test = X[:, index, None] # 1 example
		predict = self.predict(test) # 1 output
		label = Y[index]
		image = test.reshape((20, 20))

		plt.imshow(image, interpolation='nearest', cmap='gray')
		plt.title(f'Predict: {predict} Label: {label}')
		plt.show()

	def test_predict(self, X_test):
		predict = self.predict(X_test)

		for i in range(10):
			image = X_test[:, i, None].reshape((28, 28)) * 255
			plt.imshow(image, interpolation='nearest', cmap='gray')
			plt.title(f'Predict on test data: {predict[i]}')
			plt.show()