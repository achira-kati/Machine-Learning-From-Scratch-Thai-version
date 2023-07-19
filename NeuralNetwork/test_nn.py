import numpy as np
import pandas as pd

from NeuralNetwork import Nn

# data = pd.read_csv('F:/Documents/ML/NeuralNetwork/data/train.csv')
# test = pd.read_csv('F:/Documents/ML/NeuralNetwork/data/test.csv')
# data = np.array(data)
# test = np.array(test)

# m, n = data.shape

# data_train = data.T
# X_train = data_train[1:n]
# Y_train = data_train[0]
# X_train = X_train / 255.

# X_test = test.T
# X_test = X_test / 255.

# nn = Nn()
# nn.gradient_descent(X_train, Y_train, 0.3, 500, m)

# nn.test_predict(X_test)

X = np.load('F:/Documents/ML/NeuralNetwork/data/X.npy')
y = np.load('F:/Documents/ML/NeuralNetwork/data/y.npy')

m, n = X.shape
X_cv = X[:1000].T 
y_cv = y[:1000].reshape(1000)

X_train = X[1000:].T 
y_train = y[1000:].reshape(m - 1000)

print(X_cv.shape)
print(y_cv.shape)
print(X_train.shape)
print(y_train.shape)


nn = Nn()
nn.gradient_descent(X_train, y_train, 0.1, 500, m)

nn.test_train(0, X_train, y_train)
nn.test_train(1, X_train, y_train)
nn.test_train(2, X_train, y_train)
nn.test_train(3, X_train, y_train)
nn.test_train(4, X_train, y_train)
nn.test_train(5, X_train, y_train)
nn.test_train(6, X_train, y_train)
nn.test_train(7, X_train, y_train)
nn.test_train(8, X_train, y_train)
nn.test_train(9, X_train, y_train)

