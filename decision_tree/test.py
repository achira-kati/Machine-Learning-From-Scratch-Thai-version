import numpy as np
from DecisionTree import DecisionTree

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])


root = [i for i in range(len(X_train))]
model = DecisionTree()
model.fit(X_train, y_train, root, 2, 0, 'Root')