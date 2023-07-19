import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from LinearRegression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=11, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

print(f"X_test: {X_test.shape} X_train: {X_train.shape}")
print(f"y_test: {y_test.shape} y_train: {y_train.shape}")

model = LinearRegression(alpha=0.01)
iters, cost = model.fit(X_train, y_train, n_iters=1000)
yhat = model.predict(X)
print(
    f"Error in train set: {model.error(X_train, y_train):.2f} Error on test set: {model.error(X_test, y_test):.2f}"
)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(X_train, y_train, c="b", s=20)
ax1.scatter(X_test, y_test, c="r", marker="x")
ax1.plot(X, yhat, color="black")
ax2.plot(iters, cost)
plt.show()
