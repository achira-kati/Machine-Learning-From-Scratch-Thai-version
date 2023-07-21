import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=50, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

model = LinearRegression()
model.fit(X_train, y_train, 0.001, 1000)
predict = model.predict(X)


plt.scatter(X_train, y_train,c='b')
plt.scatter(X_test, y_test,c='r')
plt.plot(X, predict, color='black')
plt.show()