import matplotlib.pyplot as plt
import utils_log as ut
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

X, y = ut.load_data("F:/Documents/ML/logistic/data/data.txt")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234
)

model = LogisticRegression()
error, n_iters, w, b = model.fit(X_train, y_train, 0.001, 500000)

predict = model.predict(X_test)
print(f'Accuracy on test data: {model.get_accuracy(predict, y_test)}')


ut.plot_decision_boundary(w, b, X, y)
plt.show()