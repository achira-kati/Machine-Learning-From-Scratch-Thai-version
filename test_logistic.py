from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
import utils_log as ut

X, y = ut.load_data("ex2data1.txt")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234
)

model = LogisticRegression()
w, b = model.fit(X_train, y_train)
p = model.predict(X_test)
print(
    f"Error on train: {model.error(X_train, y_train):.2f} Error on test: {model.error(X_test, y_test):.2f}"
)
ut.plot_decision_boundary(w, b, X_test, p)
plt.show()
