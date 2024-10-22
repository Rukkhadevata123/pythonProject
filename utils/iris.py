from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def iris():
    iris_test = load_iris()
    model = DecisionTreeClassifier()

    X_train, X_test, y_train, y_test = train_test_split(iris_test.data, iris_test.target, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
