import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def download():
    mnist = datasets.fetch_openml('mnist_784', version=1)
    joblib.dump(mnist, "local/mnist_784.joblib", compress=9)


def train():
    mnist = joblib.load("local/mnist_784.joblib")

    mnist_data = mnist.data / 255.0
    mnist_label = mnist.target

    x_train, x_test, y_train, y_test = train_test_split(
        mnist_data, mnist_label)

    clf = RandomForestClassifier(
        n_estimators=200
    ).fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    score = accuracy_score(y_test, y_pred)

    print(score)
    # 0.9681142857142857


if __name__ == "__main__":
    download()
    train()
