import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_iris_classifier(C=30, seed=42, max_interations=250, softmax_binary_name='iris_classifier.bin'):
    iris = load_iris(as_frame=True)

    X = iris.data.values
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    softmax_regression = LogisticRegression(C=C, random_state=seed, max_iter=max_interations)
    softmax_regression.fit(X_train, y_train)

    # Evaluate the model
    y_pred = softmax_regression.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    with open(softmax_binary_name, 'wb') as binary_output:
        pickle.dump((softmax_regression, iris), binary_output)
        print(f"Model trained and saved to {softmax_binary_name}")


if __name__ == "__main__":
    train_iris_classifier()
