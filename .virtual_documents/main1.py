import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def load_mnist_from_folder(folder_path):
    X = []
    y = []
    for label in sorted(os.listdir(folder_path)):
        label_folder = os.path.join(folder_path, label)
        for file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, file)
            img = Image.open(img_path).convert('L') 
            img_array = np.array(img).flatten()
            X.append(img_array)
            y.append(int(label))
    return np.array(X), np.array(y)


X_train, y_train = load_mnist_from_folder('./training')
X_test, y_test = load_mnist_from_folder('./testing')


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        m, n = X.shape
        self.num_classes = len(np.unique(y))
        self.W = np.zeros((n, self.num_classes))
        self.b = np.zeros(self.num_classes)

        y_one_hot = np.zeros((m, self.num_classes))
        y_one_hot[np.arange(m), y] = 1

        for _ in range(self.iterations):
            z = np.dot(X, self.W) + self.b
            y_pred = softmax(z)

            dw = (1/m) * np.dot(X.T, (y_pred - y_one_hot))
            db = (1/m) * np.sum(y_pred - y_one_hot, axis=0)

            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        y_pred = softmax(z)
        return np.argmax(y_pred, axis=1)


model = SoftmaxRegression(learning_rate=0.1, iterations=1000)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
accuracy = np.mean(predictions == y_test)
print("Test Accuracy:", accuracy)
