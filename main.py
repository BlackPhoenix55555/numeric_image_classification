import os
import numpy as np
from PIL import Image

# -------------------- LOAD DATA --------------------
def load_data(folder):
    X, y = [], []
    for label in sorted(os.listdir(folder)):
        path = os.path.join(folder, label)
        if not os.path.isdir(path):
            continue
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            if os.path.isdir(img_path):
                continue

            img = Image.open(img_path).convert('L')   # grayscale
            img = img.resize((28, 28))                # ensure size
            img = np.array(img) / 255.0               # normalize

            X.append(img.flatten())                  # 784 vector
            y.append(int(label))

    return np.array(X), np.array(y)


X_train, y_train = load_data("./training")
X_test, y_test = load_data("./testing")

# -------------------- ONE HOT --------------------
def one_hot(y, num_classes=10):
    result = []
    for label in y:
        row = [0] * num_classes
        row[label] = 1
        result.append(row)
    return np.array(result)

y_train_oh = one_hot(y_train)

# -------------------- INIT WEIGHTS --------------------
W1 = np.random.randn(784, 196) / np.sqrt(784)
b1 = np.zeros(196)

W2 = np.random.randn(196, 49) / np.sqrt(196)
b2 = np.zeros(49)

W3 = np.random.randn(49, 10) / np.sqrt(49)
b3 = np.zeros(10)

# -------------------- HYPERPARAMETERS --------------------
lr = 0.05
batch = 32
epochs = 5

# -------------------- ACTIVATIONS --------------------
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / np.sum(ex, axis=1, keepdims=True)

# -------------------- TRAINING --------------------
for epoch in range(epochs):

    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train_oh = y_train_oh[indices]

    for i in range(0, len(X_train), batch):
        Xb = X_train[i:i+batch]
        yb = y_train_oh[i:i+batch]

        # Forward
        z1 = Xb @ W1 + b1
        a1 = relu(z1)

        z2 = a1 @ W2 + b2
        a2 = relu(z2)

        z3 = a2 @ W3 + b3
        out = softmax(z3)

        # Backprop
        grad = (out - yb) / len(Xb)

        dW3 = a2.T @ grad
        db3 = np.sum(grad, axis=0)

        grad = (grad @ W3.T) * relu_deriv(z2)
        dW2 = a1.T @ grad
        db2 = np.sum(grad, axis=0)

        grad = (grad @ W2.T) * relu_deriv(z1)
        dW1 = Xb.T @ grad
        db1 = np.sum(grad, axis=0)

        # Update
        W3 -= lr * dW3
        b3 -= lr * db3

        W2 -= lr * dW2
        b2 -= lr * db2

        W1 -= lr * dW1
        b1 -= lr * db1

    print(f"Epoch {epoch+1}/{epochs}")

# -------------------- PREDICT --------------------
def predict(X):
    a1 = relu(X @ W1 + b1)
    a2 = relu(a1 @ W2 + b2)
    out = softmax(a2 @ W3 + b3)
    return np.argmax(out, axis=1)

# -------------------- TEST ACCURACY --------------------
pred = predict(X_test)
acc = np.mean(pred == y_test) * 100
print(f"\nTest Accuracy: {acc:.2f}%")

# -------------------- PREDICT FROM IMAGE --------------------
def predict_from_path(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    x = np.array(img) / 255.0
    x = x.flatten().reshape(1, -1)

    a1 = relu(x @ W1 + b1)
    a2 = relu(a1 @ W2 + b2)
    out = softmax(a2 @ W3 + b3)

    pred_class = np.argmax(out)
    confidence = np.max(out)

    print("\n--- Prediction ---")
    print(f"Predicted Digit: {pred_class}")
    print(f"Confidence: {confidence*100:.2f}%")

    return pred_class
