from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import LogisticRegression

mnist = fetch_openml('mnist_784', as_frame=True)

x, y = mnist["data"], mnist["target"]

some_digit = x.iloc[6000]
some_digit_image = some_digit.values.reshape(28, 28)

import matplotlib.pyplot as plt
plt.imshow(some_digit_image, cmap="binary")
plt.show()

x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

# Convert labels to int
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

# Convert to binary classification: Is digit == 9?
y_train_9 = (y_train == 6)
y_test_9 = (y_test == 6)

# ⭐ Scale the data (important!)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train logistic regression
clf = LogisticRegression(max_iter=2000, solver="lbfgs")
clf.fit(x_train_scaled, y_train_9)

# Predict that specific digit
prediction = clf.predict(scaler.transform([some_digit]))
prediction


from sklearn.model_selection import cross_val_score
cross_val_score(clf, x_train, y_train_9, cv=3, scoring="accuracy")


