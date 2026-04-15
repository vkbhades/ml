import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([1, 2, 3, 4, 5])
Y = np.array([0, 0, 0, 1, 1])

w = 0.0
b = 0.0
lr = 0.1

for i in range(1000):
    z = w * X + b
    y_pred = sigmoid(z)

    dw = np.mean((y_pred - Y) * X)
    db = np.mean(y_pred - Y)

    w -= lr * dw
    b -= lr * db

probabilities = sigmoid(w * X + b)
predicted_class = (probabilities >= 0.5).astype(int)

print("Predicted Probabilities:", probabilities)
print("Predicted Classes:", predicted_class)

x_values = np.linspace(0, 6, 100)
y_values = sigmoid(w * x_values + b)

plt.scatter(X, Y)
plt.plot(x_values, y_values)
plt.xlabel("Input Feature (X)")
plt.ylabel("Probability / Class")
plt.title("Logistic Regression Sigmoid Curve")
plt.show()