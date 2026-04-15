import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

n = len(x)

x_mean = sum(x) / n
y_mean = sum(y) / n

num = 0
den = 0

for i in range(n):
    num += (x[i] - x_mean) * (y[i] - y_mean)
    den += (x[i] - x_mean) ** 2

m = num / den
c = y_mean - m * x_mean

print("Slope (m):", m)
print("Intercept (c):", c)

y_line = [m * xi + c for xi in x]

x_new = 6
y_pred = m * x_new + c

print("Predicted value for x = 6:", y_pred)

plt.scatter(x, y, color='blue', label="Actual Data")
plt.plot(x, y_line, color='red', label="Regression Line")
plt.scatter(x_new, y_pred, color='green', label="Prediction (x=6)")

plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (Y)")
plt.title("Linear Regression Graph")
plt.legend()
plt.show()