import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

m, c = np.polyfit(x, y, 1)

print("Slope (m):", round(m, 5))
print("Intercept (c):", round(c, 5))

y_pred = m * x + c

print("Predicted Output:", y_pred)

plt.scatter(x, y)
plt.plot(x, y_pred)
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (Y)")
plt.title("Linear Regression Graph")
plt.show()
