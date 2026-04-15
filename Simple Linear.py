import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = {
    'Hours_Studied': [1, 2, 3, 4, 5],
    'Marks': [35, 40, 50, 60, 70]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied']]
Y = df['Marks']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(X, Y, color='red')
plt.plot(X, model.predict(X), color='blue')

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Simple Linear Regression")
plt.show()