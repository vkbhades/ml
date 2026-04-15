import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = {
    'Hours_Studied': [2, 4, 6, 8, 10, 3, 7, 5, 9, 1],
    'Attendance':    [50, 60, 55, 80, 90, 40, 85, 65, 95, 30],
    'Marks':         [40, 55, 50, 75, 85, 35, 80, 60, 90, 25]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied', 'Attendance']]
Y = df['Marks']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R2 Score:", r2_score(Y_test, Y_pred))

print("\nPredictions:")
for actual, pred in zip(Y_test, Y_pred):
    print(f"Actual: {actual}, Predicted: {pred:.2f}")