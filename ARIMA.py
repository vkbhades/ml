import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

data = [100, 120, 130, 140, 150, 160, 170, 180, 200, 210]

df = pd.Series(data)

model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=3)

print("Forecasted Values:")
print(forecast)

plt.plot(df, label='Original')
plt.plot(range(len(df), len(df) + 3), forecast, label='Forecast', color='red')
plt.legend()
plt.show()
