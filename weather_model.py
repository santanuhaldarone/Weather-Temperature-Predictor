import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

df = pd.read_csv('DailyDelhiClimateTrain.csv')

df = df.dropna()

X = df[['humidity', 'wind_speed', 'meanpressure']]
y = df['meantemp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ RMSE: {rmse:.2f}°C")

joblib.dump(model, 'weather_model.pkl')
