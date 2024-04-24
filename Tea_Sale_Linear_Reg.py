import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('tea_sales.csv')


X = data['Temperature (°C)'].values.reshape(-1, 1)
y = data['Tea Sale (litres)'].values


model = LinearRegression()
model.fit(X, y)


def predict_sales(temperature):
    return model.predict([[temperature]])


temperature_prompt = int(input("Enter the temperature (in °C): "))
predicted_sales = predict_sales(temperature_prompt)
print(f"Predicted tea sales for {temperature_prompt}°C: {predicted_sales[0]} litres")
