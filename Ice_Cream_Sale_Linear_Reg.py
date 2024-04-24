import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('ice_cream_sales.csv')


X = data['Temp(degree Celsius)'].values.reshape(-1,1);
y = data['Ice Cream Sale(In litres)'].values


model = LinearRegression()
model.fit(X, y)


def predict_sales(temperature):
    return model.predict([[temperature]])


temperature_prompt = int(input("Enter the temperature (in °C): "))
predicted_sales = predict_sales(temperature_prompt)
print(f"Predicted ice cream sales for {temperature_prompt}°C: {predicted_sales[0]} litres")
