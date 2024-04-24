import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('car_data.csv')


X = data[['VOLUME', 'WEIGHT']].values.reshape(-1,2)
y = data['CO2'].values


model = LinearRegression()
model.fit(X, y)


def predict_co2(volume, weight):
    return model.predict([[volume, weight]])


volume_prompt = float(input("Enter the volume of the car (in cc): "))
weight_prompt = float(input("Enter the weight of the car (in kg): "))
predicted_co2 = predict_co2(volume_prompt, weight_prompt)
print(f"Predicted CO2 emissions: {predicted_co2[0]} g/km")
