import pandas as pd
from sklearn.linear_model import LinearRegression


data = pd.read_csv('student_scores.csv')


x_train = data['hours_studied'].values.reshape(-1,1)
y_train = data['exam_scores'].values


model = LinearRegression()
model.fit(x_train, y_train)


def predict_score(hours):
    return model.predict([[hours]])


hours_prompt = int(input("Enter the number of hours studied: "))
predicted_score = predict_score(hours_prompt)
print(f"Predicted exam score based on {hours_prompt} hours studied: {predicted_score[0]}")