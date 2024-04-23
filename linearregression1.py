import pandas as pd
from sklearn.linear_model import LinearRegression

# Read the CSV dataset
data = pd.read_csv('ex1.csv')

# Extracting features (hours studied) and target (exam scores)
x_train = data['hours_studied'].values.reshape(-1,1)
y_train = data['exam_scores'].values


# Initialize and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Function to predict exam score based on hours studied
# def predict_score(hours):
#     return model.predict([[hours]])

# Example usage
hours_prompt = int(input("Enter the number of hours studied: "))
predicted_score = model.predict([[hours_prompt]])
print(f"Predicted exam score based on {hours_prompt} hours studied: {predicted_score[0]}")