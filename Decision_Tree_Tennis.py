import numpy as np

import pandas as pd

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.metrics import classification_report, confusion_matrix

# Read csv file

df=pd.read_csv("tennis_dataset.csv")

value=['Outlook','Temprature','Humidity','Wind']


from sklearn import preprocessing

string_to_int= preprocessing.LabelEncoder()

df=df.apply(string_to_int.fit_transform)

feature_cols = ['Outlook','Temprature','Humidity','Wind']

X = df[feature_cols ]

y = df['Play Tennis']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# perform training

from sklearn.tree import DecisionTreeClassifier # import the classifier

classifier =DecisionTreeClassifier(criterion="entropy", random_state=100) # create a classifier object

classifier.fit(X_train, y_train)

#Predict the response for test dataset

y_pred= classifier.predict(X_test)

# Accuracy

from sklearn.metrics import accuracy_score

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Confusion Matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred,zero_division=1))