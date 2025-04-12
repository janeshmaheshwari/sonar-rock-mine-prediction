# Importing the Dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Data Processing

# Loading the DataSet to a Pandas DataFrame
sonar_data = pd.read_csv('sonar.csv', header=None)

# print(sonar_data.head()) # First 5 rows of our DataSet

# print(sonar_data.shape) # (208 -> Data, 61 -> Fetaures)

# print(sonar_data.describe()) # Describe Statistical Measures of Data

# print(sonar_data[60].value_counts()) # At 60th Column, count Rock and Mine

# print(sonar_data.groupby(60).mean())

# Seperating Data and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# print(X)
# print(Y)

# Training and Test Data :-
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# print(X.shape, X_train.shape, X_test.shape)

# Model Training --> Logistic Regression :-
model = LogisticRegression()

# Training the model with Training Data
model.fit(X_train, Y_train)

# Model Evaluation :-

# Accuracy on Training Data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# print(training_data_accuracy)

# Accuracy on Test Data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# print(test_data_accuracy)

# Making a Predective System :-

input_data = (0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)

# Changing the input_data into a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
# print(prediction)

if(prediction[0] == 'R'):
    print("The object is a Rock")
else:
    print("The object is a Mine")