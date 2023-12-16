import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter("ignore")

df = pd.read_csv("data/final_test.csv")

print(df.head())

# Filling the missing values with the median
df['age'] = df['age'].fillna(df['age'].median())
df['height'] = df['height'].fillna(df['height'].median())

print(df.isna().sum())

print(df["size"].value_counts())
size_mapping = {'XXS': 1, 'S': 2, "M": 3, "L": 4, "XL": 5, "XXL": 6, "XXXL": 7}

df['size'] = df['size'].map(size_mapping)

X = df.drop("size", axis=1)
Y = df["size"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(len(X_train))
print(len(X_test))

from sklearn.linear_model import LinearRegression

clf = LinearRegression()

# clf.fit(X,Y)

clf.fit(X_train,y_train)
clfScore = clf.score(X_test,y_test)
print("Accuracy obtained by linear regression model:",clfScore*100)
while True:
    input_data = input("Enter weight, age, height (separated by spaces): ")

    # Split the input based on spaces and assign to variables
    weight, age, height = input_data.split()

    # Convert the inputs to the appropriate data types (e.g., int, float)
    weight = float(weight)
    age = int(age)
    height = int(height)
    input_features = [[weight, age, height]]  # Create a list of lists or a 2D array

    # Make predictions using the trained model
    predicted_value = clf.predict(input_features)


    # Print the predicted value
    print("Predicted value:", predicted_value[0])  # Assuming it's a single prediction

    rounded_predicted_value = int(round(predicted_value[0]))

    # Map the rounded value to the corresponding size from the dictionary
    predicted_size = [key for key, value in size_mapping.items() if value == rounded_predicted_value]

    if predicted_size:
        print("Predicted size:", predicted_size[0])
    else:
        print("No corresponding size found for the prediction.")

