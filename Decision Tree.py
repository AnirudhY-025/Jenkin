#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file
file_path = '/Users/mac/Downloads/Regression.csv'  # Replace with the path to your file
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Encode categorical variables (e.g., Gender, Type of Travel, Class, Continent)
label_encoder = LabelEncoder()

# Encode binary target variable 'Satisfied' (Y/N)
data['Satisfied'] = label_encoder.fit_transform(data['Satisfied'])

# Encode other categorical variables
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Type of Travel'] = label_encoder.fit_transform(data['Type of Travel'])
data['Class'] = label_encoder.fit_transform(data['Class'])
data['Continent'] = label_encoder.fit_transform(data['Continent'])

# Drop unnecessary columns
data_cleaned = data.drop(columns=['Ref', 'id', 'Destination', 'Age Band'])

# Handle missing values by filling them with the median of their columns
data_cleaned = data_cleaned.fillna(data_cleaned.median())

# Split the data into features (X) and target (y)
X = data_cleaned.drop(columns=['Satisfied'])
y = data_cleaned['Satisfied']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = decision_tree.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)


# In[ ]:




