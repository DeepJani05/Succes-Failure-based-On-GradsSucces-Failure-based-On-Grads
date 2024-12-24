# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Problem Definition
"""
The aim of this project is to predict student success (1) or failure (0) based on their grades.
"""

# 2. Data Loading
data = {
    'Grades': [75, 72, 69, 80, 53, 55, 98, 55, 83],
    'Success': [1, 1, 0, 1, 0, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

print("Dataset:")
print(df)

# 3. Data Exploration
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Data Visualization
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Success", y="Grades", palette="coolwarm")
plt.title("Grades Distribution by Success/Failure")
plt.xlabel("Success (1) or Failure (0)")
plt.ylabel("Grades")
plt.show()

sns.histplot(df['Grades'], bins=10, kde=True, color='blue')
plt.title("Grades Distribution")
plt.xlabel("Grades")
plt.ylabel("Frequency")
plt.show()

sns.countplot(x="Success", data=df, palette="viridis")
plt.title("Success/Failure Count")
plt.xlabel("Success (1) or Failure (0)")
plt.ylabel("Count")
plt.show()

# 5. Data Preparation
X = df[['Grades']]
y = df['Success']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Model Building and Training
# Using Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy of the Model:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Prediction
new_grades = pd.DataFrame({'Grades': [60, 85, 50, 90]})
predictions = model.predict(new_grades)
new_grades['Success_Prediction'] = predictions

print("\nPredictions on New Grades:")
print(new_grades)

# 9. Saving the Results
new_grades.to_csv("Predictions.csv", index=False)
print("\nPredictions saved to Predictions.csv")
