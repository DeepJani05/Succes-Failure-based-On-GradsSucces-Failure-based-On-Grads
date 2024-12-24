# Project Description
Title: Success/Failure Prediction Based on Grades

This project aims to build a predictive model to determine whether a student will succeed or fail based on their grades. The dataset consists of student grades and a corresponding success/failure label (1 for success, 0 for failure). By analyzing the dataset and training a machine learning model, we can uncover patterns in the data that contribute to academic success or failure.

The steps include:

Problem Definition
Data Loading and Exploration
Data Cleaning and Preparation
Data Visualization
Model Building and Training
Model Evaluation
Prediction and Conclusion

Explanation of the Code
Data Loading:
The data is manually entered as a dictionary and converted into a DataFrame.

Exploration:
The dataset is explored using methods like .info(), .describe(), and checking for missing values.

Visualization:

Boxplot to compare grades distribution between success and failure.
Histogram to show the overall grade distribution.
Count plot to visualize the count of success and failure.
Data Preparation:
The features (Grades) and target variable (Success) are split into training and testing datasets.

Model Building:
A Random Forest Classifier is used to build the model.

Evaluation:
Accuracy, confusion matrix, and classification report are generated to evaluate model performance.

Prediction:
The model predicts success/failure for new grades, and the predictions are saved to a CSV file for further use.

# Expected Output
Accuracy of the model: Percentage accuracy on the test dataset.
Confusion Matrix: Matrix showing true positives, true negatives, false positives, and false negatives.
Classification Report: Metrics such as precision, recall, and F1-score.
Predictions: New grades are labeled with predicted success or failure.
