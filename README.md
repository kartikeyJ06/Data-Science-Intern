# Data-Science-Intern
Task1_(Iris.Csv)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from google.colab import drive
drive.mount('/content/drive')

# Load the Iris dataset
iris = pd.read_csv('Iris.csv')

# Split the dataset into features (X) and target variable (y)
X = iris.drop('Species', axis=1)
y = iris['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and implement a machine learning algorithm (Decision Tree)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
#Output
DecisionTreeClassifier
DecisionTreeClassifier()

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

#Output
Accuracy: 1.00
Precision: 1.00
Recall: 1.00


#2. Simple Exploratory Data Analysis (EDA):
# Import necessary libraries for EDA
import seaborn as sns
import matplotlib.pyplot as plt

# EDA to understand the structure and characteristics of the Iris dataset
# Explore the distribution of each feature
sns.pairplot(iris,hue='Species', diag_kind='hist')
plt.show()

# Create visualizations such as histograms, box plots, or scatter plots
sns.boxplot(x='Species', y='SepalLengthCm', data=iris)
plt.show()

sns.boxplot(x='Species', y='SepalWidthCm', data=iris)
plt.show()

sns.boxplot(x='Species', y='PetalLengthCm', data=iris)
plt.show()

sns.boxplot(x='Species', y='PetalWidthCm', data=iris)
plt.show()



