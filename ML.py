#Machine learning basics
print("Machine Learning(ML) is a subset of artificial intelligence (AI) that enables computers to learn patterns from data and make decisions without explicit programming. It helps automatedecision-making processes based on historical data")

print()
print()
print("Features of machine learning")
print("Learning from Data: Algorithms analyze data to identify patterns.")
print("Generalization: The model should work on unseen data.")

print("Improvement Over Time: The more data a model is trained on, the better it performs.")
print()
print()
print("Example:")

print("A recommendation system like Netflix suggests movies based on your viewing history by learning patterns from similar users.")
print()
print()
print("A recommendation system like you tube suggests songs playlist based on our interest")

print()
print()
print()
print("            Types of Machine Learning")

print("ML is broadly categorized into three types:")

print("     A. Supervised Learning")

print("The model is trained on labeled data (input-output pairs).")

print("Used for classification (categorizing into classes) and regression (predicting continuous values).")

print("Example 1: Spam Email Classification")
print("Example 2:Package Prediction Based on CGPA")
print("Example 3:House-Price Based on number of rooms and Area in Square unit")


print("     B.Unsupervised Learning")

print("The model is trained on unlabeled data and finds hidden patterns.")

print("Used for clustering (grouping similar items) and association (finding relationships).")

print("Example 1: Customer Segmentation")

print("      A retail company groups customers based on their purchase behavior:")

print("              Group 1: Frequent buyers")

print("              Group 2: Occasional buyers")

print("              Group 3: One-time buyers")

print()
print()

print("        C. Reinforcement Learning")

print("The model learns through rewards and penalties for actions taken in an environment.")

print("Used in robotics, game playing, and self-driving cars.")

print("            Example: AlphaGo (Google's AI playing Go)")

print("              AI learns by playing millions of games, improving over time by learning from past moves.")



#project of House Price 
print("project on house-Price prediction based on area and number of rooms")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Sample Dataset
data = {
    'Area_sqft': [1500, 2000, 2500, 3000, 3500],
    'Bedrooms': [3, 4, 4, 5, 5],
    'Price': [200000, 250000, 300000, 350000, 400000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split into features and target variable
X = df[['Area_sqft', 'Bedrooms']]
y = df['Price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

data = {
    'Feature1': [10, 20, 30, 40, 50],
    'Feature2': [1, 2, 3, 2.75, 5],
    'Target': [100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

# Splitting into Features and Target
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Set:")
print(X_train)
print("Testing Set:")
print(X_test)


print(np.random.seed(42))
data_size = 100
# Creating random data
data = {
    'Feature1': np.random.randint(1, 100, data_size).astype(float),  # Random integers from 1 to 100
    'Feature2': np.random.uniform(10, 500, data_size),  # Random floating numbers between 10 and 500
    'Feature3': np.random.normal(50, 15, data_size),  # Normally distributed data with mean=50, std=15
    'Feature4': np.random.choice([np.nan, 1, 2, 3, 4, 5], data_size, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]),  
    'Feature5': np.random.choice([np.nan, 10, 20, 30, 40], data_size, p=[0.1, 0.2, 0.3, 0.2, 0.2]),  
    'Target': np.random.randint(500, 5000, data_size)  # Random target values
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Display first 10 rows
print(df.head(10))

#Check whether the null value is present
print(df.isnull().sum())

# Create an imputer with 'mean' strategy
imputer = SimpleImputer(strategy='mean')

# Apply imputation on Feature4 and Feature5
df[['Feature4', 'Feature5']] = imputer.fit_transform(df[['Feature4', 'Feature5']])

# Display processed dataset
print("Dataset after handling missing values:")
print(df.head(10))

# Create StandardScaler object
scaler = StandardScaler()

# Apply scaling on numerical features
scaled_features = scaler.fit_transform(df[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']])

# Convert back to DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])

print("Scaled dataset:")
print(scaled_df.head(10))

# Classification Concept
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
print("Feature data (X):\n", iris.data)
print("Target labels (y):\n", iris.target)
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("Dataset description:\n", iris.DESCR)
#Concept of Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target  

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Create and train the Decision Tree model
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)  # Gini for classification
model.fit(X_train, y_train)


# Step 4: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
print("the accuracy score is givenas")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print("the confusion matrix is ")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
