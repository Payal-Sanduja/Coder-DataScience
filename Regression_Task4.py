# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge,Lasso
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor




print("Linear regresssion")
# Step 2: Create the dataset
X = np.array([[1000], [1500], [1800], [2400], [3000]])
y = np.array([300000, 400000, 450000, 600000, 650000])
print(X)
print(y)
# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("the trained value of x is")
print(X_train)
print("the trained value of y is")
print(y_train)

# Step 4: Create the Linear Regression model
model = LinearRegression()

# Step 5: Train the model on the training data
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)
print("the value of y_pred is given as")
print(y_pred)
# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Output the results
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
# Step 8: Visualize the results
plt.scatter(X, y, color='blue')  # actual data points
plt.plot(X, model.predict(X), color='red')  # regression line
plt.title('Simple Linear Regression: House Price vs. Size')
plt.xlabel('Size of House (square feet)')
plt.ylabel('Price (in dollars)')
plt.show()



print("Multiple Regression")
X = np.array([
    [1000, 3, 10],
    [1500, 4, 5],
    [1800, 3, 8],
    [2400, 5, 2],
    [3000, 4, 1]
])
print("The va;ue of x is :")
print(X)

y = np.array([300000, 400000, 450000, 600000, 650000])
print("The value of y is :")
print(y)
# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create the Multiple Linear Regression model
model = LinearRegression()

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("The value of X_train is ")
print(X_train)

print("The value of Y_train  is ")
print(y_train)

print("The value of y_prediction is ")
print(y_pred)

# Output the results
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')


print("polynomial regression")
# Step 2: Create the dataset (size of house vs. price)
X = np.array([[1000], [1500], [1800], [2400], [3000]])
y = np.array([300000, 400000, 450000, 600000, 650000])

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Transform the features to a higher degree polynomial
poly = PolynomialFeatures(degree=3)  # Change degree for more complexity
X_poly = poly.fit_transform(X)

# Step 5: Create the Polynomial Regression model
model = LinearRegression()

# Step 6: Train the model
model.fit(X_poly, y)

# Step 7: Make predictions
y_pred = model.predict(poly.transform(X_test))

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 9: Visualize the polynomial curve
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(poly.transform(X)), color='red')
plt.title('Polynomial Regression: House Price vs. Size')
plt.xlabel('Size of House (square feet)')
plt.ylabel('Price (in dollars)')
plt.show()

# Output the results
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')


print("Ridge regression")
# Step 2: Create the dataset
X = np.array([[1000], [1500], [1800], [2400], [3000]])  # Size of house
y = np.array([300000, 400000, 450000, 600000, 650000])  # Price of house

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create the Ridge Regression model with a regularization parameter alpha (lambda)
ridge_model = Ridge(alpha=1.0)

# Step 5: Train the model
ridge_model.fit(X_train, y_train)
# Step 6: Make predictions
y_pred = ridge_model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 8: Visualize the results
plt.scatter(X, y, color='blue')
plt.plot(X, ridge_model.predict(X), color='red')
plt.title('Ridge Regression: House Price vs. Size')
plt.xlabel('Size of House (square feet)')
plt.ylabel('Price (in dollars)')
plt.show()
# Output the results
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
print(f'Ridge Coefficient: {ridge_model.coef_}')

print("lasso Regression")


# Step 2: Create the dataset
X = np.array([[1000], [1500], [1800], [2400], [3000]])  # Size of house
y = np.array([300000, 400000, 450000, 600000, 650000])  # Price of house

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create the Lasso Regression model with a regularization parameter alpha (lambda)
lasso_model = Lasso(alpha=0.1)

# Step 5: Train the model
lasso_model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = lasso_model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 8: Visualize the results
plt.scatter(X, y, color='blue')
plt.plot(X, lasso_model.predict(X), color='red')
plt.title('Lasso Regression: House Price vs. Size')
plt.xlabel('Size of House (square feet)')
plt.ylabel('Price (in dollars)')
plt.show()

# Output the results
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
print(f'Lasso Coefficient: {lasso_model.coef_}')

print("decision tree for classification")

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (species)
# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Create the Decision Tree Classifier model
classifier = DecisionTreeClassifier(random_state=42)

# Step 5: Train the model
classifier.fit(X_train, y_train)
# Step 6: Make predictions on the test data
y_pred = classifier.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
# Step 8: Visualize the Decision Tree
plt.figure(figsize=(15,10))
plot_tree(classifier, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title('Decision Tree for Iris Classification')
plt.show()
print("decision tree for regression")

# Step 2: Create the dataset (size of the house vs. price)
X = np.array([[1000], [1500], [1800], [2400], [3000], [3500], [4000], [5000]])  # Size in square feet
y = np.array([300000, 400000, 450000, 600000, 650000, 700000, 750000, 800000])  # Price in dollars

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Create the Decision Tree Regressor model
regressor = DecisionTreeRegressor(random_state=42)
# Step 5: Train the model
regressor.fit(X_train, y_train)

# Step 6: Make predictions on the test data
y_pred = regressor.predict(X_test)
# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
# Step 8: Visualize the decision tree's predictions
plt.scatter(X, y, color='blue')  # Actual data points
plt.plot(X, regressor.predict(X), color='red')  # Predicted values (decision tree curve)
plt.title('Decision Tree Regression: House Price vs. Size')
plt.xlabel('Size of House (square feet)')
plt.ylabel('Price (in dollars)')
plt.show()

print("Random Forest for Classification")
# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (species)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Create the Random Forest Classifier model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Step 5: Train the model
rf_classifier.fit(X_train, y_train)

# Step 6: Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Step 8: Visualize the feature importances
features = iris.feature_names
importances = rf_classifier.feature_importances_

plt.barh(features, importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Classifier')
plt.show()

print("Random Forest for Regression")
# Step 2: Create the dataset (size of house vs. price)
X = np.array([[1000], [1500], [1800], [2400], [3000], [3500], [4000], [5000]])  # Size in square feet
y = np.array([300000, 400000, 450000, 600000, 650000, 700000, 750000, 800000])  # Price in dollars

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Create the Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Step 5: Train the model
rf_regressor.fit(X_train, y_train)
# Step 6: Make predictions on the test data
y_pred = rf_regressor.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
# Step 8: Visualize the predictions vs. actual values
plt.scatter(X, y, color='blue')  # Actual data points
plt.plot(X, rf_regressor.predict(X), color='red')  # Predicted values
plt.title('Random Forest Regression: House Price vs. Size')
plt.xlabel('Size of House (square feet)')
plt.ylabel('Price (in dollars)')
plt.show()



