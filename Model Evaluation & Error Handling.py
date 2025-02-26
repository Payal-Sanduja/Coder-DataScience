
#Model Evaluation & Error Handling

import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
# Actual and Predicted values
y_actual = np.array([10, 15, 20, 25, 30])
y_predicted = np.array([12, 14, 19, 24, 28])

# Compute MAE
mae = mean_absolute_error(y_actual, y_predicted)
print("Mean Absolute Error (MAE):", mae)


# Actual and Predicted values
y_actual = np.array([10, 15, 20, 25, 30])
y_predicted = np.array([12, 14, 19, 24, 28])

# Compute MSE
mse = mean_squared_error(y_actual, y_predicted)
print("Mean Absolute Error (MAE):", mse)

# Compute R2 score
r2 = r2_score(y_actual, y_predicted)

print("R² Score:", r2) 
