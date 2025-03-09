print("Pickle ")


import pickle
from sklearn.linear_model import LogisticRegression

# Sample model
model = LogisticRegression()
model.fit([[1, 2], [2, 3], [3, 4]], [0, 1, 0])

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
# Load the model from the file
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Make predictions
print(loaded_model.predict([[2, 3]]))

print("Joblib")

import joblib

# Save the model
joblib.dump(model, 'model.joblib')
# Load the model
loaded_model = joblib.load('model.joblib')

# Make predictions
print(loaded_model.predict([[2, 3]]))

