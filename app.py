from flask import Flask, request, jsonify
import pickle  # or joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (Ensure 'model.pkl' exists in the same directory)
model_filename = "model.pkl"  # Change if using joblib: "model.joblib"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Flask API is running! Use /predict to get predictions."

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Convert input into a NumPy array
        input_features = np.array(data['features']).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)

        # Return result
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
