from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

features = ['Age', 'AST', 'CHE', 'ALT', 'ALP', 'ALB', 'BIL', 'GGT']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        inputs = []
        for feature in features:
            val = request.form.get(feature, 0.0)
            inputs.append(float(val))
            
        data = np.array(inputs).reshape(1, -1)
        scaled = scaler.transform(data)
        
        pred = model.predict(scaled)
        result = encoder.inverse_transform(pred)
        
        return jsonify({"success": True, "prediction": str(result[0])})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)