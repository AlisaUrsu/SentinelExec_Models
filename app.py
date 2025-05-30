from flask import Flask, request, jsonify
import torch
import numpy as np
import joblib  # or pickle
import os

from waitress import serve

# Import your model class and feature extractor
from models.model_v1_2018 import Model_v1_2018
from feature_extractors.pe_feature import PEFeatureExtractor
from models.model_v2_2017 import Model_v2_2017



app = Flask(__name__)

def categorize_score(score):
    if score < 0.2:
        return "Safe", "✅ This file appears to be clean and safe to run. No suspicious indicators were found during the scan."
    elif score < 0.4:
        return "Likely Safe", "🟢 No major threats detected, but some minor anomalies were found. Proceed with general caution."
    elif score < 0.6:
        return "Unknown", "🟡 The file's behavior could not be conclusively analyzed. It may require further inspection before use."
    elif score < 0.8:
        return "Suspicious", "🟠 Warning: This file exhibits unusual or potentially unsafe behavior. Avoid running it unless you trust the source."
    else:
        return "Malicious", "🔴 Danger: This file is very likely to be harmful. Running it may compromise your system or data. Do not execute."

'''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model_v1_2018()
model.load_state_dict(torch.load("models/model_v1_2018.pth", map_location=device))
model.to(device)
model.eval()

scaler = joblib.load("scalers/scaler_EB.pkl")'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model_v2_2017()
model.load_state_dict(torch.load("models/model_v2_2017.pth", map_location=device))
model.to(device)
model.eval()

scaler = joblib.load("scalers/scaler_v2_2017.pkl")

# Feature extractor instance (you can point this to your JSON config if needed)
feature_extractor = PEFeatureExtractor()

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    #if not file or not file.filename.endswith(".exe"):
    #    return jsonify({"error": "Only .exe files are supported"}), 400
    try:
        # Read file as bytes
        bytez = file.read()

        # Extract raw and vectorized features
        raw_features = feature_extractor.raw_features(bytez)
        vectorized = feature_extractor.process_raw_features(raw_features).reshape(1, -1)
        #file_tensor = torch.tensor(vectorized, dtype=torch.float32).unsqueeze(0).to(device)

        # Scale the features
        scaled = scaler.transform(vectorized)

        # Convert to tensor and predict
        input_tensor = torch.tensor(scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            prob = model(input_tensor).item()

        print(f"Probability of being malicious: {prob * 100:.2f}%")

        score = round(prob, 6)
        label, message = categorize_score(score)

        return jsonify({
            "file": file.filename,
            "score": score,
            "label": label,
            "message": message,
            "rawFeatures": raw_features
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Hello, Flask!"

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
