import os
import torch
import joblib
import numpy as np
from models.model_v1_2018 import Model_v1_2018
from models.model_v2_2017 import Model_v2_2017
from feature_extractors.pe_feature import PEFeatureExtractor
from models.model_v3 import Model_BIG_v4



model = Model_BIG_v4()
device = torch.device("cpu")

model_path = os.path.join("models", "Model_BIG_v5 (1).pth")

print(f"Loading model from: {model_path}")  

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

scaler = joblib.load("scalers/scaler_big.pkl")
feature_extractor = PEFeatureExtractor()


def analyze_file(file):
    bytez = file.read()
    raw_features = feature_extractor.raw_features(bytez)
    vectorized = feature_extractor.process_raw_features(raw_features).reshape(1, -1)
    scaled = scaler.transform(vectorized)
    input_tensor = torch.tensor(scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        prob = model(input_tensor).item()
    return prob, raw_features



