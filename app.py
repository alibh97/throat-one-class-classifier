from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib
from transformers import SwinModel, SwinConfig
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set device to CPU
device = torch.device('cpu')
print(f'Using device: {device}')

# Load pretrained Swin Transformer Base model and remove the top layer
class SwinBaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(SwinBaseFeatureExtractor, self).__init__()
        config = SwinConfig.from_pretrained('./local_swin_base_patch4_window7_224')
        self.swin = SwinModel.from_pretrained('./local_swin_base_patch4_window7_224', config=config)

    def forward(self, x):
        outputs = self.swin(pixel_values=x)
        return outputs.last_hidden_state.mean(dim=1)  # Use mean of the last hidden state as features

# Load the scaler and One-Class SVM model
scaler = joblib.load('models/scaler_swinbase.pkl')
ocsvm = joblib.load('models/ocsvm_model_swinbase.pkl')

# Preprocess new input images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image):
    image = Image.open(io.BytesIO(image)).convert('RGB')
    image = data_transforms(image)
    return image.unsqueeze(0)  # Add batch dimension

def extract_features_from_image(image_tensor):
    model = SwinBaseFeatureExtractor().to(device)
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        features = model(image_tensor)
    return features.cpu().numpy()

def predict_anomaly(features):
    # Standardize the new features using the loaded scaler
    features_scaled = scaler.transform(features)
    # Predict anomalies on the new features
    prediction = ocsvm.predict(features_scaled)
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image = file.read()
        image_tensor = preprocess_image(image)
        features = extract_features_from_image(image_tensor)
        prediction = predict_anomaly(features)
        result = 'Throat' if prediction[0] == 1 else 'NotThroat'
        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
