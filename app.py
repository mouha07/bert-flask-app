import threading
import requests
from flask import Flask, request, jsonify
import torch
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn
import time


import boto3
import botocore

# Configuration
BUCKET_NAME = 'bert-model-poids'  # Remplacez par le nom exact de votre bucket
KEY = 'best_model_weights.pt'  # Chemin relatif de l'objet dans S3
LOCAL_FILE_PATH = 'best_model_weights.pt'  # Chemin local où vous souhaitez enregistrer le fichier

# Connexion au service S3
s3 = boto3.resource('s3')

# Téléchargement avec gestion des erreurs
try:
    s3.Bucket(BUCKET_NAME).download_file(KEY, LOCAL_FILE_PATH)
    print(f"Téléchargement réussi dans {LOCAL_FILE_PATH}")
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("L'objet n'existe pas dans le bucket.")
    else:
        raise



# Initialize Flask application
app = Flask(__name__)
# Load BERT model and tokenizer for multilingual support
bert = AutoModel.from_pretrained('bert-base-multilingual-cased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', use_fast=True, do_lower_case=False)

# Function to unfreeze the last layers of BERT
def unfreeze_bert_layers(model, num_layers_to_unfreeze=2):
    for name, param in model.bert.named_parameters():
        if "encoder.layer" in name:
            layer_index = int(name.split("encoder.layer.")[-1].split(".")[0])
            param.requires_grad = layer_index >= (12 - num_layers_to_unfreeze)

# Define custom BERT architecture with a classification head
class BERT_architecture(nn.Module):
    def __init__(self, bert):
        super(BERT_architecture, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        unfreeze_bert_layers(self, num_layers_to_unfreeze=2)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)

# Initialize model and load state
model = BERT_architecture(bert)
# model.load_state_dict(torch.load('/kaggle/input/bert_weight/pytorch/bert_weight/1/best_model_weights.pt'))
# Chargement des poids dans le modèle
# model.load_state_dict(torch.load(LOCAL_FILE_PATH, map_location=torch.device('cpu')))

try:
    # Chargement des poids avec weights_only=True pour plus de sécurité
    model.load_state_dict(torch.load(LOCAL_FILE_PATH, map_location=torch.device('cpu'), weights_only=True))
    print("Poids du modèle chargés avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement des poids : {e}")

model.eval()

# Hello route
@app.route('/')
def hello():
    return "Bonjour ! Bienvenue sur l'API de prédiction."


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    
    # Tokenize and encode text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=20)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(inputs['input_ids'], mask=inputs['attention_mask'])
    
    # Convert logits to probabilities
    probabilities = torch.exp(outputs)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return jsonify({
        'text': text,
        'predicted_class': predicted_class,
        'probabilities': probabilities.tolist()
    })

# Run Flask app
def run_app():
    app.run(host='0.0.0.0', port=5001)

# Start Flask server in a separate thread
thread = threading.Thread(target=run_app)
thread.start()

# Wait for the server to start
time.sleep(5)

# Send a test request to the API
try:
    response = requests.post("http://127.0.0.1:5001/predict", json={"text": "Very Good"})
    print(response.json())
except requests.exceptions.RequestException as e:
    print("Error during the request:", e)

# Display raw response for diagnosis
print("Status Code:", response.status_code)
print("Response Text:", response.text)

# Check if response is in JSON format
try:
    print(response.json())
except requests.exceptions.JSONDecodeError:
    print("The response is not in JSON format.")
