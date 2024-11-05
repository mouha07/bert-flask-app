import threading
import requests
from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn
import time
import boto3
import botocore
import mysql.connector
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from spacy import displacy
from spacy.tokens import Span


analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')

import atexit
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



# Configuration de la connexion à la base de données
db_config = {
    'host': 'businessintelligence.cdg84mokcel1.us-east-2.rds.amazonaws.com',
    'user': 'admin',
    'password': 'AmazoneRDS2025',  # Remplacez par votre mot de passe
    'database': 'business_intelligence_db'    # Remplacez par le nom de votre base de données
}


def get_db_connection():
    db_config = {
        'host': 'businessintelligence.cdg84mokcel1.us-east-2.rds.amazonaws.com',
        'user': 'admin',
        'password': 'AmazoneRDS2025',  # Remplacez par votre mot de passe
        'database': 'business_intelligence_db'    # Remplacez par le nom de votre base de données
     }
    try:
        conn = mysql.connector.connect(**db_config)
        if conn.is_connected():
            print("Connexion réussie à la base de données")
        return conn
    except Error as e:
        print(f"Erreur lors de la connexion à la base de données: {e}")
        return None 
# cursor = connection.cursor()


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
# Liste des dépendances intéressantes pour l'analyse syntaxique
interesting_deps = {
    "nsubj": "sujet nominal",
    "dobj": "objet direct",
    "amod": "modificateur nominal",
}

# Route pour l'interface utilisateur
@app.route('/')
def home():
    conn = get_db_connection()
    if conn is None:
        return "Erreur lors de la connexion à la base de données.", 500

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, text FROM comments")
        comments = cursor.fetchall()
        cursor.close()
        return render_template('index.html', comments=comments)
    except Error as e:
        print(f"Erreur lors de la récupération des résultats: {e}")
        return "Erreur lors de la récupération des résultats.", 500
    finally:
        conn.close()  # Assurez-vous de fermer la connexion dans tous les cas



# # Route de prédiction
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     if 'text' not in data:
#         return jsonify({'error': 'No text provided'}), 400
#     text = data['text']
    
#     # Tokenize and encode text
#     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=20)
    
#     # Get model predictions
#     with torch.no_grad():
#         outputs = model(inputs['input_ids'], mask=inputs['attention_mask'])
    
#     # Convert logits to probabilities
#     probabilities = torch.exp(outputs)
#     predicted_class = torch.argmax(probabilities, dim=1).item()
    
#     # Enregistrer le commentaire et la classe dans la base de données
#     insert_query = "INSERT INTO comments (text, class) VALUES (%s, %s)"
#     cursor.execute(insert_query, (text, predicted_class))
#     connection.commit()  # Enregistrez les modifications
    
#     return jsonify({
#         'text': text,
#         'predicted_class': predicted_class,
#         'probabilities': probabilities.tolist()
#     })




# Route de prédiction
@app.route('/predict/<int:comment_id>', methods=['GET'])
def predict(comment_id):
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Erreur lors de la connexion à la base de données."}), 500
    
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT text FROM comments WHERE id = %s", (comment_id,))
        comment = cursor.fetchone()

        if not comment:
            return jsonify({"error": "Commentaire introuvable."}), 404
        
        text = comment['text']
        doc = nlp(text)

        positive_terms = []
        negative_terms = []
        interesting_relations = []

        # Analyse des sentiments et relations
        for token in doc:
            score = analyzer.polarity_scores(token.text)["compound"]
            token._.sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
            if score > 0:
                positive_terms.append(token.text)
            elif score < 0:
                negative_terms.append(token.text)

        # Visualisation des entités
        positive_spans = [Span(doc, token.i, token.i+1, label="POSITIVE") for token in doc if token._.sentiment == "positive"]
        negative_spans = [Span(doc, token.i, token.i+1, label="NEGATIVE") for token in doc if token._.sentiment == "negative"]
        doc.set_ents(positive_spans + negative_spans, default="unmodified")

        # Options de couleur pour les termes
        colors = {"POSITIVE": "linear-gradient(90deg, #a3e635, #3cb371)", 
                  "NEGATIVE": "linear-gradient(90deg, #ff6347, #dc143c)"}
        options = {"ents": ["POSITIVE", "NEGATIVE"], "colors": colors}

        # Ajouter les relations syntaxiques intéressantes
        for token in doc:
            if token.text in positive_terms or token.text in negative_terms:
                for child in token.children:
                    if child.dep_ in interesting_deps:
                        relation_type = interesting_deps[child.dep_]
                        interesting_relations.append(f"Relation : '{token.text}' --> '{child.text}', Type : {relation_type}")

        # Générer le rendu HTML pour la visualisation
        html_visualization = displacy.render(doc, style="ent", options=options, jupyter=False)
        
        # Rassembler les résultats
        result_html = html_visualization + "<br><strong>Relations Syntaxiques Intéressantes :</strong><br>"
        result_html += "<br>".join(interesting_relations)

        return jsonify({"html": result_html})

    except Error as e:
        print(f"Erreur lors de la récupération des résultats: {e}")
        return jsonify({"error": "Erreur lors de la récupération des résultats."}), 500
    finally:
        cursor.close()
        conn.close()

# Route pour obtenir les commentaires

@app.route('/comments', methods=['GET'])
def get_comments():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Erreur lors de la connexion à la base de données."}), 500

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, text, class, date FROM comments")
        comments = cursor.fetchall()

        # Transforme les résultats en une liste de dictionnaires
        comments_list = [
            {"id": row['id'], "text": row['text'], "class": row['class'], "date": row['date']} for row in comments
        ]

        return jsonify(comments_list)
    except Error as e:
        print(f"Erreur lors de la récupération des commentaires: {e}")
        return jsonify({"error": "Erreur lors de la récupération des commentaires."}), 500
    finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    app.run(debug=True)


# @atexit.register
# def close_db_connection():
#     if cursor:
#         cursor.close()
#     if connection:
#         connection.close()

# Exécuter l'application Flask
def run_app():
    app.run(host='0.0.0.0', port=5001)

# Démarrer le serveur Flask dans un thread séparé
thread = threading.Thread(target=run_app)
thread.start()
