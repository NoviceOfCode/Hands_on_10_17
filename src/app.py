from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
import networkx as nx

# Load the pre-trained model
rf_model = joblib.load('models/rf_model.pkl')
nn_model = keras.load('models/nn_model.pkl')

def get_recommendation(motion_type):
    def get_recommendation(motion_type):
        G = nx.Graph()

        # Example knowledge graph for fitness management and pain reliefrecommendations

        G.add_edge(0, 'Stretching', recommendation='Do 10 mins of light stretching to relieve tension')
        G.add_edge(1, 'Yoga', recommendation='Practice 15 mins of gentle yoga for painrelief')
        G.add_edge(2, 'Strength Training', recommendation='Perform low-impact strength training exercises')
        # Query based on motion type
        if motion_type == 0:
            return {'exercise': G[0]['Stretching']['recommendation']}
        elif motion_type == 1:
            return {'exercise': G[1]['Yoga']['recommendation']}
        else:
            return {'exercise': G[2]['Strength Training']['recommendation']}
#Initialize Flask app
app = Flask(__name__)
# Defin route for motion deteciton using ensemble model
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['accelaration_x'], data['accelaration_y'], data['accelaration_z']])

    # Scale the features
    features_scaled = scaler.transform(features)
    # Get predictions from both models
    rf_pred = rf_model.predict(features_scaled)
    nn_pred = np.argmax(nn_model.predict(features_scaled), axis=1)
    # Ensemble: Majority voting
    final_pred = np.round((rf_pred + nn_pred) / 2)
    # Query the knowledge graph for pain management recommendations based on the motion type
    recommendation = get_recommendation(final_pred)
    # Return prediction and recommendation as JSON
    return jsonify({
        'motion_type': int(final_pred),
        'recommendation': recommendation
    })