
from flask import Flask, request, jsonify
from flask_session import Session
import pickle
import numpy as np
from PIL import Image
import paho.mqtt.client as mqtt
from time import sleep

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './sessions'
session = Session(app)

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method != 'POST':
        return jsonify({'error': 'POST method expected'}), 400
    
    if 'image' not in request.form:
        return jsonify({'error': 'No image in request'}), 400

    # Get the image file from the request
    file = request.form['image']

    # Open the image file using PIL
    image = Image.open(file)

    # Resize the image to the target size used during training
    image_resized = image.resize((128, 128))

    # Convert the image to an array of pixels
    image_array = img_to_array(image_resized)

    # Normalize the pixel values to be between 0 and 1
    image_normalized = image_array / 255.0

    # Make the prediction using the model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(image_normalized)

    # Get the last prediction from the session
    last_prediction = session.get('last_prediction', None)

    if prediction != last_prediction:
        result = session.get('result', None) + prediction
        # Update the last prediction in the session
        session['last_prediction'] = prediction
        session['result'] = result

        # Publish the new prediction to the MQTT broker
        mqttc.publish('result', result)

    # Generate a JSON response containing the prediction
    response = {'prediction': prediction}

    # Return the JSON response
    return jsonify(response)

if __name__ == '__main__':
    mqttc = mqtt.Client()
    mqttc.connect('localhost', 1883, 60)
    app.run(debug=True)
