
from flask import Flask, request, jsonify, session
from flask_session import Session
import pickle
import numpy as np
from PIL import Image
import paho.mqtt.client as mqtt
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array

mqttc = mqtt.Client()
mqttc.connect('localhost', 1883, 60)
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './sessions'
Session(app)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method != 'POST':
        return jsonify({'error': 'POST method expected'}), 400
    print(str(request.files), len(request.files))
    if 'image' not in request.files:
        return jsonify({'error': 'No image in request'}), 400

    # Get the image file from the request
    file = request.files['image']

    # Open the image file using PIL
    image = Image.open(file)

    # Resize the image to the target size used during training
    image_resized = image.resize((128, 128))

    # Convert the image to an array of pixels
    image_array = img_to_array(image_resized)

    # Normalize the pixel values to be between 0 and 1
    image_normalized = image_array / 255.0

    # Make the prediction using the model
    # with open('model.pkl', 'rb') as f:
    #     model = pickle.load(f)

    prediction = "hello" # model.predict(image_normalized)

    # Get the last prediction from the session
    last_prediction = session['last_prediction'] if 'last_prediction' in session else '' #.get('last_prediction', None)

    if prediction != last_prediction or True:
        result = last_prediction + prediction
        # Update the last prediction in the session
        session['last_prediction'] = prediction
        session['result'] = result

        print(result)

        # Publish the new prediction to the MQTT broker
        mqttc.publish('result', result)

    # Generate a JSON response containing the prediction
    response = {'prediction': prediction}

    # Return the JSON response
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
