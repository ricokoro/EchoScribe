# ubuntu
from flask import Flask, request, jsonify, session
from flask_session import Session
import pickle
import numpy as np
from PIL import Image
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

npz_file = np.load('model_weights.npz')

with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(6, activation='softmax') 
])

weights = [npz_file[key] for key in npz_file]
model.set_weights(weights)

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './sessions'
Session(app)

# File to store the text data
file_path = "text_data.txt"

# Read initial text from the file
try:
    with open(file_path, "r") as file:
        text = file.read()
except FileNotFoundError:
    # If the file doesn't exist yet, create it
    text = ""
    with open(file_path, "w") as file:
        file.write(text)

@app.route('/', methods=['GET'])
def index():
    global text
    return 'Hi! This is your transcript:<br>' + text.replace('\n', '<br>')

@app.route('/predict', methods=['POST'])
def predict():
    global text, file_path, model, le

    if request.method != 'POST':
        return jsonify({'error': 'POST method expected'}), 400

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

    # Expand dimensions to create a batch with a single image
    image_normalized = np.expand_dims(image_normalized, axis=0)

    # Make the prediction using the model
    prediction = model.predict(image_normalized)

    predicted_class = np.argmax(prediction, axis=1)

    # If you have label encodings (e.g., 0='A', 1='B', etc.), decode the predicted class
    predicted_label = le.inverse_transform(predicted_class)
    print(f'Predicted label: {predicted_label[0]}')

    prediction = predicted_label[0]

    print(prediction)

    # Get the last prediction from the session
    last_prediction = session.get('last_prediction', '')
    result = ''

    if prediction == 'del':
        text += "\n"
        session['last_prediction'] = ''
        session['result'] = ''
    elif prediction == 'nothing':
        result = session.get('result', '')
        session['last_prediction'] = ''
        session['result'] = result
    else: # prediction != last_prediction:
        text += prediction
        result = session.get('result', '') + prediction
        # Update the last prediction in the session
        session['last_prediction'] = prediction
        session['result'] = result

    print(prediction)

    with open(file_path, "w") as file:
        file.write(text)

    # Generate a JSON response containing the prediction
    response = {'result': result}

    # Return the JSON response
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='3237', debug=True)
