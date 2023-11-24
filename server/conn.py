
import paho.mqtt.client as mqtt
from time import sleep
import base64
import requests

api_endpoint = "http://172.26.61.181:3237/predict"
file_path = "cam_out.png"

session = requests.Session()

def on_connect(client, userdata, flags, rc):
    print("Connected with result code: " + str(rc))
    client.publish("result/Letters", "Connected to python")
    client.subscribe("result/Camera")

def on_message(client, userdata, message):
    print("Message received: "  + message.topic)
    data = message.payload
    res = base64.b64decode(data)

    with open(file_path, "wb") as fh:
        fh.write(res)

    
    response = session.post(api_endpoint, files={'image': res})

    print(response.json())

    result = response.json()['result'] if response.json()['result'] != '' else 'NIL'

    client.publish("result/Letters", result)
    


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect('192.168.103.1', 1883, 60)
client.loop_forever()

session.close()

