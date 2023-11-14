
import paho.mqtt.client as mqtt
from time import sleep

# Define the MQTT broker address and topic
broker_address = 'localhost'
cam_topic = 'result'
esp_topic = 'esp/com'


def on_connect(client, userdata, flags, rc):
    print("Connected with result code: " + str(rc))
    client.subscribe(cam_topic)
    client.subscribe(esp_topic)


def on_message(client, userdata, message):
	res = float(message.payload.decode('utf-8'))
	if message.topic == cam_topic:
		println("cam: ", str(res))
		mqtt.publish(esp_topic, str(res)
	elif message.topic == esp_topic:
		println("esp: ", str(res))
        


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker_address, 1883, 60)
client.loop_forever()