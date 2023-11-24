# EchoScribe

## Running the project

### Setting up

1. Run mosquitto on a server or device. Add the IP address of this device to `conn.py`, and the two arduino files.
2. Run `app.py` on a device or server (can be the same or different device). Add this IP address to `conn.py` as the `api_endpoint`.
3. Run `conn.py` on any device.
4. Connect the LCD and IR to an ESP32 an run `CS3237_Shawn_1611.ino` on it. 

Note: all devices need to be on the same network for this to work

### Using webcamera (device camera)
5. Run cam.py on the device. Ensure the url contains the IP address of the device on which `app.py` is running.

### Using external camera (OV2640 Camera)
5. Run `Camera_photo_MQTT.ino` on the camera. 
