#include "Arduino.h"
#include <WiFi.h>
#include "ESP32MQTTClient.h"
#include "esp_camera.h"

const char *ssid = "Shawn";
const char *pass = "Shawnstevechang";
char *server = "mqtt://192.168.103.1:1883";
char *publishTopicCam = "result/Camera";
ESP32MQTTClient mqttClient;

// OV2640 camera module pins (CAMERA_MODEL_AI_THINKER)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// FLASH pin, always 4
#define FLASH 4

const int triggerPin = 13;
// Set the camera details
camera_config_t camera_config = {
  .pin_pwdn = PWDN_GPIO_NUM,
  .pin_reset = RESET_GPIO_NUM,
  .pin_xclk = XCLK_GPIO_NUM,
  .pin_sscb_sda = SIOD_GPIO_NUM,
  .pin_sscb_scl = SIOC_GPIO_NUM,
  .pin_d7 = Y9_GPIO_NUM,
  .pin_d6 = Y8_GPIO_NUM,
  .pin_d5 = Y7_GPIO_NUM,
  .pin_d4 = Y6_GPIO_NUM,
  .pin_d3 = Y5_GPIO_NUM,
  .pin_d2 = Y4_GPIO_NUM,
  .pin_d1 = Y3_GPIO_NUM,
  .pin_d0 = Y2_GPIO_NUM,
  .pin_vsync = VSYNC_GPIO_NUM,
  .pin_href = HREF_GPIO_NUM,
  .pin_pclk = PCLK_GPIO_NUM,
  .xclk_freq_hz = 20000000,  // Adjust the frequency as needed
  .ledc_timer = LEDC_TIMER_0,
  .ledc_channel = LEDC_CHANNEL_0,
  .pixel_format = PIXFORMAT_JPEG, // Adjust as needed
  .frame_size = FRAMESIZE_SVGA,   // Adjust as needed
  .jpeg_quality = 16,            // Adjust as needed (0-63)
  .fb_count = 3,                   // Double buffer
  .grab_mode = CAMERA_GRAB_LATEST
};

void setup() {
  Serial.begin(115200);
  pinMode(FLASH, OUTPUT); 
  // Set the FLASH pin as an output
  pinMode(triggerPin, INPUT_PULLUP);
  // Use INPUT_PULLUP to enable the internal pull-up resistor
  // Initialize the camera based on AI Thinker config
  esp_camera_init(&camera_config); 
  Serial.println("Connecting to Wi-Fi...");
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) {
      delay(1000);
      Serial.print(".");
    }
  // Once connected, print the IP address
  Serial.println("");
  Serial.println("Connected to Wi-Fi!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  log_i();
  log_i("setup, ESP.getSdkVersion(): ");
  log_i("%s", ESP.getSdkVersion());
  mqttClient.enableDebuggingMessages();
  mqttClient.setURI(server);
  mqttClient.enableLastWillMessage("lwt", "I am going offline");
  mqttClient.setKeepAlive(120);
  WiFi.setHostname("c3test");
  mqttClient.loopStart();
}

void loop() {
  if(mqttClient.isConnected()){
    delay(5000);
    digitalWrite(FLASH, HIGH); // Flash to show photo is taken
    captureAndSendPhoto();
    digitalWrite(FLASH, LOW); 
    delay(2000); // Wait a bit to send
  }
}


void captureAndSendPhoto() {
  // This function captures a photo and stores it in reserved memory, fb is just the pointer
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { // If no photo raise an error
    Serial.println("Photo Error");
    return;
  }
  String imageBase64 = base64_encode(fb->buf, fb->len); // Run 64base encoder using encoded string
  mqttClient.publish(publishTopicCam, imageBase64); // Publish the encoded message thru MQTT
  esp_camera_fb_return(fb); // Clear the reserved memory for next photo
}

String base64_encode(const uint8_t* data, size_t length) {
  const char base64Chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; // 64 base encoding, can be any 64 characters
  String encoded;
  int i = 0;
  while (i < length) {
    // Read the first 8-bit chunk
    uint8_t octet_a = i < length ? data[i++] : 0;
    uint8_t octet_b = i < length ? data[i++] : 0;
    uint8_t octet_c = i < length ? data[i++] : 0;
    // Combine the 8-bit chunks into a 24-bit group
    uint32_t triple = (octet_a << 16) + (octet_b << 8) + octet_c;
    // Break the 24-bit group into four 6-bit chunks and append base64 characters
    encoded += base64Chars[(triple >> 18) & 63];
    encoded += base64Chars[(triple >> 12) & 63];
    encoded += base64Chars[(triple >> 6) & 63];
    encoded += base64Chars[triple & 63];
  }
  // Add padding if necessary
  while (length % 3 != 0) {
    encoded += '=';
    length++;
  }
  return encoded;
}

void onConnectionEstablishedCallback(esp_mqtt_client_handle_t client)
{
  mqttClient.subscribe(publishTopicCam, [](const String &payload)
                        { Serial.println (String (publishTopicCam) + String("") + String(payload.c_str())); });
}

esp_err_t handleMQTT(esp_mqtt_event_handle_t event)
{
  mqttClient.onEventCallback(event);
  return ESP_OK;
}

