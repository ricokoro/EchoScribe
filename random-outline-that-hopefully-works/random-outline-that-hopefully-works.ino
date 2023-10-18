#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid = "PLACEHOLDER";
const char* password = "PLACEHOLDER";
const char* serverURL = "PLACEHOLDER";

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
}

void loop() {
  if (shouldCaptureAudio()) {
    byte audioData[2048]; //2048 is just a random number, its a nice number :)
    int audioLength = captureAudio(audioData, sizeof(audioData));

    if (audioLength > 0) {
      sendAudioToCloud(audioData, audioLength);
    }
  }
  delay(100);
}

bool shouldCaptureAudio() {
  // empty for now coz im tired!
  return false;
}

int captureAudio(byte* buffer, int bufferSize) {
  // this logic should be implemented after we decide the cloud service we will be using i believe
  return 0;
}

void sendAudioToCloud(byte* audioData, int audioLength) {
  HTTPClient http;
  http.begin(serverURL);

  http.addHeader("Content-Type", "audio/wav"); //probably need to change also, depending on which cloud service we use
  int httpResponseCode = http.POST(audioData, audioLength);

  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.println(response);
  } else {
    Serial.println("Error sending audio to the cloud");
  }
  http.end();
}
