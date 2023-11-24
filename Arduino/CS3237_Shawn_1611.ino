#include <LiquidCrystal_I2C.h>
#include "Arduino.h"
#include <WiFi.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "ESP32MQTTClient.h"

String letter; 
String prev_letter;
const char *ssid = "Shawn";
const char *pass = "Shawnstevechang";

// Test Mosquitto server, see: https://test.mosquitto.org
char *server = "mqtt://192.168.103.1:1883";

char *subscribeTopic = "result/Letters";
char *publishTopic = "";
ESP32MQTTClient mqttClient; // all params are set later


int IRINPUT = 4; // IR Sensor input pin, used for wakeup
int val = LOW;
int countdown_to_sleep = 0;
LiquidCrystal_I2C lcd(0x27,16,2);  // set the LCD address to 0x27 for a 16 chars and 2 line display

void print_wakeup_reason(){ // this is the wakeup function when IR sensor is triggered. Keep it short, main part (EG sending signals for void loop)
  lcd.print("Hello World!"); // print on LCD 
  delay(1000);
  lcd.clear();
  Serial.println("Wakeup");

}

void setup() { // Setup runs even after waking from deepsleep mode
  Serial.begin(115200);
  lcd.init(); //clear LCD on startup
  lcd.clear();  
  lcd.backlight();
  pinMode(IRINPUT, INPUT); // Set IR Sensor to input mode
  esp_sleep_enable_ext0_wakeup(GPIO_NUM_4,1); //1 = High, 0 = Low Set wakeup to PIN 4, IRSENSOR
  print_wakeup_reason(); // Runs if wokeup


  mqttClient.enableDebuggingMessages();

  mqttClient.setURI(server);
  mqttClient.enableLastWillMessage("lwt", "I am going offline");
  mqttClient.setKeepAlive(30);
  WiFi.begin(ssid, pass);
  WiFi.setHostname("c3test");
  mqttClient.loopStart();

}

void loop() {
  val = digitalRead(IRINPUT); // Read IR Sensor
  
  if (val == LOW){
    countdown_to_sleep++; // start counting down each second to sleep mode if IR sensor not in use
    
  }  else{
    Serial.println("Motion Detected");
    countdown_to_sleep = 0;
  }
  delay(2000);
  

  if (countdown_to_sleep == 20){ // GOTO SLEEP on timer or button press
    countdown_to_sleep = 0;
    lcd.clear();
    lcd.print("Sleep Time");
    delay(1000);
    lcd.clear();
    lcd.noBacklight(); // turn backlight off

    esp_deep_sleep_start(); 

  }


mqttClient.subscribe(subscribeTopic, subscribe_info);
countdown_to_sleep = 0; 
if (letter == "Del"){
  lcd.clear();
}
else if (letter == "NIL"){
  Serial.println("Nothing");
}
else{
  lcd.print(letter);
  prev_letter = letter;

}
  
delay(3000);
lcd.clear();

}


void subscribe_info(const String &message){
  letter = message;
}


void onConnectionEstablishedCallback(esp_mqtt_client_handle_t client)
{
  //print out message to test, add a saved value for task 8
  mqttClient.subscribe(subscribeTopic, [](const String &payload)
    { Serial.println(payload.c_str()); String stored_val = String(payload.c_str()); letter = String(payload.c_str());});
  mqttClient.subscribe("subscribeTopic", [](const String &topic, const String &payload)
    { log_i("%s: %s", subscribeTopic, payload.c_str()); });
    
}


esp_err_t handleMQTT(esp_mqtt_event_handle_t event)
{
    mqttClient.onEventCallback(event);
    return ESP_OK;
}
