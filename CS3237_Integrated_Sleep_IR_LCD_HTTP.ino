#include <LiquidCrystal_I2C.h>
#include <WiFi.h>
#include <ESP32MQTTClient.h>
int IRINPUT = 4; // IR Sensor input pin, used for wakeup ONLY THIS WORKS

int delay_timing = 100; // fixed delay timing
int val = LOW;
int countdown_to_sleep = 0;
int CAMERATOGGLE = 14;
const char *ssid = "Shawn";
const char *pass = "Shawnstevechang";
// Test Mosquitto server, see: https://test.mosquitto.org
char *server = "http://127.0.0.1:5000";

char *subscribeTopic = "/result";
ESP32MQTTClient mqttClient; // all params are set later

LiquidCrystal_I2C lcd(0x27,16,2);  // set the LCD address to 0x27 for a 16 chars and 2 line display

void print_wakeup_reason(){ // this is the wakeup function when IR sensor is triggered. Keep it short, main part (EG sending signals for void loop)
  lcd.print("Hello World!"); // print on LCD 
  delay(1000);
  lcd.clear();
  digitalWrite(CAMERATOGGLE, HIGH);

}


void setup() { // Setup runs even after waking from deepsleep mode
  Serial.begin(115200);
  lcd.init(); //clear LCD on startup
  lcd.clear();  
  lcd.backlight();
  pinMode(IRINPUT, INPUT); // Set IR Sensor to input mode
  esp_sleep_enable_ext0_wakeup(GPIO_NUM_4,1); //1 = High, 0 = Low Set wakeup to PIN 4, IRSENSOR
  print_wakeup_reason(); // Runs if wokeup
  pinMode(CAMERATOGGLE, OUTPUT);
  
  mqttClient.enableDebuggingMessages();

  mqttClient.setURI(server);
  mqttClient.connect();
  mqttClient.subscribe(subscribeTopic, onCommandReceived)
  WiFi.begin(ssid, pass);
  WiFi.setHostname("c3test");



}

void loop() {
  val = digitalRead(IRINPUT); // Read IR Sensor
  // Always subscribe to MQTT

  if (val == LOW){
    countdown_to_sleep++; // start counting down each second to sleep mode if IR sensor not in use
    delay(1000);
  }

  if (countdown_to_sleep == 10){ // GOTO SLEEP 
    lcd.clear();
    lcd.print("Sleep Time");
    delay(1000);
    lcd.clear();
    lcd.noBacklight(); // turn backlight off
    digitalWrite(CAMERATOGGLE, LOW);
    esp_deep_sleep_start(); 

  }
}



void onCommandReceived(const String &result)
{
  lcd.print(result);
    
}


esp_err_t handleMQTT(esp_mqtt_event_handle_t event)
{
    mqttClient.onEventCallback(event);
    return ESP_OK;
}
