#include <LiquidCrystal_I2C.h>
#include <Arduino.h>
#include <WiFi.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "esp_camera.h"


int IRINPUT = 5; // IR Sensor input pin, used for wakeup
int delay_timing = 100; // fixed delay timing
int val = LOW;
int countdown_to_sleep = 0;
int CAMERATOGGLE = 15;
LiquidCrystal_I2C lcd(0x27,16,2);  // set the LCD address to 0x27 for a 16 chars and 2 line display

void print_wakeup_reason(){ // this is the wakeup function when IR sensor is triggered. Keep it short, main part (EG sending signals for void loop)
  lcd.print("Hello World!"); // print on LCD 
  delay(1000);
  lcd.clear();
  digitalWrite(CAMERATOGGLE, HIGH);
  Serial.println("Wakeup");

}

void setup() { // Setup runs even after waking from deepsleep mode
  Serial.begin(115200);
  lcd.init(); //clear LCD on startup
  lcd.clear();  
  lcd.backlight();
  pinMode(IRINPUT, INPUT); // Set IR Sensor to input mode
  pinMode(CAMERATOGGLE, OUTPUT);
  esp_sleep_enable_ext0_wakeup(GPIO_NUM_5,1); //1 = High, 0 = Low Set wakeup to PIN 4, IRSENSOR
  print_wakeup_reason(); // Runs if wokeup

}

void loop() {
  val = digitalRead(IRINPUT); // Read IR Sensor
  
  if (val == LOW){
    countdown_to_sleep++; // start counting down each second to sleep mode if IR sensor not in use
  }
  else{
    countdown_to_sleep = 0;
  }
  delay(1000);
  

  if (countdown_to_sleep == 20){ // GOTO SLEEP on timer or button press
    countdown_to_sleep = 0;
    Serial.println("Sleep mode");
    lcd.clear();
    lcd.print("Sleep Time");
    delay(1000);
    lcd.clear();
    lcd.noBacklight(); // turn backlight off
    digitalWrite(CAMERATOGGLE, LOW);
    esp_deep_sleep_start(); 

  }

}

