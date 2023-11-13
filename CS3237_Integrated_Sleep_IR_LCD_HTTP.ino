#include <LiquidCrystal_I2C.h>
int IRINPUT = 4; // IR Sensor input pin, used for wakeup ONLY THIS WORKS
int delay_timing = 100; // fixed delay timing
int val = LOW;
int countdown_to_sleep = 0;

LiquidCrystal_I2C lcd(0x27,16,2);  // set the LCD address to 0x27 for a 16 chars and 2 line display

void print_wakeup_reason(){ // this is the wakeup function when IR sensor is triggered. Keep it short, main part (EG sending signals for void loop)
  lcd.print("Hello World!"); // print on LCD 
  delay(1000);
  lcd.clear();
}


void setup() { // Setup runs even after waking from deepsleep mode

  lcd.init(); //clear LCD on startup
  lcd.clear();  
  lcd.backlight();
  pinMode(IRINPUT, INPUT); // Set IR Sensor to input mode
  esp_sleep_enable_ext0_wakeup(GPIO_NUM_4,1); //1 = High, 0 = Low Set wakeup to PIN 4, IRSENSOR
  print_wakeup_reason(); // Runs if wokeup
}

void loop() {
  val = digitalRead(IRINPUT); // Read IR Sensor
  if (val == HIGH){
    lcd.clear();
    lcd.print("Motion Detected"); // print on LCD motion detected
    delay(1000);
    lcd.clear();
    countdown_to_sleep = 0;
    }
  else{
    countdown_to_sleep++; // start counting down each second to sleep mode if IR sensor not in use
    delay(1000);
  }

  if (countdown_to_sleep == 10){ // GOTO SLEEP 
    lcd.clear();
    lcd.print("Sleep Time");
    delay(1000);
    lcd.clear();
    lcd.noBacklight(); // turn backlight off
    esp_deep_sleep_start(); 
  }
}

