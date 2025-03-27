#include <Arduino.h>

#define SERVO_X_PIN 14  // Change to your ESP32-CAM PWM-capable pin
#define SERVO_Y_PIN 15  // Change to your ESP32-CAM PWM-capable pin

int currentX = 90;
int currentY = 90;
int targetX = 90;
int targetY = 90;
const int movementSpeed = 3; // Smaller value = smoother movement

const int freq = 50;  // Standard PWM frequency for servos
const int resolution = 16; // 16-bit resolution
const int channelX = 0;
const int channelY = 1;

void setup() {
  Serial.begin(115200);
  ledcSetup(channelX, freq, resolution);
  ledcSetup(channelY, freq, resolution);
  ledcAttachPin(SERVO_X_PIN, channelX);
  ledcAttachPin(SERVO_Y_PIN, channelY);
}

void loop() {
  readSerial();
  smoothServoMove();
  delay(20); // Add delay for smoother movement
}

void readSerial() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('#');
    if (data.startsWith("X")) {
      int splitPos = data.indexOf('Y');
      targetX = data.substring(1, splitPos).toInt();
      targetY = data.substring(splitPos + 1).toInt();
    }
  }
}

void smoothServoMove() {
  if (abs(targetX - currentX) > movementSpeed) {
    currentX += (targetX > currentX) ? movementSpeed : -movementSpeed;
  } else {
    currentX = targetX;
  }

  if (abs(targetY - currentY) > movementSpeed) {
    currentY += (targetY > currentY) ? movementSpeed : -movementSpeed;
  } else {
    currentY = targetY;
  }

  int dutyCycleX = map(currentX, 0, 180, 1638, 8192);
  int dutyCycleY = map(currentY, 0, 180, 1638, 8192);
  
  ledcWrite(channelX, dutyCycleX);
  ledcWrite(channelY, dutyCycleY);
}
