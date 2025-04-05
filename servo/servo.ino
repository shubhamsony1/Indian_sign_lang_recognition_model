#include <ESP32Servo.h>

Servo servoX, servoY;
int servoPinX = 14;  // X-axis Servo (ESP32-CAM GPIO 14), yellow wire
int servoPinY = 15;  // Y-axis Servo (ESP32-CAM GPIO 15), orange wire
int posX = 90, posY = 60;

void setup() {
    Serial.begin(115200);
    servoX.attach(servoPinX);
    servoY.attach(servoPinY);
    servoX.write(posX);
    servoY.write(posY);
}

void loop() {
    if (Serial.available()) {
        String data = Serial.readStringUntil('#');  // Read Serial Data
        Serial.print("Received: ");
        Serial.println(data);

        int xIndex = data.indexOf('X');
        int yIndex = data.indexOf('Y');

        if (xIndex != -1 && yIndex != -1) {
            posX = data.substring(xIndex + 1, yIndex).toInt();
            posY = data.substring(yIndex + 1).toInt();

            Serial.print("Moving to X: ");
            Serial.print(posX);
            Serial.print(", Y: ");
            Serial.println(posY);

            servoX.write(posX);
            servoY.write(posY);
        }
    }
}
