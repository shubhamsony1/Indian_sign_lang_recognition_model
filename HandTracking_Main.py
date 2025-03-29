import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(min_detection_confidence=0.8)

# Open Camera
cap = cv2.VideoCapture(1)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

# Connect to ESP32 via Serial
port = "COM5"  # Change to your ESP32 COM Port
ser = serial.Serial(port, 115200, timeout=1)
time.sleep(2)  # Allow time to establish connection

# Initialize Servo Positions
current_servoX = 90  # Start at center position
current_servoY = 80
step_size = 2  # Adjust for smoothness (smaller = smoother, slower)
hand_detected_frames = 0  # Counter for consecutive hand detections

# Patrol mode management
patrol_mode_active = False

phase = 0
# Define the Y adjustment for each phase
y_adjustments = [10, -10, -10, 10]

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    multiHandDetection = results.multi_hand_landmarks
    hand_label = None  # Left or Right Hand

    if multiHandDetection:
        # When a hand is detected, disable patrol mode.
        patrol_mode_active = False
        hand_detected_frames += 1

        for i, lm in enumerate(multiHandDetection):
            mpDraw.draw_landmarks(img, lm, mpHands.HAND_CONNECTIONS)

        # Get Handedness (Left or Right Hand)
        if results.multi_handedness:
            hand_label = results.multi_handedness[0].classification[0].label  # 'Left' or 'Right'

        singleHandDetection = multiHandDetection[0]
        x_min, y_min = ws, hs
        x_max, y_max = 0, 0

        for lm in singleHandDetection.landmark:
            h, w, c = img.shape
            lm_x, lm_y = int(lm.x * w), int(lm.y * h)
            # Get Bounding Box
            x_min = min(x_min, lm_x)
            x_max = max(x_max, lm_x)
            y_min = min(y_min, lm_y)
            y_max = max(y_max, lm_y)

        # Expand Bounding Box Slightly
        x_min -= 20
        x_max += 20
        y_min -= 20
        y_max += 20

        # Draw Bounding Box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Find Center of the Bounding Box
        hand_cx = (x_min + x_max) // 2
        hand_cy = (y_min + y_max) // 2

        # Draw Small Red Dot at Camera Center
        camera_cx, camera_cy = ws // 2, hs // 2
        cv2.circle(img, (camera_cx, camera_cy), 5, (0, 0, 255), -1)  # Small Red Dot

        # Convert Position to Servo Degrees
        if hand_label == "Left":
            target_servoX = int(np.interp(hand_cx, [0, ws], [0, 180]))  # Fixed Left Hand Movement
        else:
            target_servoX = int(np.interp(hand_cx, [0, ws], [0, 180]))  # Normal for Right Hand

        target_servoY = int(np.interp(hand_cy, [0, hs], [0, 180]))  # Normal Up-Down

        # **Check if the Camera Center is Inside the Bounding Box**
        inside_x = x_min <= camera_cx <= x_max
        inside_y = y_min <= camera_cy <= y_max

        if inside_x and inside_y:
            # Hand is centered, stop movement
            pass
        else:
            # Move the servos smoothly towards the target
            if not inside_x:
                if abs(target_servoX - current_servoX) > step_size:
                    current_servoX += step_size if target_servoX > current_servoX else -step_size
                else:
                    current_servoX = target_servoX

            if not inside_y:
                if abs(target_servoY - current_servoY) > step_size:
                    current_servoY += step_size if target_servoY > current_servoY else -step_size
                else:
                    current_servoY = target_servoY

        # Send Servo Data to ESP32
        command = f"X{current_servoX}Y{current_servoY}#"
        ser.write(command.encode())
        print(f"Sent: {command}")

        # Show Values on Screen
        cv2.rectangle(img, (40, 20), (400, 150), (0, 255, 255), cv2.FILLED)
        cv2.putText(img, f'Servo X: {current_servoX} deg', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, f'Servo Y: {current_servoY} deg', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(img, f'Tracking: {hand_label}', (50, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    else:
        # No hand detected: reset counter
        hand_detected_frames = 0

        if not patrol_mode_active:
            # Reset servos to initial position (90,90)
            if current_servoX != 90:
                if abs(current_servoX - 90) > step_size:
                    current_servoX += step_size if 90 > current_servoX else -step_size
                else:
                    current_servoX = 90
            if current_servoY != 90:
                if abs(current_servoY - 90) > step_size:
                    current_servoY += step_size if 90 > current_servoY else -step_size
                else:
                    current_servoY = 90

            # Once both servos are at the initial position, initialize patrol mode.
            if current_servoX == 90 and current_servoY == 90:
                patrol_mode_active = True
                # Initialize the patrol cycle:
                # Start at the left endpoint (20°) and set phase = 0.
                current_servoX = 20
                phase = 0
                # current_servoY remains 90

        if patrol_mode_active:
            # Execute patrol cycle based on the current phase.
            if phase in [0, 2]:
                # Moving from 20 to 160 (increasing X)
                current_servoX += step_size
                if current_servoX >= 160:
                    current_servoX = 160

                    if phase == 2:
                        current_servoY += y_adjustments[phase]
                    phase = (phase + 1) % 4
            else:
                # phase in [1, 3]: Moving from 160 to 20 (decreasing X)
                current_servoX -= step_size
                if current_servoX <= 20:
                    current_servoX = 20
                    # When reaching the right endpoint in phase 1 or 3,
                    # update Y according to the current phase.
                    if phase == 1:
                        current_servoY += y_adjustments[phase]
                    elif phase == 3:
                        current_servoY += y_adjustments[phase]
                    phase = (phase + 1) % 4

            # Additionally, apply the Y update at the beginning of phase 0 if starting fresh.
            # (This handles the case when patrol mode is just activated.)
            if phase == 0 and current_servoX == 20:
                current_servoY += y_adjustments[phase]

        # Send Servo Data to ESP32
        command = f"X{current_servoX}Y{current_servoY}#"
        ser.write(command.encode())
        print(f"Patrol Mode: {command}")

        # Show Patrol Mode on Screen
        cv2.rectangle(img, (40, 20), (400, 100), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, "Patrol Mode", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()
