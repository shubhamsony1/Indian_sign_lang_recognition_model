import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import pickle

# ================== Initialization ==================
# --- Servo Tracking Setup ---
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands_tracking = mpHands.Hands(min_detection_confidence=0.8)

# --- Gesture Recognition Setup ---
# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Gesture detection is run in static_image_mode for more stable landmarks.
hands_gesture = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
               'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H',
               'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P',
               'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X',
               'Y': 'Y', 'Z': 'Z', 'SPACE': 'SPACE', 'NEXT': 'DELETE', 'DONE': 'DONE'}

# --- Camera and Serial Setup ---
cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

port = "COM5"  # Change to your ESP32 COM Port
ser = serial.Serial(port, 115200, timeout=1)
time.sleep(2)  # Allow time to establish connection

# --- Servo Tracking Variables ---
current_servoX = 90  # Start at center
current_servoY = 80
step_size = 2       # Adjust for smoothness
hand_detected_frames = 0
patrol_mode_active = False
phase = 0
y_adjustments = [10, -10, -10, 10]

# Timer for waiting 5 sec before entering patrol mode when no hand is detected.
no_hand_timer = None

# Variable to store servo tracking bounding box (for overlay).
servo_bbox = None

# ================== Main Loop ==================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame once for consistency
    frame_combined = cv2.flip(frame, 1)

    # ---------------- Servo Tracking ----------------
    frame_tracking_rgb = cv2.cvtColor(frame_combined.copy(), cv2.COLOR_BGR2RGB)
    results_tracking = hands_tracking.process(frame_tracking_rgb)

    multiHandDetection = results_tracking.multi_hand_landmarks
    hand_label = None

    if multiHandDetection:
        # Hand detected: reset the no-hand timer and disable patrol mode.
        no_hand_timer = None
        patrol_mode_active = False
        hand_detected_frames += 1

        # Get handedness if available.
        if results_tracking.multi_handedness:
            hand_label = results_tracking.multi_handedness[0].classification[0].label

        # Use first detected hand for tracking.
        singleHandDetection = multiHandDetection[0]
        x_min, y_min = ws, hs
        x_max, y_max = 0, 0

        for lm in singleHandDetection.landmark:
            h_img, w_img, _ = frame_combined.shape
            lm_x, lm_y = int(lm.x * w_img), int(lm.y * h_img)
            x_min = min(x_min, lm_x)
            x_max = max(x_max, lm_x)
            y_min = min(y_min, lm_y)
            y_max = max(y_max, lm_y)

        # Expand bounding box slightly.
        x_min -= 20
        x_max += 20
        y_min -= 20
        y_max += 20

        servo_bbox = (x_min, y_min, x_max, y_max)

        # Determine center of bounding box.
        hand_cx = (x_min + x_max) // 2
        hand_cy = (y_min + y_max) // 2

        # Draw a small red dot at the camera center.
        camera_cx, camera_cy = ws // 2, hs // 2
        cv2.circle(frame_combined, (camera_cx, camera_cy), 5, (0, 0, 255), -1)

        # Map hand center to servo motor angles.
        target_servoX = int(np.interp(hand_cx, [0, ws], [0, 180]))
        target_servoY = int(np.interp(hand_cy, [0, hs], [0, 180]))

        # Check if camera center lies within the hand bounding box.
        inside_x = x_min <= camera_cx <= x_max
        inside_y = y_min <= camera_cy <= y_max

        if not (inside_x and inside_y):
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

        # Send servo command.
        command = f"X{current_servoX}Y{current_servoY}#"
        ser.write(command.encode())
        print(f"Sent: {command}")

        # Overlay tracking info.
        cv2.rectangle(frame_combined, (40, 20), (400, 150), (0, 255, 255), cv2.FILLED)
        cv2.putText(frame_combined, f'Servo X: {current_servoX} deg', (50, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(frame_combined, f'Servo Y: {current_servoY} deg', (50, 100),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(frame_combined, f'Tracking: {hand_label}', (50, 130),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    else:
        # No hand detected.
        if no_hand_timer is None:
            no_hand_timer = time.time()
        elapsed = time.time() - no_hand_timer
        if elapsed < 5:
            remaining = int(5 - elapsed)
            cv2.putText(frame_combined, f'Waiting {remaining} sec...', (50, hs - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            patrol_mode_active = True

        servo_bbox = None  # Clear bounding box when no hand is detected

        # When not yet in patrol mode, reset servos to initial position (90,90).
        if not patrol_mode_active:
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
            # Once servos reach (90,90), activate patrol mode.
            if current_servoX == 90 and current_servoY == 90:
                patrol_mode_active = True
                current_servoX = 20  # Start patrol cycle from left endpoint.
                phase = 0

        # ---------------- Patrol Mode (Discrete Phase-Based) ----------------
        if patrol_mode_active:
            # For phases 0 and 2: increase X from 20 to 160.
            if phase in [0, 2]:
                current_servoX += step_size
                if current_servoX >= 160:
                    current_servoX = 160
                    if phase == 2:
                        current_servoY += y_adjustments[phase]
                    phase = (phase + 1) % 4
            else:
                # For phases 1 and 3: decrease X from 160 to 20.
                current_servoX -= step_size
                if current_servoX <= 20:
                    current_servoX = 20
                    if phase == 1:
                        current_servoY += y_adjustments[phase]
                    elif phase == 3:
                        current_servoY += y_adjustments[phase]
                    phase = (phase + 1) % 4
            # Additional Y update at the beginning of phase 0.
            if phase == 0 and current_servoX == 20:
                current_servoY += y_adjustments[phase]

            command = f"X{current_servoX}Y{current_servoY}#"
            ser.write(command.encode())
            print(f"Patrol Mode: {command}")

            cv2.rectangle(frame_combined, (40, 20), (400, 100), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame_combined, "Patrol Mode", (50, 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # ---------------- Gesture Recognition ----------------
    frame_gesture_rgb = cv2.cvtColor(frame_combined.copy(), cv2.COLOR_BGR2RGB)
    results_gesture = hands_gesture.process(frame_gesture_rgb)

    predicted_character = None
    if results_gesture.multi_hand_landmarks:
        for hand_landmarks in results_gesture.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_combined, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        data_aux = []
        x_vals = []
        y_vals = []
        for hand_landmarks in results_gesture.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_vals.append(lm.x)
                y_vals.append(lm.y)
        for hand_landmarks in results_gesture.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_vals))
                data_aux.append(lm.y - min(y_vals))
        while len(data_aux) < 84:
            data_aux.append(0.0)
        prediction = model.predict([np.asarray(data_aux)])
        if isinstance(prediction[0], (int, np.integer)):
            predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
        else:
            predicted_character = prediction[0] if prediction[0] in labels_dict.values() else "Unknown"
        print(f"Predicted Raw Output: {prediction[0]}")

    # ---------------- Combined Overlays ----------------
    if servo_bbox is not None:
        x_min, y_min, x_max, y_max = servo_bbox
        cv2.rectangle(frame_combined, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        if predicted_character is not None:
            cv2.putText(frame_combined, predicted_character, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    elif predicted_character is not None:
        cv2.putText(frame_combined, predicted_character, (50, hs - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # ---------------- Show Combined Frame ----------------
    cv2.imshow("Combined Frame", frame_combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================== Cleanup ==================
cap.release()
ser.close()
cv2.destroyAllWindows()
