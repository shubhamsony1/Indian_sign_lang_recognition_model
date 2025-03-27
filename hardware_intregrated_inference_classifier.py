import pickle
import serial
import time
import cv2
import numpy as np
import mediapipe as mp
from HandTrackingModule import HandDetector

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the serial connection to the ESP32-CAM
ser = serial.Serial('COM8', 115200)  # Adjust COM port if necessary
time.sleep(2)  # Allow time for the ESP32-CAM to initialize

# Hand tracking setup
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Servo angle tracking for smooth movement
s_speed = 1  # speed of servo motor
current_x = 90  # Start at center position
current_y = 50
movement_speed = s_speed  # Movement speed
hand_in_buffer = False  # Tracking previous state to stop unnecessary movement
patrol_active = False

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
               'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H',
               'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P',
               'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X',
               'Y': 'Y', 'Z': 'Z', 'SPACE': 'SPACE', 'NEXT': 'NEXT', 'DONE': 'DONE'}


def send_servo_position(x, y):
    global current_x, current_y
    if abs(x - current_x) > movement_speed:
        current_x += movement_speed if x < current_x else -movement_speed
    else:
        current_x = x

    current_x = max(40, min(140, current_x))
    current_y = max(40, min(100, current_y))

    if abs(y - current_y) > movement_speed:
        current_y += movement_speed if y < current_y else -movement_speed
    else:
        current_y = y

    command = f"X{int(current_x)}Y{int(current_y)}#"
    ser.write(command.encode())


def main():
    global hand_in_buffer, patrol_active
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        img = detector.findHands(frame)
        lmList, bbox, center, hand_label = detector.findPosition(img)

        data_aux = []
        x_, y_ = [], []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            while len(data_aux) < 84:
                data_aux.append(0.0)

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
            print(f"Predicted Sign: {predicted_character}")

            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

        if center:
            patrol_active = False
            hand_center_x, hand_center_y = center
            servo_x = int(hand_center_x / W * 180)
            servo_y = int((H - hand_center_y) / H * 180)
            servo_x = max(40, min(140, servo_x))
            send_servo_position(servo_x, servo_y)
        else:
            patrol_active = True

        cv2.imshow("Hand Tracking & Sign Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
