import os
import cv2
import mediapipe as mp
import tensorflow as tf
import threading

tf.constant(0)

cap = cv2.VideoCapture(1)

# Flag to stop live feed
global stop_live_feed
stop_live_feed = False


def start_live_feed():
    while not stop_live_feed:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (500, 380))  # Resize frame
        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


live_feed_thread = threading.Thread(target=start_live_feed, daemon=True)
live_feed_thread.start()

reg_no = input("\nEnter Registration Number: ")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

BASE_DIR = r'C:\Users\ASUS\Desktop\sem 8 new\Indian_sign_lang_recognition_model\DATA_COL\imgs_without_landmarks'
BASE_DIR_L = r'C:\Users\ASUS\Desktop\sem 8 new\Indian_sign_lang_recognition_model\DATA_COL'
DATA_DIR = os.path.join(BASE_DIR, reg_no)
LANDMARKS_DIR = os.path.join(BASE_DIR_L, 'imgs_with_landmark', reg_no)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(LANDMARKS_DIR):
    os.makedirs(LANDMARKS_DIR)

dataset_size = 100

while True:

    gesture_name = input("\nEnter gesture name (or type 'quit' to exit): ")
    if gesture_name.lower() == 'quit':
        break

    class_dir = os.path.join(DATA_DIR, gesture_name)
    landmark_class_dir = os.path.join(LANDMARKS_DIR, gesture_name)

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    if not os.path.exists(landmark_class_dir):
        os.makedirs(landmark_class_dir)

    print(f'Collecting data for gesture: {gesture_name}')

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_with_landmarks = frame.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_with_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame = cv2.resize(frame, (500, 380))  # Resize original frame
        frame_with_landmarks = cv2.resize(frame_with_landmarks, (500, 380))  # Resize landmarks frame

        cv2.imshow('Original Frame', frame)
        cv2.imshow('Frame with Landmarks', frame_with_landmarks)

        img_path = os.path.join(class_dir, f'{counter}.jpg')
        landmark_img_path = os.path.join(landmark_class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        cv2.imwrite(landmark_img_path, frame_with_landmarks)
        counter += 1
        cv2.waitKey(25)

    print("Image collection complete!")

stop_live_feed = True  # Stop live feed thread
cap.release()
cv2.destroyAllWindows()
exit()