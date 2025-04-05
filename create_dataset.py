import os
import pickle

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dataset paths
DATA_DIR = './data'
PICKLE_DIR = r'C:\Users\ASUS\Documents\sem 8 new\Indian_sign_lang_recognition_model'
LANDMARK_IMG_DIR = r'C:\Users\ASUS\Documents\sem 8 new\Indian_sign_lang_recognition_model\Dataset\Landmark_imgs'
REJECTED_DIR = r'C:\Users\ASUS\Documents\sem 8 new\Indian_sign_lang_recognition_model\Dataset\Rejected_images'

# Create necessary directories if they don't exist
for directory in [PICKLE_DIR, LANDMARK_IMG_DIR, REJECTED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Define which gestures require two hands
two_hand_gestures = {'A', 'B', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'X', 'Y', 'Z'}  # Add two-hand gestures here
one_hand_gestures = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'C', 'I', 'L', 'O', 'U', 'V', 'SPACE', 'NEXT', 'DONE'}  # Add single-hand gestures here

# Initialize dataset lists
data = []
labels = []
accepted_count = 0
rejected_count = 0

# Process images
for dir_ in os.listdir(DATA_DIR):
    gesture_type = "two_hand" if dir_ in two_hand_gestures else "one_hand"

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            #print(f"Rejected (cannot read): {dir_}/{img_path}")
            rejected_count += 1
            cv2.imwrite(os.path.join(REJECTED_DIR, img_path), img)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        frame_with_landmarks = img.copy()

        # Count detected hands
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        # Reject images with incorrect hand count
        if (gesture_type == "two_hand" and num_hands != 2) or num_hands == 0:
            #print(f"Rejected (wrong hand count): {dir_}/{img_path}")
            rejected_count += 1
            cv2.imwrite(os.path.join(REJECTED_DIR, img_path), img)
            continue

        # Process landmarks if valid
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Ensure each sample has a fixed length (padding if needed)
        while len(data_aux) < 84:  # 21 landmarks * 2 coordinates * 2 hands
            data_aux.append(0.0)

        data.append(data_aux)
        labels.append(dir_)
        accepted_count += 1
        #print(f"Accepted: {dir_}/{img_path}")

        # Draw landmarks on the image
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame_with_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

        # Save image with landmarks
        landmark_img_path = os.path.join(LANDMARK_IMG_DIR, f"{dir_}_{img_path}")
        cv2.imwrite(landmark_img_path, frame_with_landmarks)

print(f"\nTotal images accepted: {accepted_count}")
print(f"Total images rejected: {rejected_count}")

# Save dataset
pickle_path = os.path.join(PICKLE_DIR, 'data.pickle')
with open(pickle_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset saved at: {pickle_path}")
