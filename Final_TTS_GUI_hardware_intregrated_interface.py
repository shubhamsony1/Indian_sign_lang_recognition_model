import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from tkinter import Tk, Label, Frame, Button
from PIL import Image, ImageTk
from gtts import gTTS
import os
from playsound import playsound
import tempfile
import serial

# ------------------------- Load Trained Model and Initialize Mediapipe -------------------------
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H',
    'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P',
    'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X',
    'Y': 'Y', 'Z': 'Z', 'SPACE': ' ', 'DELETE': 'DELETE', 'DONE': 'DONE'
}

# ------------------------- Global Variables for Timing, Sentence Accumulation, and Recording Control -------------------------
current_gesture = None       # Currently detected gesture
gesture_start_time = None    # Time when the current gesture started being detected
sentence = ""                # Accumulated sentence
recording = False            # Controls whether sentence accumulation is active

# Threshold times (in seconds)
FIRST_GESTURE_THRESHOLD = 4     # For the first word gesture in a sentence
SUBSEQUENT_GESTURE_THRESHOLD = 4   # For subsequent word gestures
DONE_GESTURE_THRESHOLD = 4        # For finalizing the sentence when DONE is detected

# ------------------------- Text-to-Speech Function -------------------------
def speak_sentence(text):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_path = temp_file.name
        tts = gTTS(text=text, lang='en')
        tts.save(temp_path)
        playsound(temp_path)
        os.remove(temp_path)
    except Exception as e:
        print("TTS Error:", e)

def tts_thread(sentence_text):
    threading.Thread(target=speak_sentence, args=(sentence_text,)).start()

# ------------------------- Tkinter GUI Setup -------------------------
window = Tk()
window.title("Sign Gesture Recognition with TTS")

# Create frames for layout
video_frame = Frame(window)
video_frame.pack()

output_frame = Frame(window)
output_frame.pack()

# Label for video feed (camera output)
video_label = Label(video_frame)
video_label.pack()

# Label for current predicted gesture output
gesture_label = Label(output_frame, text="Predicted: ", font=("Helvetica", 16))
gesture_label.pack()

# Label for the sentence output
sentence_label = Label(output_frame, text="Sentence: ", font=("Helvetica", 16))
sentence_label.pack()

# Label for status (Writing or Not Writing)
status_label = Label(output_frame, text="Status: Not Writing Words", font=("Helvetica", 14))
status_label.pack(pady=5)

# ------------------------- Button Functions -------------------------
def start_recording():
    global recording, sentence
    recording = True
    sentence = ""  # Reset the sentence when starting
    sentence_label.config(text="Sentence: ")
    status_label.config(text="Status: Writing Words")

def stop_recording():
    global recording
    recording = False
    status_label.config(text="Status: Not Writing Words")
    # Optionally finalize or speak the sentence on stop:
    # if sentence.strip() != "":
    #     tts_thread(sentence)

# ------------------------- Buttons -------------------------
start_button = Button(output_frame, text="Start", font=("Helvetica", 14), command=start_recording)
start_button.pack(side="left", padx=10, pady=10)

stop_button = Button(output_frame, text="Stop", font=("Helvetica", 14), command=stop_recording)
stop_button.pack(side="left", padx=10, pady=10)

# ------------------------- Servo Control Variables -------------------------


ws, hs = 1280, 720
port = "COM5"  # Change to your ESP32 COM port
ser = serial.Serial(port, 115200, timeout=1)
time.sleep(2)

current_servoX = 90
current_servoY = 60
step_size = 1
hand_detected_frames = 0
patrol_mode_active = False
phase = 0
y_adjustments = [10, -10, -10, 10]

# ------------------------- Main Update Loop -------------------------
def update():
    global current_gesture, gesture_start_time, sentence
    global current_servoX, current_servoY, step_size, hand_detected_frames
    global patrol_mode_active, phase

    ret, frame = cap.read()
    if not ret:
        window.after(10, update)
        return

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []
    predicted_character = ""

    if results.multi_hand_landmarks:
        patrol_mode_active = False
        hand_detected_frames += 1

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

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
        while len(data_aux) < 84:
            data_aux.append(0.0)

        prediction = model.predict([np.asarray(data_aux)])
        if isinstance(prediction[0], (int, np.integer)):
            predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
        else:
            try:
                predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
            except:
                predicted_character = labels_dict.get(prediction[0], "Unknown")

        gesture_label.config(text=f"Predicted: {predicted_character}")
        print(f"Predicted Raw Output: {prediction[0]}")

        # Bounding box for gesture prediction
        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # ------------------------- Servo Logic -------------------------
        x_min, y_min = int(min(x_) * ws), int(min(y_) * hs)
        x_max, y_max = int(max(x_) * ws), int(max(y_) * hs)
        x_min -= 20
        y_min -= 20
        x_max += 20
        y_max += 20

        hand_cx = (x_min + x_max) // 2
        hand_cy = (y_min + y_max) // 2
        camera_cx, camera_cy = ws // 2, hs // 2

        target_servoX = int(np.interp(hand_cx, [0, ws], [0, 180]))
        target_servoY = int(np.interp(hand_cy, [0, hs], [0, 180]))

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

        command = f"X{current_servoX}Y{current_servoY}#"
        ser.write(command.encode())
        print(f"Sent: {command}")

    else:
        hand_detected_frames = 0

        if not patrol_mode_active:
            if abs(current_servoX - 90) > step_size:
                current_servoX += step_size if 90 > current_servoX else -step_size
            else:
                current_servoX = 90
            if abs(current_servoY - 90) > step_size:
                current_servoY += step_size if 90 > current_servoY else -step_size
            else:
                current_servoY = 90

            if current_servoX == 90 and current_servoY == 90:
                patrol_mode_active = True
                current_servoX = 20
                phase = 0

        if patrol_mode_active:
            if phase in [0, 2]:
                current_servoX += step_size
                if current_servoX >= 160:
                    current_servoX = 160
                    if phase == 2:
                        current_servoY += y_adjustments[phase]
                    phase = (phase + 1) % 4
            else:
                current_servoX -= step_size
                if current_servoX <= 20:
                    current_servoX = 20
                    current_servoY += y_adjustments[phase]
                    phase = (phase + 1) % 4

            if phase == 0 and current_servoX == 20:
                current_servoY += y_adjustments[phase]

        command = f"X{current_servoX}Y{current_servoY}#"
        ser.write(command.encode())
        print(f"Patrol Mode: {command}")

    # ------------------------- Sentence Accumulation Logic -------------------------
    if recording:
        current_time = time.time()
        if predicted_character != "":
            if predicted_character == current_gesture:
                elapsed = current_time - gesture_start_time
            else:
                current_gesture = predicted_character
                gesture_start_time = current_time
                elapsed = 0

            if current_gesture == "DONE":
                threshold = DONE_GESTURE_THRESHOLD
            elif current_gesture == "DELETE":
                threshold = 5
            else:
                threshold = FIRST_GESTURE_THRESHOLD if sentence == "" else SUBSEQUENT_GESTURE_THRESHOLD

            if elapsed >= threshold:
                if current_gesture == "DONE":
                    if sentence.strip() != "":
                        sentence_label.config(text=f"Sentence: {sentence}")
                        tts_thread(sentence)
                        sentence = ""
                elif current_gesture == "DELETE":
                    sentence = sentence[:-1]
                    sentence_label.config(text=f"Sentence: {sentence}")
                else:
                    if current_gesture == " ":
                        sentence += " "
                    else:
                        sentence += current_gesture
                    sentence_label.config(text=f"Sentence: {sentence}")
                current_gesture = None
                gesture_start_time = None
        else:
            current_gesture = None
            gesture_start_time = None

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    window.after(10, update)

# Start the GUI update loop
update()
window.mainloop()

cap.release()
cv2.destroyAllWindows()
