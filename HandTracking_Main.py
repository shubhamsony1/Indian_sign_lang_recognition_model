import serial
import time
import cv2
import numpy as np
from HandTrackingModule import HandDetector

# Initialize the serial connection to the ESP32-CAM
ser = serial.Serial('COM8', 115200)  # Adjust COM port if necessary
time.sleep(2)  # Allow time for the ESP32-CAM to initialize

# Hand tracking setup
detector = HandDetector(detectionCon=0.8, maxHands=2)  # Adjust detection confidence and max hands

# Servo angle tracking for smooth movement
s_speed = 1   # speed of servo motor
current_x = 90  # Start at center position
current_y = 50
movement_speed = s_speed  # Movement speed

# Tracking previous state to stop unnecessary movement
hand_in_buffer = False

# Patrol movement settings
patrol_direction_x = 1  # 1 for right, -1 for left
patrol_direction_y = 1  # 1 for down, -1 for up
patrol_speed = s_speed  # Speed of patrol movement
patrol_x_min = 40
patrol_x_max = 140
patrol_y_min = 40
patrol_y_max = 100
patrol_active = False

# Set the color for the fixed bounding box
# Using RGBA format (Red, Green, Blue, Alpha), Alpha = 0 for transparency
color = (0, 0, 255, 255)  # Invisible (Transparent)

def send_servo_position(x, y):
    global current_x, current_y

    # Smooth transition by moving step-by-step towards target
    if abs(x - current_x) > movement_speed:
        current_x += movement_speed if x < current_x else -movement_speed
    else:
        current_x = x

    # Clamp X axis to range between 40 and 140 degrees
    current_x = max(40, min(140, current_x))

    # Clamp Y axis to range between 40 and 100 degrees
    current_y = max(40, min(100, current_y))

    if abs(y - current_y) > movement_speed:
        current_y += movement_speed if y < current_y else -movement_speed
    else:
        current_y = y

    # Clamp Y after adjustment to ensure it stays within bounds
    current_y = max(40, min(100, current_y))

    command = f"X{int(current_x)}Y{int(current_y)}#"
    ser.write(command.encode())
    #print(f"Sent position: X={int(current_x)}, Y={int(current_y)}")

def patrol_movement():
    global current_x, current_y, patrol_direction_x, patrol_direction_y

    current_x += patrol_direction_x * patrol_speed
    if current_x >= patrol_x_max or current_x <= patrol_x_min:
        patrol_direction_x *= -1  # Reverse X direction
        current_y += patrol_direction_y * 10  # Move Y by 10 degrees when X completes a cycle

        if current_y >= patrol_y_max or current_y <= patrol_y_min:
            patrol_direction_y *= -1  # Reverse Y direction when reaching limits

    send_servo_position(current_x, current_y)

def main():
    global hand_in_buffer, patrol_active
    cap = cv2.VideoCapture(1)  # Adjust the camera index if necessary

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        # Convert to BGRA (RGBA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # Optional if you want transparency handling

        # Now ensure the image is in BGR format (3 channels) for hand detection
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Ensure it's in BGR format

        # Find hand landmarks
        img = detector.findHands(img)

        lmList, bbox, center, hand_label = detector.findPosition(img)

        # Draw the buffer zone at the center of the video frame
        h, w, _ = img.shape
        buffer_size = 5  # Increased buffer zone size for stability
        top_left = (w // 2 - buffer_size // 2, h // 2 - buffer_size // 2)
        bottom_right = (w // 2 + buffer_size // 2, h // 2 + buffer_size // 2)
        cv2.rectangle(img, top_left, bottom_right, color, 2)  # Red buffer zone

        if center:
            patrol_active = False  # Stop patrol when a hand is detected
            hand_center_x, hand_center_y = center  # Use detected hand center

            # Fixed bounding box size (100x100 pixels) centered on the hand
            box_size = 160
            x_min = max(0, hand_center_x - box_size // 2)
            y_min = max(0, hand_center_y - box_size // 2)
            x_max = min(w, x_min + box_size)
            y_max = min(h, y_min + box_size)

            # Ensure box stays inside the frame
            if x_max > w:
                x_min -= (x_max - w)
                x_max = w
            if y_max > h:
                y_min -= (y_max - h)
                y_max = h

            # Only draw the fixed-size hand box if the color is not fully transparent
            if color[3] != 0:  # If Alpha is not 0 (transparent)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Convert hand position to servo angle
            servo_x = int(hand_center_x / w * 180)  # Scale X to 0-180
            servo_y = int((h - hand_center_y) / h * 180)  # Inverted Y-axis scaling

            # Clamp servo_x to the range between 40 and 140 degrees
            servo_x = max(40, min(140, servo_x))

            # Check if buffer box is fully inside hand box
            buffer_inside_hand = (x_min < top_left[0] < x_max and
                                  y_min < top_left[1] < y_max and
                                  x_min < bottom_right[0] < x_max and
                                  y_min < bottom_right[1] < y_max)

            if buffer_inside_hand:
                if not hand_in_buffer:
                    hand_in_buffer = True  # Stop moving servo if buffer is inside hand box
                    print("Buffer inside hand box. Stopping servo.")
            else:
                if hand_in_buffer:
                    hand_in_buffer = False  # Reset state if buffer is no longer inside hand box
                    print("Buffer outside hand box. Moving servo.")
                send_servo_position(servo_x, servo_y)
        else:
            patrol_active = True  # Start patrol when no hand is detected

        if patrol_active:
            patrol_movement()

        # Display the image with RGBA (including transparency)
        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
