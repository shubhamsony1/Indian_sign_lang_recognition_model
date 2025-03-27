import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        bbox = None
        center = None
        hand_label = None

        if self.results.multi_hand_landmarks:
            # Swap handedness here
            # Instead of using the actual handedness, reverse it:
            if self.results.multi_handedness[handNo].classification[0].label == "Left":
                hand_label = "Right"  # Swap left hand to right
            else:
                hand_label = "Left"  # Swap right hand to left

            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape
            xList, yList = [], []

            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                xList.append(cx)
                yList.append(cy)

            if xList and yList:
                x_min, x_max = min(xList), max(xList)
                y_min, y_max = min(yList), max(yList)
                bbox = (x_min, y_min, x_max, y_max)
                center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

                if draw:
                    # Draw the bounding box and center for the swapped hand
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.circle(img, center, 5, (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, f"{hand_label} Hand", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Place (x, y) coordinates at the top-right edge of the box
                    cv2.putText(img, f"({x_max}, {y_min})", (x_max + 10, y_min + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        return lmList, bbox, center, hand_label


def main():
    cap = cv2.VideoCapture(1)  # Change to 1 if the wrong camera is used
    detector = HandDetector()
    pTime = 0

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        # Find hand landmarks and bounding box
        img = detector.findHands(img)
        lmList, bbox, center, hand_label = detector.findPosition(img)

        # Draw the 30px buffer zone (square) at the center of the video frame
        h, w, _ = img.shape
        buffer_size = 60  # Size of the buffer zone (60px)
        top_left = (w // 2 - buffer_size // 2, h // 2 - buffer_size // 2)
        bottom_right = (w // 2 + buffer_size // 2, h // 2 + buffer_size // 2)

        # Draw the buffer zone (30px square)
        cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)  # Red rectangle
        cv2.putText(img, "Buffer Zone", (w // 2 - buffer_size // 2, h // 2 - buffer_size // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw a small center point (circle) inside the buffer zone
        buffer_center = (w // 2, h // 2)  # Center of the buffer zone
        cv2.circle(img, buffer_center, 3, (0, 255, 255), cv2.FILLED)  # Smaller yellow center point

        # Print swapped hand label and coordinates
        if hand_label:
            print(f"Swapped {hand_label} Hand Center: {center}")

        # Calculate FPS
        cTime = time.time()
        fps = int(1 / (cTime - pTime))
        pTime = cTime
        cv2.putText(img, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the final image with bounding box, hand info, and video center buffer zone
        cv2.imshow("Hand Tracking", img)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
