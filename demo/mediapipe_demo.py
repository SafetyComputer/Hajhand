import cv2
import mediapipe as mp
import numpy as np

# MediaPipe solutions for hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8)

# Drawing utility
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the color space from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)

        # Draw the hand landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow('Live Hand Detection', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
