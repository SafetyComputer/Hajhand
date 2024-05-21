import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from utils.shortcut import gesture_to_shortcuts


class TripletAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(TripletAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            # nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


fingers = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20]
]

model = TripletAutoencoder(input_dim=123, latent_dim=2)
model.load_state_dict(torch.load('../model/triplet_autoencoder_9677.pth'))
model.eval()

centroids = np.load('../model/triplet_autoencoder_9677_centroids.npy')
centroids = torch.tensor(centroids, dtype=torch.float32)

label_map = {
    0: 'call',
    1: 'dislike',
    2: 'fist',
    3: 'like',
    4: 'ok',
    5: 'one',
    6: 'palm',
    7: 'peace',
    8: 'rock',
    9: 'three',
    10: 'three2',
}


# MediaPipe solutions for hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Drawing utility
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

last_class = 12
continuous_frame_count = 0

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


        # print(results.multi_handedness)

        # Draw the hand landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        if results.multi_hand_world_landmarks:
            for hand_landmarks in results.multi_hand_world_landmarks:

                ## get the hand landmarks
                data = np.zeros((21, 3))
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    data[idx] = [landmark.x, landmark.y, landmark.z]

                # print(data)
                ## print the mean abs value of data
                # print(np.mean(np.abs(data)))


                ## add a new dimension
                data = np.expand_dims(data, axis=0)
                # print(data.shape)


                ## data augmentation
                ## add the two nearby length in each finger
                for finger in fingers:
                    for i in range(len(finger) - 1):
                        dist = data[:, finger[i + 1]] - data[:, finger[i]]
                        # add a new dimension
                        dist = np.expand_dims(dist, axis=1)
                        data = np.concatenate((data, dist), axis=1)

                # print(data.shape)


                data = torch.tensor(data.flatten(), dtype=torch.float32)
                data = data.view(1, 123)

                latent = model.encoder(data)

                # print(latent)

                ## calculate the distance between the latent and the centroids
                distance = torch.cdist(latent, centroids)

                ## get the nearest centroid
                _, pred = torch.min(distance, 1)

                if pred.item() == last_class:
                    continuous_frame_count += 1
                else:
                    continuous_frame_count = 0
                    last_class = pred.item()


                ## get the label
                label = label_map[pred.item()]

                if continuous_frame_count >= 5 and gesture_to_shortcuts[label]:
                    print(f"Gesture {label} detected")
                    gesture_to_shortcuts[label](continuous_frame_count)

        # Display the frame
        cv2.imshow('Live Hand Detection', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
