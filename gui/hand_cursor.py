import sys
import tkinter as tk
from tkinter import ttk

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import torch
import torch.nn as nn
from PIL import Image, ImageTk


class Line:
    def __init__(self, point1: tuple[float, float], point2: tuple[float, float]) -> None:
        t = (point1[0] - point2[0]) / (point1[1] - point2[1])
        m = point1[0] - t * point1[1]
        self.t = t
        self.m = m

    def __call__(self, y: float) -> float:
        return self.t * y + self.m


class PerspectiveTransformer:
    def __init__(self,
                 A1: tuple[float, float] = (0.1, 0.2),
                 B1: tuple[float, float] = (0.9, 0.2),
                 C1: tuple[float, float] = (0.9, 0.9),
                 D1: tuple[float, float] = (0.1, 0.9)):
        self.A = (0, 0)
        self.B = (1, 0)
        self.C = (1, 1)
        self.D = (0, 1)

        self.A1 = A1
        self.B1 = B1
        self.C1 = C1
        self.D1 = D1

    def transform(self, point: tuple[float, float]) -> tuple[float, float]:
        y = (point[1] - self.B1[1]) * (self.C[1] - self.B[1]) / (self.C1[1] - self.B1[1])
        left_side = Line(self.A1, self.D1)
        right_side = Line(self.B1, self.C1)
        lm = left_side(point[1])
        rm = right_side(point[1])
        x = (point[0] - lm) / (rm - lm) * (self.C[0] - self.D[0])
        return x, y

    def draw_quadrangle(self, frame: cv2.VideoCapture):
        # draw the quadrangle on frame
        h, w, _ = frame.shape

        a = (int(self.A1[0] * w), int(self.A1[1] * h))
        b = (int(self.B1[0] * w), int(self.B1[1] * h))
        c = (int(self.C1[0] * w), int(self.C1[1] * h))
        d = (int(self.D1[0] * w), int(self.D1[1] * h))

        cv2.line(frame, a, b, (0, 255, 0), 2)
        cv2.line(frame, b, c, (0, 255, 0), 2)
        cv2.line(frame, c, d, (0, 255, 0), 2)
        cv2.line(frame, d, a, (0, 255, 0), 2)


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


class HandDetector:
    fingers = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
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

    def __init__(self,
                 model_path='../model/triplet_autoencoder_9677.pth',
                 centroids_path='../model/triplet_autoencoder_9677_centroids.npy'):

        self.model = TripletAutoencoder(input_dim=123, latent_dim=2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.centroids = np.load(centroids_path)
        self.centroids = torch.tensor(self.centroids, dtype=torch.float32)

        # MediaPipe solutions for hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Drawing utility
        self.mp_drawing = mp.solutions.drawing_utils

        self.transformer = PerspectiveTransformer()

    def process(self, frame):
        # Convert the color space from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = self.hands.process(frame_rgb)

        # draw the A1 B1 C1 D1
        self.transformer.draw_quadrangle(frame)

        label = None
        position = None

        if results.multi_hand_world_landmarks:
            world_landmarks = results.multi_hand_world_landmarks[0]

            # get the hand landmarks
            data = np.zeros((21, 3))
            for idx, landmark in enumerate(world_landmarks.landmark):
                data[idx] = [landmark.x, landmark.y, landmark.z]

            # add a new dimension
            data = np.expand_dims(data, axis=0)

            # data augmentation
            # add the two nearby length in each finger
            for finger in self.fingers:
                for i in range(len(finger) - 1):
                    dist = data[:, finger[i + 1]] - data[:, finger[i]]
                    # add a new dimension
                    dist = np.expand_dims(dist, axis=1)
                    data = np.concatenate((data, dist), axis=1)

            data = torch.tensor(data.flatten(), dtype=torch.float32)
            data = data.view(1, 123)

            latent = self.model.encoder(data)

            # calculate the distance between the latent and the centroids
            distance = torch.cdist(latent, self.centroids)

            # get the nearest centroid
            _, pred = torch.min(distance, 1)

            ## get the label
            label = self.label_map[pred.item()]


        # Draw the hand landmarks on the image
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            h, w, c = frame.shape
            # print(h, w)
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            # print(cx, cy)
            position = self.transformer.transform((index_finger_tip.x, index_finger_tip.y))
            # Draw a circle at the index finger tip
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)


        return frame, position, label

    def close(self):
        self.hands.close()


class HandDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Detection with MediaPipe")

        self.detector = HandDetector()

        self.camera_index = tk.StringVar()
        self.cap = None
        self.running = False
        self.flip = False

        self.history = []
        self.last_label = None
        self.last_label_count = 0

        self.create_widgets()
        self.start_camera()

    def create_widgets(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        camera_label = ttk.Label(control_frame, text="Select Camera:")
        camera_label.pack(side=tk.LEFT)

        self.camera_selector = ttk.Combobox(control_frame, textvariable=self.camera_index)
        self.camera_selector['values'] = self.get_available_cameras()
        self.camera_selector.current(0)
        self.camera_selector.pack(side=tk.LEFT, padx=5)

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_camera)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.flip_button = ttk.Button(control_frame, text="Flip", command=self.flip_camera)
        self.flip_button.pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.camera_selector.bind("<<ComboboxSelected>>", self.on_camera_change)

    def get_available_cameras(self):
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(f"Camera {index}")
            cap.release()
            index += 1
        if not arr:
            print("Error: No cameras detected.")
            sys.exit(1)
        return arr

    def start_camera(self):
        if not self.running:
            self.running = True
            self.update_camera()
            self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.canvas.delete("all")

    def flip_camera(self):
        self.flip = not self.flip

    def update_camera(self):
        if self.cap:
            self.cap.release()
        selected_camera = int(self.camera_index.get().split()[-1])
        self.cap = cv2.VideoCapture(selected_camera)
        if not self.cap.isOpened():
            print(f"Error: Unable to open camera {selected_camera}.")
            sys.exit(1)

    def on_camera_change(self, event):
        if self.running:
            self.update_camera()

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.flip:
                    frame = cv2.flip(frame, 1)

                frame, pos, label = self.detector.process(frame)

                if pos:
                    pos_x, pos_y = pos
                    self.history.append((pos, label))
                    if len(self.history) > 3:
                        self.history.pop(0)

                    avg_x = sum([x for (x, _), _ in self.history]) / len(self.history)
                    avg_y = sum([y for (_, y), _ in self.history]) / len(self.history)

                    self.track_mouse(avg_x, avg_y, 0)

                if label:
                    if label == self.last_label:
                        self.last_label_count += 1

                        if self.last_label_count > 5:
                            self.gesture_operation(label)
                            self.last_label_count = -20
                    else:
                        self.last_label = label
                        self.last_label_count = 0

                self.display_frame(frame)

            self.root.after(1, self.update_frame)

    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk

    def on_closing(self):
        self.stop_camera()
        self.detector.close()
        self.root.destroy()

    def track_mouse(self, norm_x, norm_y, duration=0.1):
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))

        screenWidth, screenHeight = pyautogui.size()

        pyautogui.moveTo(
            norm_x * screenWidth,
            norm_y * screenHeight,
            duration=duration,
            _pause=False
        )

    def gesture_operation(self, label: str):
        if label == 'one':
            pyautogui.leftClick()

        if label == 'peace':
            pyautogui.rightClick()

        if label == 'like':
            pyautogui.scroll(200)

        if label == 'dislike':
            pyautogui.scroll(-200)


if __name__ == "__main__":
    root = tk.Tk()
    app = HandDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
