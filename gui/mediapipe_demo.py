import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import sys


class HandDetector:
    def __init__(self, min_detection_confidence=0.8, min_tracking_confidence=0.8):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    def process(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

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

        self.create_widgets()
        self.default_start_camera()

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

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.camera_selector.bind("<<ComboboxSelected>>", self.on_camera_change)

    def get_available_cameras(self):
        index = 0
        arr = []
        while True:
            print(f"reading camera {index}")
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

    def default_start_camera(self):
        self.start_camera()

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
                frame = self.detector.process(frame)
                self.display_frame(frame)
            self.root.after(10, self.update_frame)

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


if __name__ == "__main__":
    root = tk.Tk()
    app = HandDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
