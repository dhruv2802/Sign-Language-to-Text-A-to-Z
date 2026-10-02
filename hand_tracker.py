import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import MODEL_DIR


class HandTracker:

    def __init__(self):

        # Path to MediaPipe hand landmark model
        model_path = f"{MODEL_DIR}/hand_landmarker.task"

        # MediaPipe model configuration
        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        # Hand Landmarker configuration
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Create hand detector
        self.detector = vision.HandLandmarker.create_from_options(
            options
        )

        # Timestamp for video frames
        self.timestamp_ms = 0

    def process_frame(self, frame):

        # OpenCV uses BGR
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Increase timestamp for every frame
        self.timestamp_ms += 1

        # Detect hand landmarks
        results = self.detector.detect_for_video(
            mp_image,
            self.timestamp_ms
        )

        return results

    def get_landmarks(self, results):

        # No hand detected
        if not results.hand_landmarks:
            return None

        # Get first detected hand
        hand = results.hand_landmarks[0]

        landmarks = []

        # Extract x, y, z for all 21 landmarks
        for landmark in hand:

            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        return landmarks

    def draw_landmarks(self, frame, results):

        # Check whether a hand was detected
        if results.hand_landmarks:

            for hand_landmarks in results.hand_landmarks:

                # Draw all 21 landmark points
                for landmark in hand_landmarks:

                    x = int(
                        landmark.x * frame.shape[1]
                    )

                    y = int(
                        landmark.y * frame.shape[0]
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

        return frame

    def close(self):

        self.detector.close()