
import cv2
import os
import numpy as np
from collections import deque

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    WINDOW_NAME,
    SEQUENCE_LENGTH
)

from hand_tracker import HandTracker

from utils import (
    landmarks_to_array,
    predict_sign,
    get_prediction_probability,
    get_top_predictions
)


# ============================================================
# LOAD CALIBRATED MODEL
# ============================================================

import joblib


MODEL_PATH = os.path.join(
    "models",
    "sign_language_model_calibrated.pkl"
)

CLASSES_PATH = os.path.join(
    "models",
    "classes_calibrated.npy"
)


print("\nLoading calibrated model...")

model = joblib.load(
    MODEL_PATH
)

classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

print("Calibrated model loaded successfully!")
print("Classes:", classes)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.45

PREDICTION_INTERVAL = 2

HISTORY_SIZE = 8

MIN_STABLE_PREDICTIONS = 4

MAX_MISSING_FRAMES = 5


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


if not cap.isOpened():

    print(
        "ERROR: Camera could not be opened."
    )

    raise SystemExit


# ============================================================
# HAND TRACKER
# ============================================================

tracker = HandTracker()


# ============================================================
# VARIABLES
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

prediction_history = deque(
    maxlen=HISTORY_SIZE
)


frame_counter = 0

missing_frames = 0

current_prediction = ""

stable_prediction = ""

confidence = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print(
            "ERROR: Could not read camera frame."
        )

        break


    frame_counter += 1


    # ========================================================
    # KEEP ORIGINAL FRAME
    # ========================================================

    original_frame = frame.copy()


    # ========================================================
    # MEDIAPIPE PROCESSING
    # ========================================================

    results = tracker.process_frame(
        original_frame
    )


    original_frame = tracker.draw_landmarks(
        original_frame,
        results
    )


    landmarks = tracker.get_landmarks(
        results
    )


    # ========================================================
    # COLLECT LANDMARKS
    # ========================================================

    if landmarks is not None:

        missing_frames = 0

        landmark_array = landmarks_to_array(
            landmarks
        )

        sequence.append(
            landmark_array
        )


    else:

        missing_frames += 1


        if missing_frames > MAX_MISSING_FRAMES:

            sequence.clear()

            prediction_history.clear()

            current_prediction = ""

            stable_prediction = ""

            confidence = 0.0


    # ========================================================
    # PREDICTION
    # ========================================================

    if (
        len(sequence) == SEQUENCE_LENGTH
        and
        frame_counter % PREDICTION_INTERVAL == 0
    ):

        try:

            sequence_array = list(
                sequence
            )


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            prediction = predict_sign(
                model,
                classes,
                sequence_array
            )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = (
                get_prediction_probability(
                    model,
                    sequence_array
                )
            )


            # ------------------------------------------------
            # TOP 3
            # ------------------------------------------------

            top_predictions = (
                get_top_predictions(
                    model,
                    classes,
                    sequence_array
                )
            )


            print(
                "Top predictions:",
                [
                    (
                        str(letter),
                        f"{prob * 100:.1f}%"
                    )
                    for letter, prob
                    in top_predictions
                ]
            )


            current_prediction = str(
                prediction
            )


            # ------------------------------------------------
            # STABILITY
            # ------------------------------------------------

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_history.append(
                    current_prediction
                )


                if (
                    len(prediction_history)
                    >= MIN_STABLE_PREDICTIONS
                ):

                    counts = {}


                    for letter in prediction_history:

                        counts[letter] = (
                            counts.get(
                                letter,
                                0
                            ) + 1
                        )


                    most_common = max(
                        counts,
                        key=counts.get
                    )


                    if (
                        counts[most_common]
                        >= MIN_STABLE_PREDICTIONS
                    ):

                        stable_prediction = (
                            most_common
                        )


        except Exception as error:

            print(
                "Prediction error:",
                error
            )


    # ========================================================
    # MIRRORED DISPLAY
    # ========================================================

    display_frame = cv2.flip(
        original_frame,
        1
    )


    # ========================================================
    # INFORMATION BOX
    # ========================================================

    cv2.rectangle(
        display_frame,
        (0, 0),
        (640, 145),
        (0, 0, 0),
        -1
    )


    # ========================================================
    # CURRENT PREDICTION
    # ========================================================

    cv2.putText(
        display_frame,
        f"Prediction: {current_prediction}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    # ========================================================
    # STABLE PREDICTION
    # ========================================================

    cv2.putText(
        display_frame,
        f"Stable: {stable_prediction}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    cv2.putText(
        display_frame,
        f"Confidence: {confidence * 100:.1f}%",
        (390, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # FRAME COUNT
    # ========================================================

    cv2.putText(
        display_frame,
        f"Frames: {len(sequence)}/30",
        (390, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # MODEL NAME
    # ========================================================

    cv2.putText(
        display_frame,
        "Model: Calibrated",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

tracker.close()

cap.release()

cv2.destroyAllWindows()
