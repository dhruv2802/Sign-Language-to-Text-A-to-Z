import os
import numpy as np
import joblib

from config import MODEL_DIR, SEQUENCE_LENGTH


# ============================================================
# LOAD CALIBRATED MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sign_language_model_calibrated.pkl"
)

CLASSES_PATH = os.path.join(
    MODEL_DIR,
    "classes_calibrated.npy"
)


print("\nLoading calibrated model...")

model = joblib.load(MODEL_PATH)

classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

print("Calibrated model loaded successfully!")
print("Classes:", classes)


# ============================================================
# CALIBRATION DATA
# ============================================================

base_path = os.path.join(
    "Calibration_Data",
    "A"
)


print("\n" + "=" * 60)
print("TESTING CALIBRATED MODEL ON YOUR A CALIBRATION DATA")
print("=" * 60)


# ============================================================
# FIND CALIBRATION SEQUENCES
# ============================================================

sequence_folders = sorted(
    [
        folder
        for folder in os.listdir(base_path)
        if os.path.isdir(
            os.path.join(
                base_path,
                folder
            )
        )
    ],
    key=int
)


print(
    "\nTotal calibration sequences:",
    len(sequence_folders)
)


# ============================================================
# TEST EACH SEQUENCE
# ============================================================

correct = 0
total = 0


for sequence_number in sequence_folders:

    sequence_path = os.path.join(
        base_path,
        sequence_number
    )

    frames = []


    # --------------------------------------------------------
    # LOAD 30 FRAMES
    # --------------------------------------------------------

    for frame_number in range(
        SEQUENCE_LENGTH
    ):

        frame_path = os.path.join(
            sequence_path,
            f"{frame_number}.npy"
        )

        frame = np.load(
            frame_path
        )

        frames.append(frame)


    sequence = np.array(
        frames,
        dtype=np.float32
    )


    # ========================================================
    # EXTRACT SAME FEATURES USED DURING TRAINING
    # ========================================================

    frame_features = []


    for frame in sequence:

        frame = frame.reshape(
            21,
            3
        )


        # ----------------------------------------------------
        # WRIST RELATIVE COORDINATES
        # ----------------------------------------------------

        wrist = frame[0].copy()

        relative = frame - wrist


        distances_from_wrist = np.linalg.norm(
            relative,
            axis=1
        )


        scale = np.max(
            distances_from_wrist
        )


        if scale > 0:

            relative = (
                relative / scale
            )


        features = (
            relative
            .flatten()
            .tolist()
        )


        # ----------------------------------------------------
        # PAIRWISE DISTANCES
        # ----------------------------------------------------

        important_pairs = [

            (0, 4),
            (0, 8),
            (0, 12),
            (0, 16),
            (0, 20),

            (4, 8),
            (4, 12),
            (4, 16),
            (4, 20),

            (8, 12),
            (8, 16),
            (8, 20),

            (12, 16),
            (12, 20),

            (16, 20)

        ]


        for a, b in important_pairs:

            distance = np.linalg.norm(
                relative[a] -
                relative[b]
            )

            features.append(
                float(distance)
            )


        # ----------------------------------------------------
        # FINGER DIRECTION VECTORS
        # ----------------------------------------------------

        finger_pairs = [

            (0, 4),
            (0, 8),
            (0, 12),
            (0, 16),
            (0, 20)

        ]


        for a, b in finger_pairs:

            vector = (
                relative[b] -
                relative[a]
            )

            features.extend(
                vector.tolist()
            )


        frame_features.append(
            np.array(
                features,
                dtype=np.float32
            )
        )


    frame_features = np.array(
        frame_features,
        dtype=np.float32
    )


    # ========================================================
    # SEQUENCE FEATURES
    # ========================================================

    features = (
        frame_features
        .flatten()
        .tolist()
    )


    # ========================================================
    # MOTION FEATURES
    # ========================================================

    differences = np.diff(
        frame_features,
        axis=0
    )


    mean_motion = np.mean(
        differences,
        axis=0
    )


    max_motion = np.max(
        np.abs(differences),
        axis=0
    )


    features.extend(
        mean_motion.tolist()
    )


    features.extend(
        max_motion.tolist()
    )


    # ========================================================
    # PREPARE MODEL INPUT
    # ========================================================

    input_data = np.array(
        features,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction_index = model.predict(
        input_data
    )[0]


    prediction = classes[
        prediction_index
    ]


    # ========================================================
    # CONFIDENCE
    # ========================================================

    probabilities = model.predict_proba(
        input_data
    )[0]


    confidence = np.max(
        probabilities
    )


    # ========================================================
    # TOP 3 PREDICTIONS
    # ========================================================

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]


    top_predictions = []


    for index in top_indices:

        top_predictions.append(
            (
                classes[index],
                probabilities[index] * 100
            )
        )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print(
        f"\nSequence {sequence_number}:"
    )

    print(
        f"Predicted = {prediction}"
    )

    print(
        f"Confidence = "
        f"{confidence * 100:.1f}%"
    )

    print(
        "Top 3 =",
        [
            (
                letter,
                f"{prob:.1f}%"
            )
            for letter, prob
            in top_predictions
        ]
    )


    # ========================================================
    # CHECK WHETHER A WAS CORRECT
    # ========================================================

    if prediction == "A":

        correct += 1


    total += 1


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)

print(
    f"A recognized correctly: "
    f"{correct}/{total}"
)

print(
    f"Accuracy on your calibration A: "
    f"{(correct / total) * 100:.1f}%"
)

print("=" * 60)
