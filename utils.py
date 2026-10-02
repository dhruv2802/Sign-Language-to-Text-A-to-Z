import os
import numpy as np
import joblib

from config import (
    MODEL_DIR,
    SEQUENCE_LENGTH
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sign_language_model_final.pkl"
)

CLASSES_PATH = os.path.join(
    MODEL_DIR,
    "classes_final.npy"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading final model...")

    model = joblib.load(
        MODEL_PATH
    )

    classes = np.load(
        CLASSES_PATH,
        allow_pickle=True
    )

    print("Final model loaded successfully!")
    print("Classes:", classes)

    return model, classes


# ============================================================
# LANDMARK ARRAY
# ============================================================

def landmarks_to_array(landmarks):

    if landmarks is None:
        return None

    return np.array(
        landmarks,
        dtype=np.float32
    )


# ============================================================
# FRAME FEATURES
# ============================================================

def extract_frame_features(frame):

    frame = np.asarray(
        frame,
        dtype=np.float32
    ).reshape(21, 3)

    # Wrist normalization
    wrist = frame[0].copy()

    relative = frame - wrist

    distances = np.linalg.norm(
        relative,
        axis=1
    )

    scale = np.max(distances)

    if scale > 0:
        relative = relative / scale

    features = relative.flatten().tolist()

    # Landmark distances
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
            relative[a] - relative[b]
        )

        features.append(
            float(distance)
        )

    # Finger direction vectors
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

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# SEQUENCE FEATURES
# ============================================================

def prepare_sequence(sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        63
    )

    if sequence.shape != expected_shape:

        raise ValueError(
            f"Expected sequence shape "
            f"{expected_shape}, "
            f"but got {sequence.shape}"
        )

    frame_features = []

    for frame in sequence:

        frame_features.append(
            extract_frame_features(
                frame
            )
        )

    frame_features = np.array(
        frame_features,
        dtype=np.float32
    )

    # Position features
    features = (
        frame_features
        .flatten()
        .tolist()
    )

    # Motion features
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

    features = np.array(
        features,
        dtype=np.float32
    )

    return features.reshape(
        1,
        -1
    )


# ============================================================
# PREDICT
# ============================================================

def predict_sign(
    model,
    classes,
    sequence
):

    input_data = prepare_sequence(
        sequence
    )

    prediction = model.predict(
        input_data
    )[0]

    return classes[prediction]


# ============================================================
# CONFIDENCE
# ============================================================

def get_prediction_probability(
    model,
    sequence
):

    input_data = prepare_sequence(
        sequence
    )

    probabilities = (
        model.predict_proba(
            input_data
        )[0]
    )

    return float(
        np.max(probabilities)
    )


# ============================================================
# TOP PREDICTIONS
# ============================================================

def get_top_predictions(
    model,
    classes,
    sequence,
    top_n=3
):

    input_data = prepare_sequence(
        sequence
    )

    probabilities = (
        model.predict_proba(
            input_data
        )[0]
    )

    indices = np.argsort(
        probabilities
    )[::-1][:top_n]

    results = []

    for index in indices:

        results.append(
            (
                classes[index],
                float(
                    probabilities[index]
                )
            )
        )

    return results