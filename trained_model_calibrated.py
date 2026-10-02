import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from config import (
    DATASET_PATH,
    MODEL_DIR,
    SEQUENCE_LENGTH
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_frame_features(frame):

    frame = np.asarray(
        frame,
        dtype=np.float32
    ).reshape(21, 3)

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
        relative = relative / scale

    features = relative.flatten().tolist()


    # --------------------------------------------------------
    # Pairwise distances
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Finger direction vectors
    # --------------------------------------------------------

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
# SEQUENCE FEATURE EXTRACTION
# ============================================================

def extract_sequence_features(sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )


    frame_features = []


    for frame in sequence:

        frame_features.append(
            extract_frame_features(frame)
        )


    frame_features = np.array(
        frame_features,
        dtype=np.float32
    )


    features = (
        frame_features
        .flatten()
        .tolist()
    )


    # --------------------------------------------------------
    # Motion features
    # --------------------------------------------------------

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


    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

def load_original_dataset():

    print("\n" + "=" * 60)
    print("LOADING ORIGINAL DATASET")
    print("=" * 60)


    classes = sorted([
        folder
        for folder in os.listdir(
            DATASET_PATH
        )
        if os.path.isdir(
            os.path.join(
                DATASET_PATH,
                folder
            )
        )
    ])


    print("\nClasses:")
    print(classes)


    X = []
    y = []


    for class_index, class_name in enumerate(
        classes
    ):

        class_path = os.path.join(
            DATASET_PATH,
            class_name
        )


        print(
            f"\nProcessing class: {class_name}"
        )


        sequence_folders = sorted(
            os.listdir(
                class_path
            ),
            key=lambda x: int(x)
        )


        for sequence_folder in sequence_folders:

            sequence_path = os.path.join(
                class_path,
                sequence_folder
            )


            if not os.path.isdir(
                sequence_path
            ):
                continue


            frame_files = sorted(
                [
                    f
                    for f in os.listdir(
                        sequence_path
                    )
                    if f.endswith(".npy")
                ],
                key=lambda x: int(
                    os.path.splitext(x)[0]
                )
            )


            if len(frame_files) != SEQUENCE_LENGTH:

                continue


            frames = []

            valid = True


            for frame_file in frame_files:

                frame = np.load(
                    os.path.join(
                        sequence_path,
                        frame_file
                    )
                )


                if frame.shape != (63,):

                    valid = False

                    break


                frames.append(
                    frame
                )


            if valid:

                X.append(
                    np.array(
                        frames,
                        dtype=np.float32
                    )
                )

                y.append(
                    class_index
                )


    return (
        np.array(X, dtype=np.float32),
        np.array(y),
        classes
    )


# ============================================================
# LOAD CALIBRATION A DATA
# ============================================================

def load_calibration_data(
    classes
):

    calibration_path = os.path.join(
        "Calibration_Data",
        "A"
    )


    X_calibration = []
    y_calibration = []


    if not os.path.exists(
        calibration_path
    ):

        print(
            "\nCalibration data not found."
        )

        return (
            np.empty(
                (0, SEQUENCE_LENGTH, 63),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=int
            )
        )


    sequence_folders = sorted(
        [
            folder
            for folder in os.listdir(
                calibration_path
            )
            if os.path.isdir(
                os.path.join(
                    calibration_path,
                    folder
                )
            )
        ],
        key=int
    )


    class_index = classes.index("A")


    print("\n" + "=" * 60)
    print("LOADING CALIBRATION DATA")
    print("=" * 60)


    for sequence_folder in sequence_folders:

        sequence_path = os.path.join(
            calibration_path,
            sequence_folder
        )


        frame_files = sorted(
            [
                f
                for f in os.listdir(
                    sequence_path
                )
                if f.endswith(".npy")
            ],
            key=lambda x: int(
                os.path.splitext(x)[0]
            )
        )


        if len(frame_files) != SEQUENCE_LENGTH:

            print(
                "Skipping sequence:",
                sequence_folder
            )

            continue


        frames = []


        for frame_file in frame_files:

            frame = np.load(
                os.path.join(
                    sequence_path,
                    frame_file
                )
            )


            if frame.shape != (63,):

                break


            frames.append(
                frame
            )


        if len(frames) == SEQUENCE_LENGTH:

            X_calibration.append(
                np.array(
                    frames,
                    dtype=np.float32
                )
            )

            y_calibration.append(
                class_index
            )


    print(
        "Calibration A sequences:",
        len(X_calibration)
    )


    return (
        np.array(
            X_calibration,
            dtype=np.float32
        ),
        np.array(
            y_calibration
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load original dataset
    # --------------------------------------------------------

    X_original, y_original, classes = (
        load_original_dataset()
    )


    print(
        "\nOriginal sequences:",
        len(X_original)
    )


    # --------------------------------------------------------
    # Load calibration data
    # --------------------------------------------------------

    X_calibration, y_calibration = (
        load_calibration_data(
            classes
        )
    )


    # --------------------------------------------------------
    # Combine datasets
    # --------------------------------------------------------

    X = np.concatenate(
        [
            X_original,
            X_calibration
        ],
        axis=0
    )


    y = np.concatenate(
        [
            y_original,
            y_calibration
        ],
        axis=0
    )


    print("\n" + "=" * 60)
    print("COMBINED DATASET")
    print("=" * 60)


    print(
        "Original sequences:",
        len(X_original)
    )

    print(
        "Calibration sequences:",
        len(X_calibration)
    )

    print(
        "Total sequences:",
        len(X)
    )


    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print(
        "\nExtracting enhanced features..."
    )


    feature_data = []


    for sequence in X:

        feature_data.append(
            extract_sequence_features(
                sequence
            )
        )


    X = np.array(
        feature_data,
        dtype=np.float32
    )


    print(
        "Feature shape:",
        X.shape
    )


    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )


    print(
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print(
        "\nTraining calibrated Random Forest..."
    )


    model = RandomForestClassifier(

        n_estimators=500,

        max_features="sqrt",

        random_state=42,

        n_jobs=-1,

        class_weight="balanced"

    )


    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    print(
        "\n" + "=" * 60
    )

    print(
        "CALIBRATED MODEL ACCURACY:",
        accuracy * 100,
        "%"
    )

    print(
        "=" * 60
    )


    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=classes
        )
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    model_path = os.path.join(
        MODEL_DIR,
        "sign_language_model_calibrated.pkl"
    )


    classes_path = os.path.join(
        MODEL_DIR,
        "classes_calibrated.npy"
    )


    joblib.dump(
        model,
        model_path
    )


    np.save(
        classes_path,
        np.array(classes)
    )


    print(
        "\nCalibrated model saved:"
    )

    print(
        model_path
    )


    print(
        "\nCalibrated classes saved:"
    )

    print(
        classes_path
    )


    print(
        "\nTraining completed!"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()