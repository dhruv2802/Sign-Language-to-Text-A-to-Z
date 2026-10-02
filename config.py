import os


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "MP_Data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sign_language_model.pkl"
)

CLASSES_PATH = os.path.join(
    MODEL_DIR,
    "classes.npy"
)


# ==========================================
# SIGN LANGUAGE SETTINGS
# ==========================================

SEQUENCE_LENGTH = 30

NUM_LANDMARKS = 21

LANDMARK_VALUES = 3

FEATURES_PER_FRAME = (
    NUM_LANDMARKS * LANDMARK_VALUES
)

TOTAL_FEATURES = (
    SEQUENCE_LENGTH * FEATURES_PER_FRAME
)


# ==========================================
# CAMERA SETTINGS
# ==========================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640

FRAME_HEIGHT = 480


# ==========================================
# DISPLAY SETTINGS
# ==========================================

WINDOW_NAME = "Sign Language Recognition"