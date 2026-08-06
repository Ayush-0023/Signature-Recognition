import os


# ==========================================
# Common Constants
# ==========================================

ARTIFACTS_DIR = "artifacts"

PARAMS_FILE_PATH = "params.yaml"


# ==========================================
# Data Ingestion Constants
# ==========================================

DATA_INGESTION_DIR_NAME = "data_ingestion"

DATA_INGESTION_ARTIFACTS_DIR = os.path.join(
    ARTIFACTS_DIR,
    DATA_INGESTION_DIR_NAME
)

DATASET_URL = (
    "https://drive.google.com/file/d/"
    "11Ycb3rszf6Wd61T7rpahtJrczZ19YK5K/"
    "view?usp=drive_link"
)

ZIP_FILE_NAME = "signatures.zip"

ZIP_FILE_PATH = os.path.join(
    DATA_INGESTION_ARTIFACTS_DIR,
    ZIP_FILE_NAME
)

DATASET_DIR_NAME = "signatures"

DATASET_PATH = os.path.join(
    DATA_INGESTION_ARTIFACTS_DIR,
    DATASET_DIR_NAME
)


# ==========================================
# Data Validation Constants
# ==========================================

DATA_VALIDATION_DIR_NAME = "data_validation"

DATA_VALIDATION_ARTIFACTS_DIR = os.path.join(
    ARTIFACTS_DIR,
    DATA_VALIDATION_DIR_NAME
)

VALIDATION_STATUS_FILE = os.path.join(
    DATA_VALIDATION_ARTIFACTS_DIR,
    "validation_status.txt"
)

REQUIRED_FOLDERS = [
    "Forged",
    "Original"
]


# ==========================================
# Data Transformation Constants
# ==========================================

DATA_TRANSFORMATION_DIR_NAME = "data_transformation"

DATA_TRANSFORMATION_ARTIFACTS_DIR = os.path.join(
    ARTIFACTS_DIR,
    DATA_TRANSFORMATION_DIR_NAME
)

IMAGE_SIZE = 224

BATCH_SIZE = 32

TRAIN_RATIO = 0.60

VALIDATION_RATIO = 0.30

TEST_RATIO = 0.10

RANDOM_SEED = 42

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ==========================================
# Model Trainer Constants
# ==========================================

MODEL_TRAINER_DIR_NAME = "model_trainer"

MODEL_TRAINER_ARTIFACTS_DIR = os.path.join(
    ARTIFACTS_DIR,
    MODEL_TRAINER_DIR_NAME
)

TRAINED_MODEL_NAME = "signature_resnet34.pth"

TRAINED_MODEL_PATH = os.path.join(
    MODEL_TRAINER_ARTIFACTS_DIR,
    TRAINED_MODEL_NAME
)


# ==========================================
# Model Evaluation Constants
# ==========================================

MODEL_EVALUATION_DIR_NAME = "model_evaluation"

MODEL_EVALUATION_ARTIFACTS_DIR = os.path.join(
    ARTIFACTS_DIR,
    MODEL_EVALUATION_DIR_NAME
)

METRICS_FILE_PATH = os.path.join(
    MODEL_EVALUATION_ARTIFACTS_DIR,
    "metrics.json"
)

CONFUSION_MATRIX_FILE_PATH = os.path.join(
    MODEL_EVALUATION_ARTIFACTS_DIR,
    "confusion_matrix.png"
)


# ==========================================
# Prediction Constants
# ==========================================

CLASS_NAMES = [
    "Forged",
    "Original"
]