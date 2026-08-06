from dataclasses import dataclass
from typing import List


# ==========================================
# Data Ingestion Configuration
# ==========================================

@dataclass
class DataIngestionConfig:
    DATASET_URL: str
    DATA_INGESTION_ARTIFACTS_DIR: str
    ZIP_FILE_NAME: str
    ZIP_FILE_PATH: str
    DATASET_PATH: str


# ==========================================
# Data Validation Configuration
# ==========================================

@dataclass
class DataValidationConfig:
    DATA_VALIDATION_ARTIFACTS_DIR: str
    VALIDATION_STATUS_FILE: str
    REQUIRED_FOLDERS: List[str]


# ==========================================
# Data Transformation Configuration
# ==========================================

@dataclass
class DataTransformationConfig:
    DATA_TRANSFORMATION_ARTIFACTS_DIR: str
    IMAGE_SIZE: int
    BATCH_SIZE: int
    TRAIN_RATIO: float
    VALIDATION_RATIO: float
    TEST_RATIO: float
    RANDOM_SEED: int
    IMAGENET_MEAN: List[float]
    IMAGENET_STD: List[float]


# ==========================================
# Model Trainer Configuration
# ==========================================

@dataclass
class ModelTrainerConfig:
    MODEL_TRAINER_ARTIFACTS_DIR: str
    TRAINED_MODEL_PATH: str
    NUM_CLASSES: int
    EPOCHS: int
    LEARNING_RATE: float
    MOMENTUM: float


# ==========================================
# Model Evaluation Configuration
# ==========================================

@dataclass
class ModelEvaluationConfig:
    MODEL_EVALUATION_ARTIFACTS_DIR: str
    TRAINED_MODEL_PATH: str
    METRICS_FILE_PATH: str
    CONFUSION_MATRIX_FILE_PATH: str
    NUM_CLASSES: int