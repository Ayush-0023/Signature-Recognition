from dataclasses import dataclass
from typing import List, Dict, Any


# ==========================================
# Data Ingestion Artifact
# ==========================================

@dataclass
class DataIngestionArtifacts:
    dataset_path: str


# ==========================================
# Data Validation Artifact
# ==========================================

@dataclass
class DataValidationArtifacts:
    validation_status: bool
    dataset_path: str


# ==========================================
# Data Transformation Artifact
# ==========================================

@dataclass
class DataTransformationArtifacts:
    train_loader: Any
    validation_loader: Any
    test_loader: Any

    class_names: List[str]
    class_to_idx: Dict[str, int]

    train_size: int
    validation_size: int
    test_size: int


# ==========================================
# Model Trainer Artifact
# ==========================================

@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str
    best_validation_accuracy: float
    training_history: Dict[str, List[float]]


# ==========================================
# Model Evaluation Artifact
# ==========================================

@dataclass
class ModelEvaluationArtifacts:
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    metrics_file_path: str
    confusion_matrix_file_path: str