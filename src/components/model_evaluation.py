import os
import sys
import json

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torchvision.models import (
    resnet34
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from src.logger import logging
from src.exception import CustomException

from src.entity.config_entity import (
    ModelEvaluationConfig
)

from src.entity.artifact_entity import (
    DataTransformationArtifacts,
    ModelEvaluationArtifacts
)


class ModelEvaluation:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifacts,
        model_evaluation_config: ModelEvaluationConfig
    ):

        self.data_transformation_artifact = (
            data_transformation_artifact
        )

        self.model_evaluation_config = (
            model_evaluation_config
        )

        self.device = self.get_device()

    def get_device(self):
        """
        Selects available computation device.
        """

        if torch.cuda.is_available():

            device = torch.device("cuda")

        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):

            device = torch.device("mps")

        else:

            device = torch.device("cpu")

        logging.info(
            f"Evaluation device: {device}"
        )

        return device

    def load_model(self):
        """
        Reconstructs ResNet-34 architecture and loads
        the saved best model weights.
        """

        try:
            logging.info(
                "Loading trained ResNet-34 model"
            )

            model_path = (
                self.model_evaluation_config
                .TRAINED_MODEL_PATH
            )

            if not os.path.exists(model_path):

                raise FileNotFoundError(
                    f"Trained model not found at: "
                    f"{model_path}"
                )

            # We do NOT need to download ImageNet
            # weights here because we're about to load
            # our own trained state_dict.
            model = resnet34(
                weights=None
            )

            number_of_features = (
                model.fc.in_features
            )

            model.fc = nn.Linear(
                number_of_features,
                self.model_evaluation_config
                .NUM_CLASSES
            )

            state_dict = torch.load(
                model_path,
                map_location=self.device
            )

            model.load_state_dict(
                state_dict
            )

            model = model.to(
                self.device
            )

            model.eval()

            logging.info(
                "Trained model loaded successfully"
            )

            return model

        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate_model(
        self,
        model
    ) -> ModelEvaluationArtifacts:
        """
        Evaluates the saved best model on the
        untouched test dataset.
        """

        try:
            logging.info(
                "========== MODEL EVALUATION STARTED =========="
            )

            test_loader = (
                self.data_transformation_artifact
                .test_loader
            )

            class_names = (
                self.data_transformation_artifact
                .class_names
            )

            y_true = []
            y_pred = []

            with torch.no_grad():

                for images, labels in test_loader:

                    images = images.to(
                        self.device
                    )

                    labels = labels.to(
                        self.device
                    )

                    outputs = model(
                        images
                    )

                    _, predictions = torch.max(
                        outputs,
                        dim=1
                    )

                    y_true.extend(
                        labels.cpu().tolist()
                    )

                    y_pred.extend(
                        predictions.cpu().tolist()
                    )

            # ==========================================
            # Metrics
            # ==========================================

            accuracy = accuracy_score(
                y_true,
                y_pred
            )

            precision = precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )

            logging.info(
                f"Test Accuracy: "
                f"{accuracy * 100:.2f}%"
            )

            logging.info(
                f"Test Precision: "
                f"{precision:.4f}"
            )

            logging.info(
                f"Test Recall: "
                f"{recall:.4f}"
            )

            logging.info(
                f"Test F1 Score: "
                f"{f1:.4f}"
            )

            report = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                zero_division=0
            )

            logging.info(
                "\nClassification Report:\n"
                f"{report}"
            )

            # ==========================================
            # Save metrics
            # ==========================================

            os.makedirs(
                self.model_evaluation_config
                .MODEL_EVALUATION_ARTIFACTS_DIR,
                exist_ok=True
            )

            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }

            with open(
                self.model_evaluation_config
                .METRICS_FILE_PATH,
                "w"
            ) as file:

                json.dump(
                    metrics,
                    file,
                    indent=4
                )

            logging.info(
                "Evaluation metrics saved at: "
                f"{self.model_evaluation_config.METRICS_FILE_PATH}"
            )

            # ==========================================
            # Confusion Matrix
            # ==========================================

            matrix = confusion_matrix(
                y_true,
                y_pred
            )

            display = ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=class_names
            )

            display.plot(
                values_format="d"
            )

            plt.title(
                "Signature Recognition - Confusion Matrix"
            )

            plt.tight_layout()

            plt.savefig(
                self.model_evaluation_config
                .CONFUSION_MATRIX_FILE_PATH
            )

            plt.close()

            logging.info(
                "Confusion matrix saved at: "
                f"{self.model_evaluation_config.CONFUSION_MATRIX_FILE_PATH}"
            )

            # ==========================================
            # Artifact
            # ==========================================

            model_evaluation_artifact = (
                ModelEvaluationArtifacts(

                    accuracy=float(
                        accuracy
                    ),

                    precision=float(
                        precision
                    ),

                    recall=float(
                        recall
                    ),

                    f1_score=float(
                        f1
                    ),

                    metrics_file_path=(
                        self.model_evaluation_config
                        .METRICS_FILE_PATH
                    ),

                    confusion_matrix_file_path=(
                        self.model_evaluation_config
                        .CONFUSION_MATRIX_FILE_PATH
                    )
                )
            )

            logging.info(
                "========== MODEL EVALUATION COMPLETED =========="
            )

            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(
        self
    ) -> ModelEvaluationArtifacts:
        """
        Runs complete model evaluation.
        """

        try:

            model = self.load_model()

            artifact = self.evaluate_model(
                model
            )

            return artifact

        except Exception as e:
            raise CustomException(e, sys) from e