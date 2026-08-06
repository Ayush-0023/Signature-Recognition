import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import (
    resnet34,
    ResNet34_Weights
)

from src.logger import logging
from src.exception import CustomException

from src.entity.config_entity import (
    ModelTrainerConfig
)

from src.entity.artifact_entity import (
    DataTransformationArtifacts,
    ModelTrainerArtifacts
)


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifacts,
        model_trainer_config: ModelTrainerConfig
    ):

        self.data_transformation_artifact = (
            data_transformation_artifact
        )

        self.model_trainer_config = (
            model_trainer_config
        )

        self.device = self.get_device()

    def get_device(self):
        """
        Selects the best available device.

        Priority:
        1. CUDA
        2. Apple MPS
        3. CPU
        """

        if torch.cuda.is_available():

            device = torch.device("cuda")

        elif torch.backends.mps.is_available():

            device = torch.device("mps")

        else:

            device = torch.device("cpu")

        logging.info(
            f"Using device: {device}"
        )

        return device

    def build_model(self):
        """
        Loads pretrained ResNet-34 and replaces
        the final classification layer.
        """

        try:
            logging.info(
                "Loading pretrained ResNet-34"
            )

            weights = (
                ResNet34_Weights.DEFAULT
            )

            model = resnet34(
                weights=weights
            )

            number_of_features = (
                model.fc.in_features
            )

            model.fc = nn.Linear(
                number_of_features,
                self.model_trainer_config.NUM_CLASSES
            )

            model = model.to(
                self.device
            )

            logging.info(
                "ResNet-34 model created successfully"
            )

            logging.info(
                f"Final output classes: "
                f"{self.model_trainer_config.NUM_CLASSES}"
            )

            return model

        except Exception as e:
            raise CustomException(e, sys) from e

    def train_model(
        self,
        model
    ) -> ModelTrainerArtifacts:
        """
        Trains ResNet-34 and saves the model
        with the highest validation accuracy.
        """

        try:
            logging.info(
                "========== MODEL TRAINING STARTED =========="
            )

            train_loader = (
                self.data_transformation_artifact
                .train_loader
            )

            validation_loader = (
                self.data_transformation_artifact
                .validation_loader
            )

            criterion = (
                nn.CrossEntropyLoss()
            )

            optimizer = optim.SGD(
                model.parameters(),

                lr=(
                    self.model_trainer_config
                    .LEARNING_RATE
                ),

                momentum=(
                    self.model_trainer_config
                    .MOMENTUM
                )
            )

            os.makedirs(
                self.model_trainer_config
                .MODEL_TRAINER_ARTIFACTS_DIR,
                exist_ok=True
            )

            training_history = {
                "train_loss": [],
                "train_accuracy": [],
                "validation_loss": [],
                "validation_accuracy": []
            }

            best_validation_accuracy = 0.0

            # ==========================================
            # Epoch loop
            # ==========================================

            for epoch in range(
                self.model_trainer_config.EPOCHS
            ):

                logging.info(
                    f"Epoch "
                    f"[{epoch + 1}/"
                    f"{self.model_trainer_config.EPOCHS}]"
                )

                # ======================================
                # TRAINING
                # ======================================

                model.train()

                running_train_loss = 0.0

                correct_train_predictions = 0

                total_train_samples = 0

                for images, labels in train_loader:

                    images = images.to(
                        self.device
                    )

                    labels = labels.to(
                        self.device
                    )

                    # Clear old gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(images)

                    # Calculate loss
                    loss = criterion(
                        outputs,
                        labels
                    )

                    # Backpropagation
                    loss.backward()

                    # Update model parameters
                    optimizer.step()

                    # ----------------------------------
                    # Statistics
                    # ----------------------------------

                    batch_size = (
                        images.size(0)
                    )

                    running_train_loss += (
                        loss.item()
                        * batch_size
                    )

                    _, predictions = (
                        torch.max(
                            outputs,
                            1
                        )
                    )

                    correct_train_predictions += (
                        predictions
                        .eq(labels)
                        .sum()
                        .item()
                    )

                    total_train_samples += (
                        batch_size
                    )

                train_loss = (
                    running_train_loss
                    / total_train_samples
                )

                train_accuracy = (
                    100.0
                    * correct_train_predictions
                    / total_train_samples
                )

                # ======================================
                # VALIDATION
                # ======================================

                model.eval()

                running_validation_loss = 0.0

                correct_validation_predictions = 0

                total_validation_samples = 0

                with torch.no_grad():

                    for (
                        images,
                        labels
                    ) in validation_loader:

                        images = images.to(
                            self.device
                        )

                        labels = labels.to(
                            self.device
                        )

                        outputs = model(
                            images
                        )

                        loss = criterion(
                            outputs,
                            labels
                        )

                        batch_size = (
                            images.size(0)
                        )

                        running_validation_loss += (
                            loss.item()
                            * batch_size
                        )

                        _, predictions = (
                            torch.max(
                                outputs,
                                1
                            )
                        )

                        correct_validation_predictions += (
                            predictions
                            .eq(labels)
                            .sum()
                            .item()
                        )

                        total_validation_samples += (
                            batch_size
                        )

                validation_loss = (
                    running_validation_loss
                    / total_validation_samples
                )

                validation_accuracy = (
                    100.0
                    * correct_validation_predictions
                    / total_validation_samples
                )

                # ======================================
                # Save history
                # ======================================

                training_history[
                    "train_loss"
                ].append(
                    train_loss
                )

                training_history[
                    "train_accuracy"
                ].append(
                    train_accuracy
                )

                training_history[
                    "validation_loss"
                ].append(
                    validation_loss
                )

                training_history[
                    "validation_accuracy"
                ].append(
                    validation_accuracy
                )

                logging.info(
                    f"Train Loss: "
                    f"{train_loss:.4f} | "
                    f"Train Accuracy: "
                    f"{train_accuracy:.2f}%"
                )

                logging.info(
                    f"Validation Loss: "
                    f"{validation_loss:.4f} | "
                    f"Validation Accuracy: "
                    f"{validation_accuracy:.2f}%"
                )

                # ======================================
                # Save best model
                # ======================================

                if (
                    validation_accuracy
                    > best_validation_accuracy
                ):

                    best_validation_accuracy = (
                        validation_accuracy
                    )

                    torch.save(
                        model.state_dict(),
                        self.model_trainer_config
                        .TRAINED_MODEL_PATH
                    )

                    logging.info(
                        "Best model updated and saved "
                        f"with validation accuracy: "
                        f"{best_validation_accuracy:.2f}%"
                    )

            logging.info(
                "========== MODEL TRAINING COMPLETED =========="
            )

            model_trainer_artifact = (
                ModelTrainerArtifacts(

                    trained_model_path=(
                        self.model_trainer_config
                        .TRAINED_MODEL_PATH
                    ),

                    best_validation_accuracy=(
                        best_validation_accuracy
                    ),

                    training_history=(
                        training_history
                    )
                )
            )

            logging.info(
                f"Model Trainer Artifact: "
                f"{model_trainer_artifact}"
            )

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(
        self
    ) -> ModelTrainerArtifacts:
        """
        Initiates complete model training.
        """

        try:
            model = self.build_model()

            model_trainer_artifact = (
                self.train_model(
                    model
                )
            )

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys) from e