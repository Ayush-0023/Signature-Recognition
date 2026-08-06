import os
import sys

import torch

from torch.utils.data import (
    DataLoader,
    Subset
)

from torchvision import (
    datasets,
    transforms
)

from src.logger import logging
from src.exception import CustomException

from src.entity.config_entity import (
    DataTransformationConfig
)

from src.entity.artifact_entity import (
    DataValidationArtifacts,
    DataTransformationArtifacts
)


class DataTransformation:

    def __init__(
        self,
        data_validation_artifact: DataValidationArtifacts,
        data_transformation_config: DataTransformationConfig
    ):

        self.data_validation_artifact = (
            data_validation_artifact
        )

        self.data_transformation_config = (
            data_transformation_config
        )

    def get_train_transform(self):
        """
        Transformations used only for training data.

        Training contains augmentation to help
        reduce overfitting.
        """

        try:
            logging.info(
                "Creating training transformations"
            )

            train_transform = transforms.Compose([

                transforms.Resize(
                    (
                        self.data_transformation_config.IMAGE_SIZE,
                        self.data_transformation_config.IMAGE_SIZE
                    )
                ),

                transforms.RandomRotation(
                    degrees=(-20, 20)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=(
                        self.data_transformation_config
                        .IMAGENET_MEAN
                    ),
                    std=(
                        self.data_transformation_config
                        .IMAGENET_STD
                    )
                )
            ])

            return train_transform

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_evaluation_transform(self):
        """
        Transformations for validation and test data.

        No random augmentation is applied here.
        """

        try:
            logging.info(
                "Creating validation/test transformations"
            )

            evaluation_transform = transforms.Compose([

                transforms.Resize(
                    (
                        self.data_transformation_config.IMAGE_SIZE,
                        self.data_transformation_config.IMAGE_SIZE
                    )
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=(
                        self.data_transformation_config
                        .IMAGENET_MEAN
                    ),
                    std=(
                        self.data_transformation_config
                        .IMAGENET_STD
                    )
                )
            ])

            return evaluation_transform

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(
        self
    ) -> DataTransformationArtifacts:
        """
        Creates:

        1. Training transformations
        2. Validation/test transformations
        3. Reproducible train/validation/test split
        4. PyTorch DataLoaders
        """

        try:
            logging.info(
                "========== DATA TRANSFORMATION STARTED =========="
            )

            if not (
                self.data_validation_artifact
                .validation_status
            ):
                raise Exception(
                    "Data Transformation cannot start "
                    "because Data Validation failed."
                )

            os.makedirs(
                self.data_transformation_config
                .DATA_TRANSFORMATION_ARTIFACTS_DIR,
                exist_ok=True
            )

            dataset_path = (
                self.data_validation_artifact
                .dataset_path
            )

            # ==========================================
            # Create transforms
            # ==========================================

            train_transform = (
                self.get_train_transform()
            )

            evaluation_transform = (
                self.get_evaluation_transform()
            )

            # ==========================================
            # Base dataset
            #
            # No transform is needed here.
            # We use it to obtain labels/classes and
            # determine the indices for each split.
            # ==========================================

            base_dataset = datasets.ImageFolder(
                root=dataset_path
            )

            class_names = base_dataset.classes

            class_to_idx = base_dataset.class_to_idx

            total_size = len(base_dataset)

            logging.info(
                f"Total dataset size: {total_size}"
            )

            logging.info(
                f"Classes: {class_names}"
            )

            logging.info(
                f"Class mapping: {class_to_idx}"
            )

            # ==========================================
            # Calculate split sizes
            # ==========================================

            train_size = int(
                self.data_transformation_config
                .TRAIN_RATIO
                * total_size
            )

            validation_size = int(
                self.data_transformation_config
                .VALIDATION_RATIO
                * total_size
            )

            test_size = (
                total_size
                - train_size
                - validation_size
            )

            logging.info(
                f"Train size: {train_size}"
            )

            logging.info(
                f"Validation size: {validation_size}"
            )

            logging.info(
                f"Test size: {test_size}"
            )

            # ==========================================
            # Generate reproducible shuffled indices
            # ==========================================

            generator = torch.Generator()

            generator.manual_seed(
                self.data_transformation_config
                .RANDOM_SEED
            )

            indices = torch.randperm(
                total_size,
                generator=generator
            ).tolist()

            train_end = train_size

            validation_end = (
                train_size
                + validation_size
            )

            train_indices = (
                indices[:train_end]
            )

            validation_indices = (
                indices[
                    train_end:validation_end
                ]
            )

            test_indices = (
                indices[validation_end:]
            )

            # ==========================================
            # Separate dataset objects are important.
            #
            # Training gets augmentation.
            # Validation/test do not.
            # ==========================================

            train_dataset_full = datasets.ImageFolder(
                root=dataset_path,
                transform=train_transform
            )

            evaluation_dataset_full = datasets.ImageFolder(
                root=dataset_path,
                transform=evaluation_transform
            )

            train_dataset = Subset(
                train_dataset_full,
                train_indices
            )

            validation_dataset = Subset(
                evaluation_dataset_full,
                validation_indices
            )

            test_dataset = Subset(
                evaluation_dataset_full,
                test_indices
            )

            # ==========================================
            # Create DataLoaders
            # ==========================================

            train_loader = DataLoader(
                dataset=train_dataset,

                batch_size=(
                    self.data_transformation_config
                    .BATCH_SIZE
                ),

                shuffle=True
            )

            validation_loader = DataLoader(
                dataset=validation_dataset,

                batch_size=(
                    self.data_transformation_config
                    .BATCH_SIZE
                ),

                shuffle=False
            )

            test_loader = DataLoader(
                dataset=test_dataset,

                batch_size=(
                    self.data_transformation_config
                    .BATCH_SIZE
                ),

                shuffle=False
            )

            # ==========================================
            # Create artifact
            # ==========================================

            data_transformation_artifact = (
                DataTransformationArtifacts(

                    train_loader=train_loader,

                    validation_loader=(
                        validation_loader
                    ),

                    test_loader=test_loader,

                    class_names=class_names,

                    class_to_idx=class_to_idx,

                    train_size=len(
                        train_dataset
                    ),

                    validation_size=len(
                        validation_dataset
                    ),

                    test_size=len(
                        test_dataset
                    )
                )
            )

            logging.info(
                "DataLoaders created successfully"
            )

            logging.info(
                f"Train samples: "
                f"{len(train_dataset)}"
            )

            logging.info(
                f"Validation samples: "
                f"{len(validation_dataset)}"
            )

            logging.info(
                f"Test samples: "
                f"{len(test_dataset)}"
            )

            logging.info(
                "========== DATA TRANSFORMATION COMPLETED =========="
            )

            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e