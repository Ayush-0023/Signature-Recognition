import os
import sys

from PIL import Image

from src.logger import logging
from src.exception import CustomException

from src.entity.config_entity import (
    DataValidationConfig
)

from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataValidationArtifacts
)


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifacts,
        data_validation_config: DataValidationConfig
    ):

        self.data_ingestion_artifact = (
            data_ingestion_artifact
        )

        self.data_validation_config = (
            data_validation_config
        )

    def validate_dataset_directory(self) -> bool:
        """
        Checks whether the dataset directory exists.
        """

        try:
            logging.info(
                "Validating dataset directory"
            )

            dataset_path = (
                self.data_ingestion_artifact.dataset_path
            )

            if not os.path.isdir(dataset_path):

                logging.error(
                    f"Dataset directory does not exist: "
                    f"{dataset_path}"
                )

                return False

            logging.info(
                f"Dataset directory found: "
                f"{dataset_path}"
            )

            return True

        except Exception as e:
            raise CustomException(e, sys) from e

    def validate_required_folders(self) -> bool:
        """
        Checks whether Forged and Original
        directories exist.
        """

        try:
            logging.info(
                "Validating required class folders"
            )

            dataset_path = (
                self.data_ingestion_artifact.dataset_path
            )

            required_folders = (
                self.data_validation_config
                .REQUIRED_FOLDERS
            )

            for folder_name in required_folders:

                folder_path = os.path.join(
                    dataset_path,
                    folder_name
                )

                if not os.path.isdir(folder_path):

                    logging.error(
                        f"Required folder missing: "
                        f"{folder_path}"
                    )

                    return False

                logging.info(
                    f"Required folder found: "
                    f"{folder_name}"
                )

            return True

        except Exception as e:
            raise CustomException(e, sys) from e

    def validate_images(self) -> bool:
        """
        Checks whether the class directories
        contain readable image files.
        """

        try:
            logging.info(
                "Validating image files"
            )

            dataset_path = (
                self.data_ingestion_artifact.dataset_path
            )

            required_folders = (
                self.data_validation_config
                .REQUIRED_FOLDERS
            )

            valid_extensions = (
                ".png",
                ".jpg",
                ".jpeg"
            )

            for folder_name in required_folders:

                folder_path = os.path.join(
                    dataset_path,
                    folder_name
                )

                image_files = [
                    file_name
                    for file_name in os.listdir(folder_path)
                    if file_name.lower().endswith(
                        valid_extensions
                    )
                ]

                logging.info(
                    f"Found {len(image_files)} images "
                    f"in class '{folder_name}'"
                )

                if len(image_files) == 0:

                    logging.error(
                        f"No image files found in: "
                        f"{folder_path}"
                    )

                    return False

                for image_name in image_files:

                    image_path = os.path.join(
                        folder_path,
                        image_name
                    )

                    try:

                        with Image.open(image_path) as image:
                            image.verify()

                    except Exception as image_error:

                        logging.error(
                            f"Invalid/corrupted image: "
                            f"{image_path}. "
                            f"Error: {image_error}"
                        )

                        return False

            logging.info(
                "All image files validated successfully"
            )

            return True

        except Exception as e:
            raise CustomException(e, sys) from e

    def write_validation_status(
        self,
        validation_status: bool
    ) -> None:
        """
        Writes validation result to an artifact file.
        """

        try:
            validation_file = (
                self.data_validation_config
                .VALIDATION_STATUS_FILE
            )

            with open(
                validation_file,
                "w"
            ) as file:

                file.write(
                    f"Validation Status: "
                    f"{validation_status}"
                )

            logging.info(
                f"Validation status written to: "
                f"{validation_file}"
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_validation(
        self
    ) -> DataValidationArtifacts:
        """
        Runs complete Data Validation.
        """

        try:
            logging.info(
                "========== DATA VALIDATION STARTED =========="
            )

            os.makedirs(
                self.data_validation_config
                .DATA_VALIDATION_ARTIFACTS_DIR,
                exist_ok=True
            )

            # ----------------------------------
            # Check dataset directory
            # ----------------------------------

            directory_status = (
                self.validate_dataset_directory()
            )

            # ----------------------------------
            # Check class directories
            # ----------------------------------

            folders_status = False

            if directory_status:

                folders_status = (
                    self.validate_required_folders()
                )

            # ----------------------------------
            # Check images
            # ----------------------------------

            images_status = False

            if (
                directory_status
                and folders_status
            ):

                images_status = (
                    self.validate_images()
                )

            # ----------------------------------
            # Final status
            # ----------------------------------

            validation_status = (
                directory_status
                and folders_status
                and images_status
            )

            # Save validation result
            self.write_validation_status(
                validation_status
            )

            data_validation_artifact = (
                DataValidationArtifacts(
                    validation_status=(
                        validation_status
                    ),
                    dataset_path=(
                        self.data_ingestion_artifact
                        .dataset_path
                    )
                )
            )

            logging.info(
                f"Data Validation Artifact: "
                f"{data_validation_artifact}"
            )

            if not validation_status:

                raise Exception(
                    "Dataset validation failed. "
                    "Check logs for details."
                )

            logging.info(
                "========== DATA VALIDATION COMPLETED =========="
            )

            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e