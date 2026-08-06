import os
import sys
from zipfile import ZipFile

from src.logger import logging
from src.exception import CustomException

from src.configurations.gdrive_syncer import GDriveSync
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:

    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig
    ):
        self.data_ingestion_config = data_ingestion_config
        self.gdrive = GDriveSync()

    def get_data_from_gdrive(self) -> None:
        """
        Downloads signatures.zip from Google Drive.
        """

        try:
            logging.info(
                "Entered get_data_from_gdrive method "
                "of DataIngestion class"
            )

            os.makedirs(
                self.data_ingestion_config
                .DATA_INGESTION_ARTIFACTS_DIR,
                exist_ok=True
            )

            self.gdrive.sync_file_from_gdrive(
                file_url=(
                    self.data_ingestion_config.DATASET_URL
                ),
                destination=(
                    self.data_ingestion_config.ZIP_FILE_PATH
                )
            )

            logging.info(
                "Downloaded signatures dataset successfully"
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def unzip_data(self) -> None:
        """
        Extracts signatures.zip.
        """

        try:
            logging.info(
                "Entered unzip_data method "
                "of DataIngestion class"
            )

            with ZipFile(
                self.data_ingestion_config.ZIP_FILE_PATH,
                "r"
            ) as zip_ref:

                zip_ref.extractall(
                    self.data_ingestion_config
                    .DATA_INGESTION_ARTIFACTS_DIR
                )

            logging.info(
                "Extracted signatures dataset successfully"
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def remove_zip_file(self) -> None:
        """
        Deletes signatures.zip after extraction.
        """

        try:
            logging.info(
                "Entered remove_zip_file method "
                "of DataIngestion class"
            )

            zip_file_path = (
                self.data_ingestion_config.ZIP_FILE_PATH
            )

            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)

                logging.info(
                    f"Removed ZIP file: {zip_file_path}"
                )

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Runs the complete Data Ingestion pipeline.

        If the extracted dataset already exists,
        downloading and extraction are skipped.

        Steps:
        1. Check whether dataset already exists
        2. Download signatures.zip if necessary
        3. Extract signatures.zip
        4. Delete signatures.zip
        5. Return DataIngestionArtifacts
        """

        try:
            logging.info(
                "Started Data Ingestion"
            )

            # Check whether the extracted dataset already exists
            if os.path.exists(
                    self.data_ingestion_config.DATASET_PATH
            ):
                logging.info(
                    f"Dataset already exists at: "
                    f"{self.data_ingestion_config.DATASET_PATH}"
                )

                logging.info(
                    "Skipping dataset download and extraction"
                )

            else:
                logging.info(
                    "Dataset not found locally. "
                    "Starting download."
                )

                # Step 1: Download
                self.get_data_from_gdrive()

                # Step 2: Extract
                self.unzip_data()

                # Step 3: Delete ZIP
                self.remove_zip_file()

            # Create Data Ingestion Artifact
            data_ingestion_artifact = DataIngestionArtifacts(
                dataset_path=(
                    self.data_ingestion_config.DATASET_PATH
                )
            )

            logging.info(
                "Data Ingestion completed successfully"
            )

            logging.info(
                f"Data Ingestion Artifact: "
                f"{data_ingestion_artifact}"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys) from e