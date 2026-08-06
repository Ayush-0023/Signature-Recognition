import sys

from src.logger import logging
from src.exception import CustomException

from src.configurations.configuration import (
    ConfigurationManager
)

from src.components.data_ingestion import (
    DataIngestion
)

from src.components.data_validation import (
    DataValidation
)

from src.components.data_transformation import (
    DataTransformation
)

from src.components.model_trainer import (
    ModelTrainer
)


class TrainingPipeline:

    def __init__(self):

        self.config = ConfigurationManager()

    def start_data_ingestion(self):

        try:
            logging.info(
                "Starting Data Ingestion stage"
            )

            config = (
                self.config
                .get_data_ingestion_config()
            )

            component = DataIngestion(
                data_ingestion_config=config
            )

            artifact = (
                component
                .initiate_data_ingestion()
            )

            logging.info(
                "Data Ingestion stage completed"
            )

            return artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_validation(
        self,
        data_ingestion_artifact
    ):

        try:
            logging.info(
                "Starting Data Validation stage"
            )

            config = (
                self.config
                .get_data_validation_config()
            )

            component = DataValidation(
                data_ingestion_artifact=(
                    data_ingestion_artifact
                ),
                data_validation_config=config
            )

            artifact = (
                component
                .initiate_data_validation()
            )

            logging.info(
                "Data Validation stage completed"
            )

            return artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_transformation(
        self,
        data_validation_artifact
    ):

        try:
            logging.info(
                "Starting Data Transformation stage"
            )

            config = (
                self.config
                .get_data_transformation_config()
            )

            component = DataTransformation(
                data_validation_artifact=(
                    data_validation_artifact
                ),
                data_transformation_config=config
            )

            artifact = (
                component
                .initiate_data_transformation()
            )

            logging.info(
                "Data Transformation stage completed"
            )

            return artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_trainer(
        self,
        data_transformation_artifact
    ):

        try:
            logging.info(
                "Starting Model Trainer stage"
            )

            config = (
                self.config
                .get_model_trainer_config()
            )

            component = ModelTrainer(
                data_transformation_artifact=(
                    data_transformation_artifact
                ),
                model_trainer_config=config
            )

            artifact = (
                component
                .initiate_model_trainer()
            )

            logging.info(
                "Model Trainer stage completed"
            )

            return artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self):

        try:
            logging.info(
                "Training Pipeline started"
            )

            # ==========================================
            # Stage 1: Data Ingestion
            # ==========================================

            data_ingestion_artifact = (
                self.start_data_ingestion()
            )

            # ==========================================
            # Stage 2: Data Validation
            # ==========================================

            data_validation_artifact = (
                self.start_data_validation(
                    data_ingestion_artifact
                )
            )

            # ==========================================
            # Stage 3: Data Transformation
            # ==========================================

            data_transformation_artifact = (
                self.start_data_transformation(
                    data_validation_artifact
                )
            )

            # ==========================================
            # Stage 4: Model Training
            # ==========================================

            model_trainer_artifact = (
                self.start_model_trainer(
                    data_transformation_artifact
                )
            )

            logging.info(
                f"Best Validation Accuracy: "
                f"{model_trainer_artifact.best_validation_accuracy:.2f}%"
            )

            logging.info(
                f"Trained Model: "
                f"{model_trainer_artifact.trained_model_path}"
            )

            logging.info(
                "Training Pipeline completed"
            )

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":

    training_pipeline = TrainingPipeline()

    training_pipeline.run_pipeline()