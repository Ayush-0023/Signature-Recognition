import sys

from src.logger import logging
from src.exception import CustomException

from src.configurations.configuration import (
    ConfigurationManager
)

from src.components.data_validation import (
    DataValidation
)

from src.components.data_transformation import (
    DataTransformation
)

from src.components.model_evaluation import (
    ModelEvaluation
)

from src.entity.artifact_entity import (
    DataIngestionArtifacts
)

from src.constants import (
    DATASET_PATH
)


class EvaluationPipeline:

    def __init__(self):

        self.config = ConfigurationManager()

    def run_pipeline(self):
        """
        Runs evaluation using the already-trained
        best model.

        Training is NOT executed again.
        """

        try:
            logging.info(
                "Evaluation Pipeline started"
            )

            # ==========================================
            # Existing dataset artifact
            # ==========================================

            data_ingestion_artifact = (
                DataIngestionArtifacts(
                    dataset_path=DATASET_PATH
                )
            )

            # ==========================================
            # Data Validation
            # ==========================================

            logging.info(
                "Starting Data Validation stage"
            )

            validation_config = (
                self.config
                .get_data_validation_config()
            )

            validation_component = (
                DataValidation(
                    data_ingestion_artifact=(
                        data_ingestion_artifact
                    ),
                    data_validation_config=(
                        validation_config
                    )
                )
            )

            data_validation_artifact = (
                validation_component
                .initiate_data_validation()
            )

            # ==========================================
            # Data Transformation
            # ==========================================

            logging.info(
                "Starting Data Transformation stage"
            )

            transformation_config = (
                self.config
                .get_data_transformation_config()
            )

            transformation_component = (
                DataTransformation(
                    data_validation_artifact=(
                        data_validation_artifact
                    ),
                    data_transformation_config=(
                        transformation_config
                    )
                )
            )

            data_transformation_artifact = (
                transformation_component
                .initiate_data_transformation()
            )

            # ==========================================
            # Model Evaluation
            # ==========================================

            logging.info(
                "Starting Model Evaluation stage"
            )

            evaluation_config = (
                self.config
                .get_model_evaluation_config()
            )

            evaluation_component = (
                ModelEvaluation(
                    data_transformation_artifact=(
                        data_transformation_artifact
                    ),
                    model_evaluation_config=(
                        evaluation_config
                    )
                )
            )

            evaluation_artifact = (
                evaluation_component
                .initiate_model_evaluation()
            )

            logging.info(
                f"Final Test Accuracy: "
                f"{evaluation_artifact.accuracy * 100:.2f}%"
            )

            logging.info(
                f"Final Test Precision: "
                f"{evaluation_artifact.precision:.4f}"
            )

            logging.info(
                f"Final Test Recall: "
                f"{evaluation_artifact.recall:.4f}"
            )

            logging.info(
                f"Final Test F1 Score: "
                f"{evaluation_artifact.f1_score:.4f}"
            )

            logging.info(
                "Evaluation Pipeline completed"
            )

            return evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":

    evaluation_pipeline = (
        EvaluationPipeline()
    )

    evaluation_pipeline.run_pipeline()