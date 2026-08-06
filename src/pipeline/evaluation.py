import sys

from src.logger import logging
from src.exception import CustomException

from src.configurations.configuration import (
    ConfigurationManager
)

from src.entity.artifact_entity import (
    DataIngestionArtifacts
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

from src.components.experiment_tracker import (
    ExperimentTracker
)


class EvaluationPipeline:

    def __init__(self):

        try:

            logging.info(
                "Evaluation Pipeline started"
            )

            self.config_manager = (
                ConfigurationManager()
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e


    def run_pipeline(self):

        try:

            # ==========================================
            # 1. Create Data Ingestion Artifact
            # ==========================================
            #
            # Evaluation does NOT need to download
            # the dataset again.
            #
            # We simply tell the evaluation pipeline
            # where the already-ingested dataset exists.
            # ==========================================

            data_ingestion_config = (
                self.config_manager
                .get_data_ingestion_config()
            )

            data_ingestion_artifact = (
                DataIngestionArtifacts(
                    dataset_path=(
                        data_ingestion_config
                        .DATASET_PATH
                    )
                )
            )


            # ==========================================
            # 2. Data Validation
            # ==========================================

            logging.info(
                "Starting Data Validation stage"
            )

            data_validation_config = (
                self.config_manager
                .get_data_validation_config()
            )

            data_validation = (
                DataValidation(
                    data_ingestion_artifact=(
                        data_ingestion_artifact
                    ),
                    data_validation_config=(
                        data_validation_config
                    )
                )
            )

            data_validation_artifact = (
                data_validation
                .initiate_data_validation()
            )


            # ==========================================
            # Check Validation Status
            # ==========================================

            if not (
                data_validation_artifact
                .validation_status
            ):

                raise Exception(
                    "Data validation failed. "
                    "Evaluation cannot continue."
                )


            # ==========================================
            # 3. Data Transformation
            # ==========================================

            logging.info(
                "Starting Data Transformation stage"
            )

            data_transformation_config = (
                self.config_manager
                .get_data_transformation_config()
            )

            data_transformation = (
                DataTransformation(
                    data_validation_artifact=(
                        data_validation_artifact
                    ),
                    data_transformation_config=(
                        data_transformation_config
                    )
                )
            )

            data_transformation_artifact = (
                data_transformation
                .initiate_data_transformation()
            )


            # ==========================================
            # 4. Model Evaluation
            # ==========================================

            logging.info(
                "Starting Model Evaluation stage"
            )

            model_evaluation_config = (
                self.config_manager
                .get_model_evaluation_config()
            )

            model_evaluation = (
                ModelEvaluation(
                    data_transformation_artifact=(
                        data_transformation_artifact
                    ),
                    model_evaluation_config=(
                        model_evaluation_config
                    )
                )
            )

            model_evaluation_artifact = (
                model_evaluation
                .initiate_model_evaluation()
            )


            # ==========================================
            # 5. Display Final Evaluation Results
            # ==========================================

            logging.info(
                f"Final Test Accuracy: "
                f"{model_evaluation_artifact.accuracy:.2f}%"
            )

            logging.info(
                f"Final Test Precision: "
                f"{model_evaluation_artifact.precision:.4f}"
            )

            logging.info(
                f"Final Test Recall: "
                f"{model_evaluation_artifact.recall:.4f}"
            )

            logging.info(
                f"Final Test F1 Score: "
                f"{model_evaluation_artifact.f1_score:.4f}"
            )


            # ==========================================
            # 6. Get Training Parameters
            # ==========================================

            model_trainer_config = (
                self.config_manager
                .get_model_trainer_config()
            )


            # ==========================================
            # 7. Prepare MLflow Parameters
            # ==========================================

            mlflow_params = {

                "model_architecture": (
                    "ResNet-34"
                ),

                "num_classes": (
                    model_trainer_config
                    .NUM_CLASSES
                ),

                "epochs": (
                    model_trainer_config
                    .EPOCHS
                ),

                "learning_rate": (
                    model_trainer_config
                    .LEARNING_RATE
                ),

                "momentum": (
                    model_trainer_config
                    .MOMENTUM
                )
            }


            # ==========================================
            # 8. Prepare MLflow Metrics
            # ==========================================

            mlflow_metrics = {

                "test_accuracy": (
                    model_evaluation_artifact
                    .accuracy / 100.0
                ),

                "test_precision": (
                    model_evaluation_artifact
                    .precision
                ),

                "test_recall": (
                    model_evaluation_artifact
                    .recall
                ),

                "test_f1_score": (
                    model_evaluation_artifact
                    .f1_score
                )
            }


            # ==========================================
            # 9. Initialize MLflow Experiment Tracker
            # ==========================================

            logging.info(
                "Starting MLflow Experiment Tracking"
            )

            experiment_tracker = (
                ExperimentTracker(
                    experiment_name=(
                        "Signature-Recognition-ResNet34"
                    )
                )
            )


            # ==========================================
            # 10. Create MLflow Run Name
            # ==========================================

            run_name = (
                f"resnet34_"
                f"epochs_{model_trainer_config.EPOCHS}_"
                f"lr_{model_trainer_config.LEARNING_RATE}"
            )


            # ==========================================
            # 11. Log Experiment
            # ==========================================

            run_id = (
                experiment_tracker
                .log_experiment(

                    params=mlflow_params,

                    metrics=mlflow_metrics,

                    model_path=(
                        model_trainer_config
                        .TRAINED_MODEL_PATH
                    ),

                    confusion_matrix_path=(
                        model_evaluation_artifact
                        .confusion_matrix_file_path
                    ),

                    run_name=run_name
                )
            )


            logging.info(
                f"MLflow experiment logged "
                f"successfully. Run ID: {run_id}"
            )


            logging.info(
                "Evaluation Pipeline completed"
            )


            return (
                model_evaluation_artifact
            )


        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e


if __name__ == "__main__":

    try:

        pipeline = (
            EvaluationPipeline()
        )

        pipeline.run_pipeline()

    except Exception as e:

        logging.exception(
            e
        )

        raise e