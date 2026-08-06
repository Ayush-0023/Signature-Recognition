import os
import mlflow

from dotenv import load_dotenv

from src.logger import logging
from src.exception import CustomException

import sys


class ExperimentTracker:
    """
    Handles MLflow experiment tracking.

    Responsibilities:
    1. Load MLflow/DagsHub credentials
    2. Configure MLflow tracking URI
    3. Create/select experiment
    4. Log model parameters
    5. Log evaluation metrics
    6. Log model and evaluation artifacts
    """

    def __init__(
        self,
        experiment_name: str = "Signature-Recognition-ResNet34"
    ):

        try:

            # ==========================================
            # Load environment variables from .env
            # ==========================================

            load_dotenv()


            # ==========================================
            # Read MLflow configuration
            # ==========================================

            self.tracking_uri = os.getenv(
                "MLFLOW_TRACKING_URI"
            )

            self.tracking_username = os.getenv(
                "MLFLOW_TRACKING_USERNAME"
            )

            self.tracking_password = os.getenv(
                "MLFLOW_TRACKING_PASSWORD"
            )


            # ==========================================
            # Validate configuration
            # ==========================================

            if not self.tracking_uri:

                raise ValueError(
                    "MLFLOW_TRACKING_URI is not set. "
                    "Please configure it in the .env file."
                )

            if not self.tracking_username:

                raise ValueError(
                    "MLFLOW_TRACKING_USERNAME is not set. "
                    "Please configure it in the .env file."
                )

            if not self.tracking_password:

                raise ValueError(
                    "MLFLOW_TRACKING_PASSWORD is not set. "
                    "Please configure it in the .env file."
                )


            # ==========================================
            # Configure MLflow
            # ==========================================

            mlflow.set_tracking_uri(
                self.tracking_uri
            )

            mlflow.set_experiment(
                experiment_name
            )


            self.experiment_name = (
                experiment_name
            )


            logging.info(
                "MLflow Experiment Tracker initialized"
            )

            logging.info(
                f"MLflow Experiment: "
                f"{self.experiment_name}"
            )

            logging.info(
                f"MLflow Tracking URI: "
                f"{self.tracking_uri}"
            )


        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e


    def log_experiment(
        self,
        params: dict,
        metrics: dict,
        model_path: str = None,
        confusion_matrix_path: str = None,
        run_name: str = None
    ):
        """
        Logs one complete ML experiment to MLflow.

        Parameters
        ----------
        params : dict
            Hyperparameters/model configuration.

        metrics : dict
            Evaluation metrics.

        model_path : str
            Path to trained PyTorch model.

        confusion_matrix_path : str
            Path to confusion matrix image.

        run_name : str
            Optional human-readable MLflow run name.
        """

        try:

            logging.info(
                "========== MLFLOW TRACKING STARTED =========="
            )


            # ==========================================
            # Start MLflow Run
            # ==========================================

            with mlflow.start_run(
                run_name=run_name
            ) as run:


                # ======================================
                # Log Parameters
                # ======================================

                logging.info(
                    "Logging parameters to MLflow"
                )

                mlflow.log_params(
                    params
                )


                # ======================================
                # Log Metrics
                # ======================================

                logging.info(
                    "Logging metrics to MLflow"
                )

                mlflow.log_metrics(
                    metrics
                )


                # ======================================
                # Log Trained Model File
                # ======================================

                if (
                    model_path
                    and os.path.exists(
                        model_path
                    )
                ):

                    logging.info(
                        "Logging trained model "
                        "artifact to MLflow"
                    )

                    mlflow.log_artifact(
                        model_path,
                        artifact_path="model"
                    )

                else:

                    logging.warning(
                        f"Model artifact not found: "
                        f"{model_path}"
                    )


                # ======================================
                # Log Confusion Matrix
                # ======================================

                if (
                    confusion_matrix_path
                    and os.path.exists(
                        confusion_matrix_path
                    )
                ):

                    logging.info(
                        "Logging confusion matrix "
                        "to MLflow"
                    )

                    mlflow.log_artifact(
                        confusion_matrix_path,
                        artifact_path="evaluation"
                    )

                else:

                    logging.warning(
                        "Confusion matrix artifact "
                        f"not found: "
                        f"{confusion_matrix_path}"
                    )


                # ======================================
                # Add useful tags
                # ======================================

                mlflow.set_tags(
                    {
                        "project": (
                            "End-to-End Signature Recognition"
                        ),
                        "model_architecture": (
                            "ResNet-34"
                        ),
                        "task": (
                            "Binary Image Classification"
                        ),
                        "framework": (
                            "PyTorch"
                        )
                    }
                )


                run_id = (
                    run.info.run_id
                )


                logging.info(
                    f"MLflow Run ID: "
                    f"{run_id}"
                )


            logging.info(
                "========== MLFLOW TRACKING COMPLETED =========="
            )


            return run_id


        except Exception as e:

            raise CustomException(
                e,
                sys
            ) from e