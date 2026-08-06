from src.constants import (
    PARAMS_FILE_PATH,

    DATASET_URL,
    DATA_INGESTION_ARTIFACTS_DIR,
    ZIP_FILE_NAME,
    ZIP_FILE_PATH,
    DATASET_PATH,

    DATA_VALIDATION_ARTIFACTS_DIR,
    VALIDATION_STATUS_FILE,
    REQUIRED_FOLDERS,

    DATA_TRANSFORMATION_ARTIFACTS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    IMAGENET_MEAN,
    IMAGENET_STD,

    MODEL_TRAINER_ARTIFACTS_DIR,
    TRAINED_MODEL_PATH,

    MODEL_EVALUATION_ARTIFACTS_DIR,
    METRICS_FILE_PATH,
    CONFUSION_MATRIX_FILE_PATH
)

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)

from src.utils.main_utils import (
    read_yaml
)


class ConfigurationManager:

    def __init__(self):

        self.params = read_yaml(
            PARAMS_FILE_PATH
        )

    def get_data_ingestion_config(
        self
    ) -> DataIngestionConfig:

        return DataIngestionConfig(

            DATASET_URL=DATASET_URL,

            DATA_INGESTION_ARTIFACTS_DIR=(
                DATA_INGESTION_ARTIFACTS_DIR
            ),

            ZIP_FILE_NAME=ZIP_FILE_NAME,

            ZIP_FILE_PATH=ZIP_FILE_PATH,

            DATASET_PATH=DATASET_PATH
        )

    def get_data_validation_config(
        self
    ) -> DataValidationConfig:

        return DataValidationConfig(

            DATA_VALIDATION_ARTIFACTS_DIR=(
                DATA_VALIDATION_ARTIFACTS_DIR
            ),

            VALIDATION_STATUS_FILE=(
                VALIDATION_STATUS_FILE
            ),

            REQUIRED_FOLDERS=REQUIRED_FOLDERS
        )

    def get_data_transformation_config(
        self
    ) -> DataTransformationConfig:

        return DataTransformationConfig(

            DATA_TRANSFORMATION_ARTIFACTS_DIR=(
                DATA_TRANSFORMATION_ARTIFACTS_DIR
            ),

            IMAGE_SIZE=IMAGE_SIZE,

            BATCH_SIZE=BATCH_SIZE,

            TRAIN_RATIO=TRAIN_RATIO,

            VALIDATION_RATIO=VALIDATION_RATIO,

            TEST_RATIO=TEST_RATIO,

            RANDOM_SEED=RANDOM_SEED,

            IMAGENET_MEAN=IMAGENET_MEAN,

            IMAGENET_STD=IMAGENET_STD
        )

    def get_model_trainer_config(
        self
    ) -> ModelTrainerConfig:

        return ModelTrainerConfig(

            MODEL_TRAINER_ARTIFACTS_DIR=(
                MODEL_TRAINER_ARTIFACTS_DIR
            ),

            TRAINED_MODEL_PATH=(
                TRAINED_MODEL_PATH
            ),

            NUM_CLASSES=(
                self.params["NUM_CLASSES"]
            ),

            EPOCHS=(
                self.params["EPOCHS"]
            ),

            LEARNING_RATE=(
                self.params["LEARNING_RATE"]
            ),

            MOMENTUM=(
                self.params["MOMENTUM"]
            )
        )

    def get_model_evaluation_config(
        self
    ) -> ModelEvaluationConfig:

        return ModelEvaluationConfig(

            MODEL_EVALUATION_ARTIFACTS_DIR=(
                MODEL_EVALUATION_ARTIFACTS_DIR
            ),

            TRAINED_MODEL_PATH=(
                TRAINED_MODEL_PATH
            ),

            METRICS_FILE_PATH=(
                METRICS_FILE_PATH
            ),

            CONFUSION_MATRIX_FILE_PATH=(
                CONFUSION_MATRIX_FILE_PATH
            ),

            NUM_CLASSES=(
                self.params["NUM_CLASSES"]
            )
        )