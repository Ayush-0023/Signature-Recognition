import os
import sys

import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms

from torchvision.models import (
    resnet34
)

from src.logger import logging
from src.exception import CustomException

from src.constants import (
    TRAINED_MODEL_PATH,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CLASS_NAMES
)

from src.utils.main_utils import (
    read_yaml
)

from src.constants import (
    PARAMS_FILE_PATH
)


class PredictionPipeline:

    def __init__(self):

        try:
            logging.info(
                "Initializing Prediction Pipeline"
            )

            self.params = read_yaml(
                PARAMS_FILE_PATH
            )

            self.device = (
                self.get_device()
            )

            self.transform = (
                self.get_transform()
            )

            self.model = (
                self.load_model()
            )

            logging.info(
                "Prediction Pipeline initialized successfully"
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_device(self):
        """
        Selects the best available computation device.
        """

        if torch.cuda.is_available():

            device = torch.device(
                "cuda"
            )

        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):

            device = torch.device(
                "mps"
            )

        else:

            device = torch.device(
                "cpu"
            )

        logging.info(
            f"Prediction device: {device}"
        )

        return device

    def get_transform(self):
        """
        Creates the same preprocessing used during
        validation and testing.

        IMPORTANT:
        Random augmentation is NOT used during
        prediction.
        """

        try:
            prediction_transform = (
                transforms.Compose([

                    transforms.Resize(
                        (
                            IMAGE_SIZE,
                            IMAGE_SIZE
                        )
                    ),

                    transforms.ToTensor(),

                    transforms.Normalize(
                        mean=IMAGENET_MEAN,
                        std=IMAGENET_STD
                    )
                ])
            )

            return prediction_transform

        except Exception as e:
            raise CustomException(e, sys) from e

    def load_model(self):
        """
        Reconstructs ResNet-34 architecture and loads
        the trained model weights.
        """

        try:
            logging.info(
                "Loading trained model for prediction"
            )

            if not os.path.exists(
                TRAINED_MODEL_PATH
            ):

                raise FileNotFoundError(
                    f"Trained model not found at: "
                    f"{TRAINED_MODEL_PATH}"
                )

            model = resnet34(
                weights=None
            )

            number_of_features = (
                model.fc.in_features
            )

            model.fc = nn.Linear(
                number_of_features,
                self.params["NUM_CLASSES"]
            )

            state_dict = torch.load(
                TRAINED_MODEL_PATH,
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

    def predict(
        self,
        image_path: str
    ) -> dict:
        """
        Predicts whether a signature is
        Forged or Original.

        Returns:
        {
            "prediction": "Original",
            "confidence": 0.9876
        }
        """

        try:
            logging.info(
                f"Starting prediction for: "
                f"{image_path}"
            )

            # ==========================================
            # Validate image path
            # ==========================================

            if not os.path.exists(
                image_path
            ):

                raise FileNotFoundError(
                    f"Image not found at: "
                    f"{image_path}"
                )

            # ==========================================
            # Open image
            # ==========================================

            image = Image.open(
                image_path
            )

            # ResNet expects RGB input
            image = image.convert(
                "RGB"
            )

            # ==========================================
            # Apply preprocessing
            # ==========================================

            image_tensor = (
                self.transform(
                    image
                )
            )

            # Current shape:
            #
            # [3, 224, 224]
            #
            # Model expects:
            #
            # [batch, channels, height, width]

            image_tensor = (
                image_tensor
                .unsqueeze(0)
                .to(self.device)
            )

            # ==========================================
            # Inference
            # ==========================================

            with torch.no_grad():

                outputs = self.model(
                    image_tensor
                )

                probabilities = (
                    torch.softmax(
                        outputs,
                        dim=1
                    )
                )

                confidence, predicted_class = (
                    torch.max(
                        probabilities,
                        dim=1
                    )
                )

            predicted_index = (
                predicted_class.item()
            )

            confidence_score = (
                confidence.item()
            )

            predicted_label = (
                CLASS_NAMES[
                    predicted_index
                ]
            )

            result = {
                "prediction": predicted_label,
                "confidence": round(
                    confidence_score,
                    4
                )
            }

            logging.info(
                f"Prediction result: "
                f"{result}"
            )

            return result

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: "
            "python -m src.pipeline.prediction "
            "<image_path>"
        )

        sys.exit(1)

    image_path = (
        sys.argv[1]
    )

    predictor = (
        PredictionPipeline()
    )

    result = predictor.predict(
        image_path
    )

    print("\nPrediction Result")
    print("-----------------")
    print(
        f"Prediction : "
        f"{result['prediction']}"
    )
    print(
        f"Confidence : "
        f"{result['confidence'] * 100:.2f}%"
    )