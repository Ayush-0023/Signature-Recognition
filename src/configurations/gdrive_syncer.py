import os
import sys
import gdown
from src.logger import logging
from src.exception import CustomException

class GDriveSync:

    def sync_file_from_gdrive(
        self,
        file_url: str,
        destination: str
    ) -> None:
        """
        Downloads a file from Google Drive
        to the specified destination.
        """

        try:
            logging.info(
                "Entered sync_file_from_gdrive method"
            )

            os.makedirs(
                os.path.dirname(destination),
                exist_ok=True
            )

            logging.info(
                f"Downloading file from Google Drive: {file_url}"
            )

            downloaded_file = gdown.download(
                url=file_url,
                output=destination,
                quiet=False,
                fuzzy=True
            )

            if downloaded_file is None:
                raise Exception(
                    "Google Drive download failed. "
                    "Check the URL and sharing permissions."
                )

            logging.info(
                f"File downloaded successfully to: {destination}"
            )

            logging.info(
                "Exited sync_file_from_gdrive method"
            )

        except Exception as e:
            raise CustomException(e, sys) from e