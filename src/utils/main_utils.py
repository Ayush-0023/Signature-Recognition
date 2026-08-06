import os
import sys
import yaml
import dill

from src.exception import CustomException


def read_yaml(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents
    as a Python dictionary.

    :param file_path: Path to the YAML file
    :return: Dictionary containing YAML data
    """

    try:
        with open(file_path, "r") as yaml_file:

            content = yaml.safe_load(yaml_file)

            return content

    except Exception as e:
        raise CustomException(e, sys) from e


def write_yaml_file(
    file_path: str,
    content: object,
    replace: bool = False
) -> None:
    """
    Writes content to a YAML file.
    """

    try:

        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(
            os.path.dirname(file_path)
            if os.path.dirname(file_path)
            else ".",
            exist_ok=True
        )

        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise CustomException(e, sys) from e


def save_object(
    file_path: str,
    obj: object
) -> None:
    """
    Saves a Python object using dill.
    """

    try:

        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True
        )

        with open(file_path, "wb") as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(
    file_path: str
) -> object:
    """
    Loads a Python object saved using dill.
    """

    try:

        if not os.path.exists(file_path):

            raise Exception(
                f"The file does not exist: {file_path}"
            )

        with open(file_path, "rb") as file:
            return dill.load(file)

    except Exception as e:
        raise CustomException(e, sys) from e