import logging
import zipfile
import requests
import tarfile

from pathlib import Path
from typing import Optional


def download_and_extract(
    url: str, dir: str, filename: Optional[str] = None
) -> str:
    path = Path(dir)
    path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1]

    file_path = path / filename
    result_path = str(file_path)

    if file_path.exists():
        logging.info(
            f"File '{filename}' already exists in '{dir}'. Skipping download."
        )
    else:
        logging.info(f"Downloading file '{filename}' from '{url}'...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Download successful. File saved to: '{file_path}'")

            if filename.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(path)
                file_path.unlink()
                result_path = str(path)
            elif (
                filename.endswith(".tar.gz")
                or filename.endswith(".tgz")
                or filename.endswith(".gz")
            ):
                with tarfile.open(file_path, "r:gz") as tar_ref:
                    tar_ref.extractall(path)
                file_path.unlink()
                result_path = str(path)
            elif filename.endswith(".tar"):
                with tarfile.open(file_path, "r") as tar_ref:
                    tar_ref.extractall(path)
                file_path.unlink()
                result_path = str(path)
            else:
                logging.info(
                    f"File '{filename}' is not an archive. No extraction needed."
                )

        except requests.RequestException as e:
            logging.error(f"Error downloading file from {url}: {str(e)}")
            raise
        except (zipfile.BadZipFile, tarfile.ReadError) as e:
            logging.error(f"Error extracting file '{filename}': {str(e)}")
            raise

    return result_path
