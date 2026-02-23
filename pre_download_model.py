import os
import sys
import traceback
from runpod import RunPodLogger
import json

log = RunPodLogger()

def download_one_file(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        log.info(f"File already exists, skipping download: {dest_path}")
        return
    # use curl to download the file
    import subprocess
    log.info(f"Downloading model from URL: {url}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    subprocess.run(["curl", "-L", url, "-o", dest_path], check=True)

def download_model_from_url(url: str) -> str:
    expectedFilename = os.path.basename(url)
    dest_path = os.path.join("models", expectedFilename)
    
    if "-of-" in expectedFilename:
        totalAmount = int(expectedFilename.split("-of-")[-1][:5])
        # last 5 characters before the -of- should be the current chunk number, zero-padded to 5 digits
        currentChunk = int(expectedFilename.split("-of-")[-2][-5:])

        # check NaN
        if totalAmount <= 0 or currentChunk <= 0:
            raise ValueError(f"Invalid chunk numbers in URL: {currentChunk}-of-{totalAmount}")

        # check they are integers and currentChunk is not less than 1 and not more than totalAmount
        if currentChunk < 1 or currentChunk > totalAmount:
            raise ValueError(f"Invalid chunk number in URL: {currentChunk} (total: {totalAmount})")
        if totalAmount < 1:
            raise ValueError(f"Invalid total amount in URL: {totalAmount}")
        
        if "-of-" not in dest_path:
            raise ValueError(f"Destination path must contain '-of-' for chunked downloads: {dest_path}")
        
        for chunk_num in range(1, totalAmount + 1):
            padded_chunk_num = str(chunk_num).zfill(5)
            chunk_url = url.replace("00001", padded_chunk_num)
            chunk_dest_path = dest_path.replace("00001", padded_chunk_num)
            download_one_file(chunk_url + "?download=1", chunk_dest_path)
    else:
        download_one_file(url + "?download=1", dest_path)

    log.info(f"Model downloaded to: {dest_path}")
    return dest_path

if __name__ == "__main__":
    try:
        log.info("Pre downloading files...")

        env_config = os.getenv("CONFIG_JSON", "Missing CONFIG_JSON env variable")
        log.info(f"CONFIG_JSON: {env_config}")

        if not env_config:
            raise ValueError("CONFIG_JSON environment variable is not set")
        try:
            CONFIG = json.loads(env_config, strict=False)
            CONFIG_PATH = "ENV"
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in CONFIG_JSON environment variable: {e}")
        
        model_url = CONFIG.get("modelUrl", None)
        if not model_url:
            sys.exit(0)  # No model URL provided, skip downloading

        if not isinstance(model_url, str):
            raise ValueError("modelUrl in CONFIG_JSON must be a string")
        
        download_model_from_url(model_url)

    except Exception as e:
        log.error(f"Worker pre download model failed: {e}\n{traceback.format_exc()}")
