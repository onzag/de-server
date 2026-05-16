# downloads a model from huggingface, requires wget to be installed
# this script was made to run in runpods with bad memory
# so have to use some tricks due to runpod's bad memory management

import subprocess
import os
import sys
import json

# Keep this in sync with MODES in base.py
SUPPORTED_MODES = ("mistral", "llama3", "chatml", "gemma", "phi", "deepseek", "alpaca")

if len(sys.argv) < 3:
    print("Usage: python download-model.py <model-type> <model-name>")
    print("  <model-type>: one of " + ", ".join(SUPPORTED_MODES))
    sys.exit(1)

model_type = sys.argv[1]
model_name = sys.argv[2]

if model_type not in SUPPORTED_MODES:
    print(f"Error: unknown model-type '{model_type}'.")
    print("Supported model types: " + ", ".join(SUPPORTED_MODES))
    sys.exit(1)

# make the models directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

general_info_url = "https://huggingface.co/api/models/" + model_name

try:
    # get the general info about the model
    result = subprocess.run(["wget", "-qO-", general_info_url], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error fetching model info:", result.stderr)
        sys.exit(1)

    model_info = json.loads(result.stdout)
except subprocess.CalledProcessError as e:
    print("Error downloading model:", e)
    sys.exit(1)
except json.JSONDecodeError as e:
    print("Error parsing model info:", e)
    sys.exit(1)

# get the list of files in the model repository
files_url = "https://huggingface.co/api/models/" + model_name + "/tree/main"

try:
    result = subprocess.run(["wget", "-qO-", files_url], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error fetching model files:", result.stderr)
        sys.exit(1)

    files_info = json.loads(result.stdout)
except subprocess.CalledProcessError as e:
    print("Error downloading model files:", e)
    sys.exit(1)
except json.JSONDecodeError as e:
    print("Error parsing model files info:", e)
    sys.exit(1)

for file in files_info:
    if file["type"] == "file":
        file_url = "https://huggingface.co/" + model_name + "/resolve/main/" + file["path"] + "?download=true"
        output_path = os.path.join("models", model_name, file["path"])
        output_dir = os.path.dirname(output_path)
        file_size = file["size"]
        
        # make the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # check if the specific file already exists, if it does, skip downloading
        if os.path.exists(output_path):
            print(f"File {file['path']} already exists")
            # check if the file size matches, if it doesn't, redownload the file
            if os.path.getsize(output_path) == file_size:
                print(f"File {file['path']} size matches, skipping download")
                continue
            else:
                print(f"File {file['path']} size mismatch, redownloading")

        try:
            result = subprocess.run(["wget", "-O", output_path, file_url])
            if result.returncode != 0:
                print(f"Error downloading {file['path']}:", result.stderr)
                continue
            print(f"Downloaded {file['path']}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {file['path']}:", e)

default_json = {
    "modelPath": "./models/" + model_name,
    "enforceEager": False,
    "mode": model_type,
    "standard": {
        "temperature": 1.0,
        "dynamicTemperature": [0.8, 1.05],
        "minP": 0.025,
        "dry": {
            "multiplier": 0.8,
            "base": 1.74,
            "length": 5
        },
        "maxTokens": 1024
    },
    "analyze": {
        "temperature": 0.4,
        "topP": 0.8,
        "maxTokens": 512
    }
}

config_path = "./models/" + model_name.replace("/", "_") + ".json"
if not os.path.exists(config_path):
    print("Creating default model loading json file at", config_path)
    with open(config_path, "w") as f:
        json.dump(default_json, f, indent=4)

print("Model download complete")