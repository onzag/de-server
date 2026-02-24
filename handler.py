import sys
import traceback
import os

import runpod
import multiprocessing
from runpod import RunPodLogger

import base

CACHE_DIR = "/runpod-volume/huggingface-cache/hub"

def find_model_path():
    """
    Find the path to a cached model.
    
    Args:
        model_name: The model name from Hugging Face
        (e.g., 'Qwen/Qwen2.5-0.5B-Instruct')
    
    Returns:
        The full path to the cached model, or None if not found
    """
    # Convert model name format: "Org/Model" -> "models--Org--Model"
    # cache_name = model_name.replace("/", "--")
    # snapshots_dir = os.path.join(CACHE_DIR, f"models--{cache_name}", "snapshots")
    
    # Check if the model exists in cache
    if os.path.exists(CACHE_DIR):
        models = os.listdir(CACHE_DIR)
        if models:
            log.info(f"Found cached models: {models}")
            first_model = models[0]
            snapshots_dir = os.path.join(CACHE_DIR, first_model, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    log.info(f"Found snapshots for model {first_model}: {snapshots}")
                    first_snapshot = snapshots[0]
                    model_path = os.path.join(snapshots_dir, first_snapshot)
                    log.info(f"Using model path: {model_path}")
                    return model_path
                else:
                    log.error(f"No snapshots found for model {first_model} in cache")
            else:
                log.error(f"No snapshots directory found for model {first_model} in cache")
    else:
        log.error(f"Cache directory {CACHE_DIR} does not exist, could not find model")
            
    
    return None

# from base import load_config
# import base

log = RunPodLogger()

def handler(job):
    """
    This is a simple handler that takes a name as input and returns a greeting.
    The job parameter contains the input data in job["input"]
    """
    job_input = job["input"]
    
    try:
        # Get the name from the input, default to "World" if not provided
        if job_input.get('action') == 'count-tokens':
            if not job_input.get('payload') or not isinstance(job_input['payload'].get('text'), str):
                raise ValueError("Invalid payload for count-tokens")
            text = job_input['payload']['text']
            tokens = base.MODEL.tokenizer.encode(text)
            yield {
                "type": "count",
                "rid": job_input.get('rid', 'no-rid'),
                "n_tokens": len(tokens)
            }

        elif job_input.get("action") == "infer":
            for result in base.generate_completion(job_input['payload']):
                if 'error' in result:
                    yield {"error": str(result['error']), "rid": job_input.get('rid', 'no-rid')}
                elif 'token' in result:
                    yield {"type": "token", "rid": job_input.get('rid', 'no-rid'), "text": result['token']}
                elif 'done' in result and result['done']:
                    yield {"type": "done", "rid": job_input.get('rid', 'no-rid')}
                elif 'request_id' in result:
                    pass

        elif job_input.get("action") == "analyze-prepare":
            for result in base.prepare_analysis(job_input['payload']):
                if 'error' in result:
                    yield {"error": str(result['error']), "rid": job_input.get('rid', 'no-rid')}
                elif 'done' in result and result['done']:
                    yield {"type": "analyze-ready", "rid": job_input.get('rid', 'no-rid')}

        elif job_input.get("action") == "analyze-question":
            for result in base.run_question(job_input['payload']):
                if 'error' in result:
                    yield {"error": str(result['error']), "rid": job_input.get('rid', 'no-rid')}
                elif 'answer' in result:
                    yield {"type": "answer", "rid": job_input.get('rid', 'no-rid'), "text": result['answer']}
                elif 'request_id' in result:
                    pass

        else:
            yield {"error": "Unsupported action", "rid": job_input.get('rid', 'no-rid')}

    except Exception as e:
        error_str = str(e)
        full_traceback = traceback.format_exc()

        log.error(f"Error during inference: {error_str}")
        log.error(f"Full traceback:\n{full_traceback}")

        # CUDA errors = worker is broken, exit to let RunPod spin up a healthy one
        if "CUDA" in error_str or "cuda" in error_str:
            log.error("Terminating worker due to CUDA/GPU error")
            sys.exit(1)

        yield {"error": error_str, "rid": job_input.get('rid', 'no-rid')}

# Only run in main process to prevent re-initialization when vLLM spawns worker subprocesses
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    # Start the serverless function
    try:
        log.info("Starting worker...")

        envConfig = os.getenv("CONFIG_JSON", "Missing CONFIG_JSON env variable")
        log.info(f"CONFIG_JSON: {envConfig}")

        model_path = find_model_path()
        if model_path is None:
            log.error("No model found in cache, exiting")
            sys.exit(1)

        base.load_config("ENV", model_path_override=model_path)
    except Exception as e:
        log.error(f"Worker startup failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    log.info("Starting worker...")

    runpod.serverless.start(
        {
            "handler": handler,
            # read from the environment variables, if not set, default to 5
            "concurrency_modifier": lambda x: int(os.getenv("CONCURRENCY_MODIFIER", "5")),
            "return_aggregate_stream": True,
        }
    )