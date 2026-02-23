import sys
import traceback
import os

import runpod
import multiprocessing
from runpod import RunPodLogger

from base import load_config
import base

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

        load_config("ENV")
    except Exception as e:
        log.error(f"Worker startup failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    runpod.serverless.start(
        {
            "handler": handler,
            # read from the environment variables, if not set, default to 5
            "concurrency_modifier": int(os.getenv("CONCURRENCY_MODIFIER", "5")),
            "return_aggregate_stream": True,
        }
    )