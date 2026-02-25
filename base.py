"""
USAGE:

python base.py <path to config json>

Example:
Windows:
python base.py .\\testing\\model.json
Unix/Linux/Mac:
python base.py ./testing/model.json

Example (debug mode):
Windows:
$env:DEBUG=1; python base.py .\\testing\\model.json
Unix/Linux/Mac:
DEBUG=1 python base.py ./testing/model.json

Remember in Windows:
Remove-Item Env:DEBUG

JSON File settings example:

{
    // the path of the model relative to the json file
    "modelPath": "./model.gguf",
    // standard generation used in roleplay contexts
    "standard": {
        "temperature": 1.0,
        "maxTokens": 512,
        "dynamicTemperature": [0.8, 1.05],
        "minP": 0.025,
        "dry": {
            "multiplier": 0.8,
            "base": 1.74,
            "length": 5
        }
    },
    "analyze": {
        "temperature": 0.4,
        "topP": 0.8,
        "topK": 40,
        "repeatPenalty": 1.1,
        "frequencyPenalty": 0.0,
        "presencePenalty": 0.0,
        "maxTokens": 512
    }
}
"""

import json
import os
import re
import sys
import random
from typing import Any, Callable, Optional
#from vllm.engine.async_llm_engine import AsyncLLM

CONTEXT_WINDOW_SIZE = 32768
if os.environ.get("CONTEXT_WINDOW_SIZE"):
    try:
        CONTEXT_WINDOW_SIZE = int(os.environ.get("CONTEXT_WINDOW_SIZE"))
        print(f"Using custom context window size from environment variable: {CONTEXT_WINDOW_SIZE}")
    except ValueError:
        print(f"Invalid CONTEXT_WINDOW_SIZE environment variable: {os.environ.get('CONTEXT_WINDOW_SIZE')}. Using default: {CONTEXT_WINDOW_SIZE}")

GPU_MEM_UTILIZATION = 0.96
if os.environ.get("GPU_MEM_UTILIZATION"):
    try:
        GPU_MEM_UTILIZATION = float(os.environ.get("GPU_MEM_UTILIZATION"))
        print(f"Using custom GPU memory utilization from environment variable: {GPU_MEM_UTILIZATION}")
    except ValueError:
        print(f"Invalid GPU_MEM_UTILIZATION environment variable: {os.environ.get('GPU_MEM_UTILIZATION')}. Using default: {GPU_MEM_UTILIZATION}")

# ── Module-level state (mirrors the JS globals) ──────────────────────────

MODEL: Optional[Any] = None
MODEL_PATH: str = ""
_REQUEST_COUNTER: int = 0
CONFIG: Optional[dict] = None
CONFIG_PATH: str = ""
ANALYSIS_TEXT: Optional[str] = None
DEBUG: bool = os.environ.get("DEBUG", "") == "1"

def get_num_gpus():
    # Try reading environment variable first
    num_gpus_env = os.environ.get("NUM_GPUS")
    if num_gpus_env is not None:
        try:
            return int(num_gpus_env)
        except ValueError:
            pass

    # Try PyTorch first
    try:
        import torch
        count = torch.cuda.device_count()
        if count > 0:
            return count
    except Exception:
        pass

    return 0


# ── Helpers ───────────────────────────────────────────────────────────────

def escape_regexp(string: str) -> str:
    """Escape special regex characters."""
    return re.escape(string)


def pattern_repetition_checker(text: str, min_length: int, max_length: int):
    """
    Check if `text` is composed entirely of a repeated pattern of length
    between min_length and max_length.
    Returns {"repetitionAt": pattern, "amount": count} or None.
    """
    if len(text) < min_length:
        return None
    for length in range(min_length, min(max_length, len(text)) + 1):
        pattern = text[:length]
        parts = text.split(pattern)
        if all(s == "" for s in parts):
            occurrences = len(parts) - 1
            if occurrences > 1:
                return {"repetitionAt": pattern, "amount": occurrences}
    return None


def aggressive_list_repetition_checker(text: str) -> bool:
    """
    Returns True when a comma-separated list contains a repeated item
    after at least one different item has appeared.
    """
    parts = [s.strip() for s in text.split(",") if s.strip()]
    if len(parts) <= 1:
        return False
    first = parts[0]
    has_had_different = False
    for item in parts[1:]:
        if item == first and has_had_different:
            return True
        elif item != first:
            has_had_different = True
    return False


def check_config_validity(config: dict) -> None:
    """Validate a generation-config section (standard / analyze)."""
    if not isinstance(config.get("maxTokens"), (int, float)):
        raise ValueError("Invalid config: maxTokens must be a number")
    if not isinstance(config.get("temperature"), (int, float)):
        raise ValueError("Invalid config: temperature must be a number")

    tr = config.get("temperatureRange")
    if tr is not None:
        if (not isinstance(tr, list) or len(tr) != 2
                or not isinstance(tr[0], (int, float))
                or not isinstance(tr[1], (int, float))):
            raise ValueError("Invalid config: temperatureRange must be a list of two numbers")

    for key in ("topP", "repeatPenalty", "frequencyPenalty", "presencePenalty", "minP"):
        val = config.get(key)
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError(f"Invalid config: {key} must be a number")

    dry = config.get("dry")
    if dry is not None:
        if not isinstance(dry, dict):
            raise ValueError("Invalid config: dry must be an object")
        for field in ("multiplier", "base", "length"):
            if not isinstance(dry.get(field), (int, float)):
                raise ValueError(f"Invalid config: dry.{field} must be a number")

    xtc = config.get("xtc")
    if xtc is not None:
        if not isinstance(xtc, dict):
            raise ValueError("Invalid config: xtc must be an object")


def get_dynamic_temperature(min_temp: float, max_temp: float) -> float:
    return random.random() * (max_temp - min_temp) + min_temp


# ── Model / config loading ────────────────────────────────────────────────

def _next_request_id() -> str:
    global _REQUEST_COUNTER
    _REQUEST_COUNTER += 1
    return f"req-{_REQUEST_COUNTER}"


def load_model(model_path: str, tokenizer_path: str | None = None, enforce_eager: bool = True) -> None:
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    global MODEL, MODEL_PATH, CONTEXT_WINDOW_SIZE

    print(f"Loading model: {model_path}")
    if MODEL_PATH == model_path and MODEL is not None:
        print("Model already loaded")
        return

    if MODEL is not None:
        print("Unloading previous model")
        MODEL = None
        MODEL_PATH = ""

    num_gpus = get_num_gpus()

    print(f"Detected {num_gpus} GPUs. Loading model with tensor_parallel_size={num_gpus}.")

    engine_kwargs: dict[str, Any] = dict(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEM_UTILIZATION,
        dtype="auto",
        enforce_eager=enforce_eager,
        tensor_parallel_size=num_gpus,
        max_model_len=CONTEXT_WINDOW_SIZE,
    )

    if tokenizer_path:
        print(f"Using pre-saved tokenizer: {tokenizer_path}")
        engine_kwargs["tokenizer"] = tokenizer_path

    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    MODEL = engine
    MODEL_PATH = model_path
    print("Model loaded successfully")

def save_tokenizer(model_path: str, tokenizer_path: str) -> None:
    from transformers import AutoTokenizer
    
    # check if tokenizer_path already exists
    if os.path.exists(tokenizer_path):
        print(f"Tokenizer path already exists: {tokenizer_path}")
        return

    print(f"Loading tokenizer from GGUF: {model_path}")
    print("This will be slow...")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.dirname(model_path),
        gguf_file=os.path.basename(model_path),
    )

    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to: {tokenizer_path}")

def load_config(config_path: str, model_path_override: str | None = None) -> None:
    global CONFIG, CONFIG_PATH

    print(f"Loading config: {config_path}")
    if (config_path == "ENV"):
        env_config = os.environ.get("CONFIG_JSON")
        if not env_config:
            raise ValueError("CONFIG_JSON environment variable is not set")
        try:
            CONFIG = json.loads(env_config, strict=False)
            CONFIG_PATH = "ENV"
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in CONFIG_JSON environment variable: {e}")
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = f.read()
        CONFIG = json.loads(raw, strict=False)
        CONFIG_PATH = config_path

    # check config is a dict
    if not isinstance(CONFIG, dict):
        print(CONFIG)
        raise ValueError("Invalid config file, expected a JSON object at the top level")

    if "standard" not in CONFIG or "analyze" not in CONFIG:
        print(CONFIG)
        raise ValueError("Invalid config file, missing standard or analyze sections")

    check_config_validity(CONFIG["standard"])
    check_config_validity(CONFIG["analyze"])
    print("Config loaded successfully")

    if model_path_override is not None:
        print(f"Overriding modelPath in config with: {model_path_override}")
        CONFIG["modelPath"] = model_path_override

    if CONFIG.get("modelPath") is None:
        raise ValueError("Invalid config: modelPath should be set")
        
    if (CONFIG.get("tokenizerPath") is not None) and not isinstance(CONFIG["tokenizerPath"], str):
        raise ValueError("Invalid config: tokenizerPath must be a string if provided")
    
    if (CONFIG.get("mode") is not None) and CONFIG["mode"] not in ("mistral", "llama3"):
        raise ValueError("Invalid config: mode must be 'mistral' or 'llama3' if provided")
    
    if not isinstance(CONFIG.get("enforceEager"), bool):
        raise ValueError("Invalid config: enforceEager must be a boolean")

    if CONFIG.get("tokenizerPath") is not None:
        save_tokenizer(CONFIG["modelPath"], CONFIG["tokenizerPath"])

    if MODEL_PATH != CONFIG["modelPath"]:
        base_dir = os.path.dirname(config_path)
        model_full_path = os.path.normpath(os.path.join(base_dir, CONFIG["modelPath"]))

        if not os.path.exists(model_full_path):
            model_full_path = CONFIG["modelPath"]  # try as absolute path
            if not os.path.exists(model_full_path):
                raise FileNotFoundError(f"Model file not found at: {model_full_path}")

        print(f"Resolved model path: {model_full_path}")

        tokenizer_path = None
        if CONFIG.get("tokenizerPath"):
            tokenizer_path = os.path.normpath(os.path.join(base_dir, CONFIG["tokenizerPath"]))
            if not os.path.exists(tokenizer_path):
                tokenizer_path = CONFIG["tokenizerPath"]  # try as absolute path
                if not os.path.exists(tokenizer_path):
                    raise FileNotFoundError(f"Tokenizer path not found at: {tokenizer_path}")
        print(f"Resolved tokenizer path: {tokenizer_path}")

        load_model(model_full_path, tokenizer_path=tokenizer_path, enforce_eager=CONFIG.get("enforceEager", True))


# ── Sampling-params builder helpers ───────────────────────────────────────

def _build_stop_tokens(config: dict, extra_stops: list[str] | None = None) -> list[str]:
    """Return the list of stop strings depending on chat-template mode."""
    mode = config.get("mode")
    if mode == "mistral":
        stops = ["</s>", "[INST]"]
    else:
        stops = ["<|eot_id|>", "<|start_header_id|>"]
    if extra_stops:
        stops.extend(extra_stops)
    return stops


def _make_sampling_params(
    section: dict,
    stop: list[str],
    grammar: str | None = None,
) -> Any:
    from vllm import SamplingParams

    """
    Build a vLLM SamplingParams from a config section (standard / analyze).
    """
    temperature = section["temperature"]
    tr = section.get("temperatureRange")
    if tr:
        temperature = get_dynamic_temperature(tr[0], tr[1])

    kwargs: dict[str, Any] = dict(
        temperature=temperature,
        max_tokens=section.get("maxTokens", 512),
        stop=stop,
    )

    if section.get("topP") is not None:
        kwargs["top_p"] = section["topP"]
    if section.get("minP") is not None:
        kwargs["min_p"] = section["minP"]
    if section.get("repeatPenalty") is not None:
        kwargs["repetition_penalty"] = section["repeatPenalty"]
    if section.get("frequencyPenalty") is not None:
        kwargs["frequency_penalty"] = section["frequencyPenalty"]
    if section.get("presencePenalty") is not None:
        kwargs["presence_penalty"] = section["presencePenalty"]

    if grammar is not None:
        # vLLM grammar/structured output support varies by version
        try:
            from vllm.sampling_params import GuidedDecodingParams
            kwargs["guided_decoding"] = GuidedDecodingParams(grammar=grammar)
        except ImportError:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                kwargs["structured_outputs"] = StructuredOutputsParams(grammar=grammar)
            except ImportError:
                print("WARNING: Grammar constraints requested but not supported by this vLLM version. Ignoring grammar.")

    return SamplingParams(**kwargs)


# ── Prompt formatting ─────────────────────────────────────────────────────

def _format_analysis_prompt(config: dict, system: str, user_trail: str) -> str:
    if config.get("mode") == "mistral":
        return f"<s>[SYSTEM_PROMPT] {system}[/SYSTEM_PROMPT][INST] {user_trail}"
    else:
        return (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_trail}"
        )


def _format_question_prompt(config: dict, analysis_text: str, question: str, trail: str | None) -> str:
    if config.get("mode") == "mistral":
        prompt = analysis_text + "\n\n" + question + "\n[/INST]\n\n" + (trail or "")
    else:
        prompt = (
            analysis_text + "\n" + question
            + "\n<|start_header_id|>assistant<|end_header_id|>\n\n"
            + (trail or "")
        )
    return prompt


def _format_chat_prompt(config: dict, messages: list[dict], trail: str | None) -> str:
    prompt = ""
    if config.get("mode") == "mistral":
        prompt += "<s>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if config.get("mode") == "mistral":
            if role == "system":
                prompt += f"[SYSTEM_PROMPT] {content}[/SYSTEM_PROMPT][INST]"
            else:
                prompt += "\n\n" + content
        else:
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id>"

    if config.get("mode") == "mistral":
        prompt += "[/INST]\n\n"
    else:
        prompt += "\n<|start_header_id|>assistant<|end_header_id|>\n\n"

    if trail:
        prompt += trail
    return prompt


# ── Post-processing helpers ───────────────────────────────────────────────

def _strip_trailing_newlines(text: str) -> str:
    return text.rstrip('\n')


# ── Public API (mirrors JS exports) ──────────────────────────────────────

async def prepare_analysis(
    data: dict,
):
    """
    Mirrors prepareAnalysis(). Caches the formatted analysis prompt text.
    """
    global ANALYSIS_TEXT

    if MODEL is None:
        raise RuntimeError("Model not loaded")
    if CONFIG is None:
        raise RuntimeError("Config not loaded")
    if not isinstance(data.get("system"), str) or not data["system"]:
        raise ValueError("Invalid system format")
    if not isinstance(data.get("userTrail"), str) or not data["userTrail"]:
        raise ValueError("Invalid userTrail format")

    try:
        ANALYSIS_TEXT = _format_analysis_prompt(CONFIG, data["system"], data["userTrail"])
        if DEBUG:
            print("Prepared analysis text:", ANALYSIS_TEXT)
        yield {"done": True}
    except Exception as e:
        yield {"error": str(e)}


async def run_question(
    data: dict,
):
    """
    Mirrors runQuestion(). Uses the analyze config section.
    Generates a full completion, then post-processes for paragraph/character
    limits, stop triggers, and repetition checks.
    """
    global ANALYSIS_TEXT

    if MODEL is None:
        raise RuntimeError("Model not loaded")
    if CONFIG is None:
        raise RuntimeError("Config not loaded")
    if ANALYSIS_TEXT is None:
        raise RuntimeError("Analysis not prepared")

    # ── Validate inputs ──
    if not isinstance(data.get("question"), str) or not data["question"]:
        raise ValueError("Invalid question format")
    if not isinstance(data.get("stopAt"), list):
        raise ValueError("Invalid stopAt format")
    if not isinstance(data.get("stopAfter"), list):
        raise ValueError("Invalid stopAfter format")
    if not isinstance(data.get("maxParagraphs"), (int, float)) or data["maxParagraphs"] < 0:
        raise ValueError("Invalid maxParagraphs format")
    if not isinstance(data.get("maxCharacters"), (int, float)) or data["maxCharacters"] < 0:
        raise ValueError("Invalid maxCharacters format")
    if not isinstance(data.get("maxSafetyCharacters"), (int, float)) or data["maxSafetyCharacters"] < 0:
        raise ValueError("Invalid maxSafetyCharacters format")
    if data.get("trail") is not None and not isinstance(data["trail"], str):
        raise ValueError("Invalid trail format")
    if data.get("grammar") is not None and not isinstance(data["grammar"], str):
        raise ValueError("Invalid grammar format")
    if data.get("repetitionBuster") is not None and not isinstance(data["repetitionBuster"], bool):
        raise ValueError("Invalid repetitionBuster format")
    if data.get("aggressiveListRepetitionBuster") is not None and not isinstance(data["aggressiveListRepetitionBuster"], bool):
        raise ValueError("Invalid aggressiveListRepetitionBuster format")

    regex_stop_after = [
        re.compile(rf"(^|[.,;])\s*{escape_regexp(s)}\s*([.,;]|$)", re.IGNORECASE)
        for s in data["stopAfter"]
    ]

    prompt = _format_question_prompt(CONFIG, ANALYSIS_TEXT, data["question"], data.get("trail"))

    stop_tokens = _build_stop_tokens(CONFIG, data.get("stopAt"))
    sampling = _make_sampling_params(CONFIG["analyze"], stop_tokens, grammar=data.get("grammar"))

    if DEBUG:
        print("Generation config: " + str(sampling))
        print("Prompt: " + str(prompt))
        print("Using grammar: " + str(data.get("grammar")))

    max_paragraphs = int(data["maxParagraphs"])
    max_characters = int(data["maxCharacters"])
    max_safety_characters = int(data["maxSafetyCharacters"])
    if max_paragraphs:
        print("Max paragraphs limit set to: " + str(max_paragraphs))
    if max_characters:
        print("Max characters limit set to: " + str(max_characters))
    if max_safety_characters:
        print("Max safety characters limit set to: " + str(max_safety_characters))

    try:
        request_id = _next_request_id()
        yield {"request_id": request_id}
        generation = MODEL.generate(prompt, sampling_params=sampling, request_id=request_id)

        answer = ""
        prev_len = 0

        async for output in generation:
            if (output.request_id != request_id):
                print("SKIPPED OUTPUT from other request: " + str(output.request_id))
                continue  # Ignore outputs from other requests

            new_text = output.outputs[0].text
            delta = new_text[prev_len:]
            prev_len = len(new_text)

            if not delta:
                continue

            if DEBUG:
                sys.stdout.write(delta + "\u00a7")
                sys.stdout.flush()

            answer += delta

            # ── Mid-stream checks (mirrors JS onTextChunk) ──
            if max_paragraphs > 0:
                para_count = 0
                for i in range(len(answer) - 1):
                    if answer[i] == '\n' and answer[i + 1] == '\n':
                        para_count += 1
                        if para_count >= max_paragraphs:
                            part_before = delta.split('\n')[0]
                            answer = answer[:len(answer) - len(delta)] + part_before
                            print("\nAborting completion due to max paragraphs limit " + str(max_paragraphs) + ".")
                            await MODEL.abort(request_id)
                            break

            if max_characters > 0 and len(answer) >= max_characters:
                if '\n' in delta:
                    part_before = delta.split('\n')[0]
                    answer = answer[:len(answer) - len(delta)] + part_before
                    print("\nAborting completion due to max characters limit " + str(max_characters) + ".")
                    await MODEL.abort(request_id)
                    break

            if max_safety_characters > 0 and len(answer) >= max_safety_characters:
                print("\nAborting completion due to max safety characters limit " + str(max_safety_characters) + ".")
                await MODEL.abort(request_id)
                break

            if regex_stop_after:
                for stop_re in regex_stop_after:
                    if stop_re.search(answer):
                        print(f"\nAborting completion due to stopAfter trigger: {stop_re.pattern}")
                        await MODEL.abort(request_id)
                        break

            if data.get("repetitionBuster"):
                rep = pattern_repetition_checker(answer, 5, 300)
                if rep and rep["amount"] >= 3:
                    print(f"\nAborting completion due to repetition detected: {rep}")
                    await MODEL.abort(request_id)
                    break

            if data.get("aggressiveListRepetitionBuster") and aggressive_list_repetition_checker(answer):
                print("\nAborting completion due to aggressive list repetition detected")
                await MODEL.abort(request_id)
                break

        answer = _strip_trailing_newlines(answer)
        
        yield {"answer": answer, "done": True}
    except Exception as e:
        print(str(e))
        yield {"error": str(e)}


async def generate_completion(
    data: dict,
):
    """
    Mirrors generateCompletion(). Uses the standard config section.
    Because vLLM returns the full generated text at once (not token-by-token
    in the offline API), we call on_token with the post-processed text and
    then on_done.
    """
    global ANALYSIS_TEXT

    if MODEL is None:
        raise RuntimeError("Model not loaded")
    if CONFIG is None:
        raise RuntimeError("Config not loaded")

    # ── Validate inputs ──
    if not isinstance(data.get("messages"), list):
        raise ValueError("Invalid messages format")
    if not isinstance(data.get("stopAt"), list):
        raise ValueError("Invalid stopAt format")
    if any(not isinstance(s, str) for s in data["stopAt"]):
        raise ValueError("Invalid stopAt format, all stops must be strings")
    if not isinstance(data.get("maxParagraphs"), (int, float)) or data["maxParagraphs"] < 0:
        raise ValueError("Invalid maxParagraphs format")
    if not isinstance(data.get("maxCharacters"), (int, float)) or data["maxCharacters"] < 0:
        raise ValueError("Invalid maxCharacters format")
    if not isinstance(data.get("maxSafetyCharacters"), (int, float)) or data["maxSafetyCharacters"] < 0:
        raise ValueError("Invalid maxSafetyCharacters format")
    if data.get("startCountingFromToken") is not None and not isinstance(data["startCountingFromToken"], str):
        raise ValueError("Invalid startCountingFromToken format")
    if data.get("trail") is not None and not isinstance(data["trail"], str):
        raise ValueError("Invalid trail format")
    if not isinstance(data.get("stopAfter"), list):
        raise ValueError("Invalid stopAfter format")

    for msg in data["messages"]:
        if not isinstance(msg.get("content"), str):
            raise ValueError("Invalid message content")
        if not isinstance(msg.get("role"), str):
            raise ValueError("Invalid message role")
        if msg["role"] not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid message role: {msg['role']}")

    # Clear previous analysis
    ANALYSIS_TEXT = None

    prompt = _format_chat_prompt(CONFIG, data["messages"], data.get("trail"))
    stop_tokens = _build_stop_tokens(CONFIG, data.get("stopAt"))
    sampling = _make_sampling_params(CONFIG["standard"], stop_tokens)

    max_paragraphs = int(data["maxParagraphs"])
    max_characters = int(data["maxCharacters"])
    max_safety_characters = int(data["maxSafetyCharacters"])
    if max_paragraphs:
        print("Max paragraphs limit set to: " + str(max_paragraphs))
    if max_characters:
        print("Max characters limit set to: " + str(max_characters))
    if max_safety_characters:
        print("Max safety characters limit set to: " + str(max_safety_characters))

    if DEBUG:
        print("Generation config: " + str(sampling))
        print("Prompt: " + str(prompt))

    regex_stop_after = [
        re.compile(rf"(^|[.,;])\s*{escape_regexp(s)}\s*([.,;]|$)", re.IGNORECASE)
        for s in data["stopAfter"]
    ]

    start_counting_from = data.get("startCountingFromToken")

    try:
        request_id = _next_request_id()
        yield {"request_id": request_id}
        generation = MODEL.generate(prompt, sampling_params=sampling, request_id=request_id)

        accumulated_text = ""
        accumulated_counting = ""
        has_begun_counting = start_counting_from is None
        prev_len = 0

        async for output in generation:
            if (output.request_id != request_id):
                continue  # Ignore outputs from other requests

            new_text = output.outputs[0].text
            delta = new_text[prev_len:]
            prev_len = len(new_text)

            if not delta:
                continue

            accumulated_text += delta

            if DEBUG:
                sys.stdout.write(delta + "\u00a7")
                sys.stdout.flush()

            if not has_begun_counting and start_counting_from and start_counting_from in accumulated_text:
                has_begun_counting = True

            if has_begun_counting:
                accumulated_counting += delta

                # ── Mid-stream paragraph limit ──
            if max_paragraphs > 0 and has_begun_counting:
                para_count = 0
                for i in range(len(accumulated_counting) - 1):
                    if accumulated_counting[i] == '\n' and accumulated_counting[i + 1] == '\n':
                        para_count += 1
                        if para_count >= max_paragraphs:
                            part_before = delta.split('\n')[0]
                            if part_before:
                                yield {"token": part_before}
                            print("\nAborting completion due to max paragraphs limit.")
                            await MODEL.abort(request_id)
                            break

            # ── Mid-stream character limit ──
            if max_characters > 0 and len(accumulated_text) >= max_characters:
                if '\n' in delta:
                    part_before = delta.split('\n')[0]
                    if part_before:
                        yield {"token": part_before}
                    print("\nAborting completion due to max characters limit.")
                    await MODEL.abort(request_id)
                    break

            yield {"token": delta}

            if max_safety_characters > 0 and len(accumulated_text) >= max_safety_characters:
                print("\nAborting completion due to max safety characters limit.")
                await MODEL.abort(request_id)
                break

            # ── Mid-stream stopAfter ──
            if regex_stop_after and has_begun_counting:
                for stop_re in regex_stop_after:
                    if stop_re.search(accumulated_counting):
                        print(f"\nAborting completion due to stopAfter trigger: {stop_re.pattern}")
                        await MODEL.abort(request_id)
                        break

        yield {"done": True}
    except Exception as e:
        print(str(e))
        yield {"error": str(e)}


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 1:
        print("Please provide a config path as the first argument.", file=sys.stderr)
        sys.exit(1)

    print("DEBUG mode:", DEBUG)
    load_config(argv[0])