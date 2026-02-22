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
from runpod import RunPodLogger

log = RunPodLogger()

# ── Module-level state (mirrors the JS globals) ──────────────────────────

MODEL: Optional[Any] = None
MODEL_PATH: str = ""
_REQUEST_COUNTER: int = 0
CONFIG: Optional[dict] = None
CONFIG_PATH: str = ""
ANALYSIS_TEXT: Optional[str] = None
DEBUG: bool = os.environ.get("DEBUG", "") == "1"


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
    from vllm import EngineArgs, LLMEngine

    global MODEL, MODEL_PATH

    log.info(f"Loading model: {model_path}")
    if MODEL_PATH == model_path and MODEL is not None:
        log.info("Model already loaded")
        return

    if MODEL is not None:
        log.info("Unloading previous model")
        MODEL = None
        MODEL_PATH = ""

    engine_kwargs: dict[str, Any] = dict(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        dtype="auto",
        enforce_eager=enforce_eager,
    )

    if tokenizer_path:
        log.info(f"Using pre-saved tokenizer: {tokenizer_path}")
        engine_kwargs["tokenizer"] = tokenizer_path

    engine_args = EngineArgs(**engine_kwargs)
    MODEL = LLMEngine.from_engine_args(engine_args)
    MODEL_PATH = model_path
    log.info("Model loaded successfully")

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
        totalAmount = int(expectedFilename.split("-of-")[-1])
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
            download_one_file(chunk_url, chunk_dest_path)
    else:
        download_one_file(url, dest_path)

    log.info(f"Model downloaded to: {dest_path}")
    return dest_path
        

def save_tokenizer(model_path: str, tokenizer_path: str) -> None:
    from transformers import AutoTokenizer
    
    # check if tokenizer_path already exists
    if os.path.exists(tokenizer_path):
        log.info(f"Tokenizer path already exists: {tokenizer_path}")
        return

    log.info(f"Loading tokenizer from GGUF: {model_path}")
    log.info("This will be slow...")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.dirname(model_path),
        gguf_file=os.path.basename(model_path),
    )

    tokenizer.save_pretrained(tokenizer_path)
    log.info(f"Tokenizer saved to: {tokenizer_path}")

def load_config(config_path: str) -> None:
    global CONFIG, CONFIG_PATH

    log.info(f"Loading config: {config_path}")
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
        log.info(CONFIG)
        raise ValueError("Invalid config file, expected a JSON object at the top level")

    if "standard" not in CONFIG or "analyze" not in CONFIG:
        log.info(CONFIG)
        raise ValueError("Invalid config file, missing standard or analyze sections")

    check_config_validity(CONFIG["standard"])
    check_config_validity(CONFIG["analyze"])
    log.info("Config loaded successfully")

    if (CONFIG.get("modelUrl") is not None) and not isinstance(CONFIG["modelUrl"], str):
        raise ValueError("Invalid config: modelUrl must be a string if provided")

    model_url = CONFIG.get("modelUrl", None)
    if not model_url:
        if (CONFIG.get("modelPath") is None) or not isinstance(CONFIG["modelPath"], str):
            raise ValueError("Invalid config: modelPath must be a string")
    else:
        # we don't allow modelPath or tokenizerPath when modelUrl is provided, to avoid confusion about where the model is coming from
        if CONFIG.get("modelPath") is not None:
            raise ValueError("Invalid config: modelPath should not be provided when modelUrl is used")
        
    if (CONFIG.get("tokenizerPath") is not None) and not isinstance(CONFIG["tokenizerPath"], str):
        raise ValueError("Invalid config: tokenizerPath must be a string if provided")
    
    if (CONFIG.get("mode") is not None) and CONFIG["mode"] not in ("mistral", "llama3"):
        raise ValueError("Invalid config: mode must be 'mistral' or 'llama3' if provided")
    
    if not isinstance(CONFIG.get("enforceEager"), bool):
        raise ValueError("Invalid config: enforceEager must be a boolean")
    
    if CONFIG.get("modelUrl") is not None:
        CONFIG["modelPath"] = download_model_from_url(CONFIG["modelUrl"])

    if CONFIG.get("tokenizerPath") is not None:
        save_tokenizer(CONFIG["modelPath"], CONFIG["tokenizerPath"])

    if MODEL_PATH != CONFIG["modelPath"]:
        base_dir = os.path.dirname(config_path)
        model_full_path = os.path.normpath(os.path.join(base_dir, CONFIG["modelPath"]))

        if not os.path.exists(model_full_path):
            model_full_path = CONFIG["modelPath"]  # try as absolute path
            if not os.path.exists(model_full_path):
                raise FileNotFoundError(f"Model file not found at: {model_full_path}")

        log.info(f"Resolved model path: {model_full_path}")

        tokenizer_path = None
        if CONFIG.get("tokenizerPath"):
            tokenizer_path = os.path.normpath(os.path.join(base_dir, CONFIG["tokenizerPath"]))
            if not os.path.exists(tokenizer_path):
                tokenizer_path = CONFIG["tokenizerPath"]  # try as absolute path
                if not os.path.exists(tokenizer_path):
                    raise FileNotFoundError(f"Tokenizer path not found at: {tokenizer_path}")
        log.info(f"Resolved tokenizer path: {tokenizer_path}")

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
                log.info("WARNING: Grammar constraints requested but not supported by this vLLM version. Ignoring grammar.")

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

def prepare_analysis(
    data: dict,
    on_done: Callable[[], None],
    on_error: Callable[[Exception], None],
) -> None:
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
            log.info("Prepared analysis text:", ANALYSIS_TEXT)
        on_done()
    except Exception as e:
        on_error(e)


def run_question(
    data: dict,
    on_request_id: Callable[[str], None],
    on_answer: Callable[[str], None],
    on_error: Callable[[Exception], None],
) -> None:
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
        log.info("Generation config:", sampling)
        log.info("Prompt:", prompt)
        log.info("Using grammar:", data.get("grammar"))

    max_paragraphs = int(data["maxParagraphs"])
    max_characters = int(data["maxCharacters"])
    if max_paragraphs:
        log.info("Max paragraphs limit set to:", max_paragraphs)
    if max_characters:
        log.info("Max characters limit set to:", max_characters)

    try:
        request_id = _next_request_id()
        on_request_id(request_id)
        MODEL.add_request(request_id, prompt, sampling)

        answer = ""
        prev_len = 0
        while MODEL.has_unfinished_requests():
            step_outputs = MODEL.step()

            if (len(step_outputs) == 0):
                log.info("No outputs from this step, but request is not finished. Continuing...")
                continue

            found_its_step = False
            stop_process = False

            for output in step_outputs:
                if (output.request_id != request_id):
                    log.info("SKIPPED OUTPUT from other request:", output.request_id)
                    continue  # Ignore outputs from other requests

                found_its_step = True
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
                                log.info("\nAborting completion due to max paragraphs limit.")
                                MODEL.abort_request(request_id)
                                stop_process = True
                                break

                if max_characters > 0 and len(answer) >= max_characters:
                    if '\n' in delta:
                        part_before = delta.split('\n')[0]
                        answer = answer[:len(answer) - len(delta)] + part_before
                        log.info("\nAborting completion due to max characters limit.")
                        MODEL.abort_request(request_id)
                        stop_process = True
                        break

                if regex_stop_after:
                    should_stop = False
                    for stop_re in regex_stop_after:
                        if stop_re.search(answer):
                            log.info(f"\nAborting completion due to stopAfter trigger: {stop_re.pattern}")
                            MODEL.abort_request(request_id)
                            should_stop = True
                            break
                    if should_stop:
                        stop_process = True
                        break

                if data.get("repetitionBuster"):
                    rep = pattern_repetition_checker(answer, 5, 300)
                    if rep and rep["amount"] >= 3:
                        log.info(f"\nAborting completion due to repetition detected: {rep}")
                        MODEL.abort_request(request_id)
                        stop_process = True
                        break

                if data.get("aggressiveListRepetitionBuster") and aggressive_list_repetition_checker(answer):
                    log.info("\nAborting completion due to aggressive list repetition detected")
                    MODEL.abort_request(request_id)
                    stop_process = True
                    break

            if not found_its_step or stop_process:
                break  # No more outputs for this request, exit loop

        answer = _strip_trailing_newlines(answer)
        
        on_answer(answer)
    except Exception as e:
        log.error(str(e))
        on_error(e)


def generate_completion(
    data: dict,
    on_request_id: Callable[[str], None],
    on_token: Callable[[str], None],
    on_done: Callable[[], None],
    on_error: Callable[[Exception], None],
) -> None:
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
    if max_paragraphs:
        log.info("Max paragraphs limit set to:", max_paragraphs)
    if max_characters:
        log.info("Max characters limit set to:", max_characters)

    if DEBUG:
        log.info("Generation config:", sampling)
        log.info("Prompt:", prompt)

    regex_stop_after = [
        re.compile(rf"(^|[.,;])\s*{escape_regexp(s)}\s*([.,;]|$)", re.IGNORECASE)
        for s in data["stopAfter"]
    ]

    start_counting_from = data.get("startCountingFromToken")

    try:
        request_id = _next_request_id()
        on_request_id(request_id)
        MODEL.add_request(request_id, prompt, sampling)

        accumulated_text = ""
        accumulated_counting = ""
        has_begun_counting = start_counting_from is None
        prev_len = 0

        while MODEL.has_unfinished_requests():
            step_outputs = MODEL.step()

            if (len(step_outputs) == 0):
                log.info("No outputs from this step, but request is not finished. Continuing...")
                continue

            stop_process = False
            found_its_step = False

            for output in step_outputs:
                if (output.request_id != request_id):
                    continue  # Ignore outputs from other requests

                found_its_step = True
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
                                    on_token(part_before)
                                log.info("\nAborting completion due to max paragraphs limit.")
                                MODEL.abort_request(request_id)
                                stop_process = True
                                break
                    else:
                        on_token(delta)
                        continue
                    break

                # ── Mid-stream character limit ──
                if max_characters > 0 and len(accumulated_text) >= max_characters:
                    if '\n' in delta:
                        part_before = delta.split('\n')[0]
                        if part_before:
                            on_token(part_before)
                        log.info("\nAborting completion due to max characters limit.")
                        MODEL.abort_request(request_id)
                        stop_process = True
                        break

                on_token(delta)

                # ── Mid-stream stopAfter ──
                if regex_stop_after and has_begun_counting:
                    should_stop = False
                    for stop_re in regex_stop_after:
                        if stop_re.search(accumulated_counting):
                            log.info(f"\nAborting completion due to stopAfter trigger: {stop_re.pattern}")
                            MODEL.abort_request(request_id)
                            should_stop = True
                            break
                    if should_stop:
                        stop_process = True
                        break

            if not found_its_step or stop_process:
                break  # No more outputs for this request, exit loop

        on_done()
    except Exception as e:
        log.error(str(e))
        on_error(e)


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 1:
        log.error("Please provide a config path as the first argument.", file=sys.stderr)
        sys.exit(1)

    log.info("DEBUG mode:", DEBUG)
    load_config(argv[0])