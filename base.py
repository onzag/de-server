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
    // chat template: mistral | llama3 | chatml | gemma | phi | deepseek | alpaca
    "mode": "mistral",
    // standard generation used in roleplay contexts
    "enforceEager": false,
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

# ── Chat-template registry ────────────────────────────────────────────────
#
# Each entry describes how to format prompts and which strings act as stop
# triggers for the given model family. Add a new entry to support another
# model type. Mirrors the MODES table in base.js.
#
# Fields:
#   end_token:            wire-protocol end-of-turn marker returned by load_config.
#   stop_tokens:          hard stop strings appended to user-supplied stopAt.
#   chat_bos:             prefix prepended once at the start of a chat prompt.
#   format_chat_message:  callable(role, content) -> str, serializes one msg.
#   chat_assistant_header: opens the trailing assistant turn for chat.
#   analysis_prefix:      callable(system, user_trail) -> str, opens the user
#                         turn for analysis (left open for a question).
#   analysis_to_question: callable(analysis_text, question, trail) -> str,
#                         closes the analysis user turn, appends the question,
#                         opens the assistant turn.

def _mistral_chat_msg(role: str, content: str) -> str:
    if role == "system":
        return f"[SYSTEM_PROMPT] {content}[/SYSTEM_PROMPT][INST]"
    return f"\n\n{content}"


def _gemma_chat_msg(role: str, content: str) -> str:
    if role == "system":
        return f"<start_of_turn>user\n{content}<end_of_turn>\n"
    r = "model" if role == "assistant" else "user"
    return f"<start_of_turn>{r}\n{content}<end_of_turn>\n"


def _deepseek_chat_msg(role: str, content: str) -> str:
    if role == "system":
        return content
    if role == "user":
        return f"<｜User｜>{content}"
    return f"<｜Assistant｜>{content}<｜end▁of▁sentence｜>"


def _alpaca_chat_msg(role: str, content: str) -> str:
    if role == "system":
        return f"{content}\n\n"
    if role == "user":
        return f"### Instruction:\n{content}\n\n"
    return f"### Response:\n{content}\n\n"


MODES: dict[str, dict[str, Any]] = {
    "mistral": {
        "end_token": "</s>",
        "stop_tokens": ["</s>", "[INST]"],
        "chat_bos": "<s>",
        "format_chat_message": _mistral_chat_msg,
        "chat_assistant_header": "[/INST]\n\n",
        "analysis_prefix": lambda system, user_trail:
            f"<s>[SYSTEM_PROMPT] {system}[/SYSTEM_PROMPT][INST] {user_trail}",
        "analysis_to_question": lambda analysis_text, question, trail:
            analysis_text + "\n\n" + question + "\n[/INST]\n\n" + (trail or ""),
    },
    "llama3": {
        "end_token": "<|eot_id|>",
        "stop_tokens": ["<|eot_id|>", "<|start_header_id|>"],
        "chat_bos": "",
        "format_chat_message": lambda role, content:
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id>",
        "chat_assistant_header": "\n<|start_header_id|>assistant<|end_header_id|>\n\n",
        "analysis_prefix": lambda system, user_trail: (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_trail}"
        ),
        "analysis_to_question": lambda analysis_text, question, trail:
            analysis_text + "\n" + question
            + "\n<|start_header_id|>assistant<|end_header_id|>\n\n"
            + (trail or ""),
    },
    # Qwen, Hermes, Yi, generic ChatML
    "chatml": {
        "end_token": "<|im_end|>",
        "stop_tokens": ["<|im_end|>", "<|im_start|>"],
        "chat_bos": "",
        "format_chat_message": lambda role, content:
            f"<|im_start|>{role}\n{content}<|im_end|>\n",
        "chat_assistant_header": "<|im_start|>assistant\n",
        "analysis_prefix": lambda system, user_trail:
            f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_trail}",
        "analysis_to_question": lambda analysis_text, question, trail:
            analysis_text + "\n\n" + question
            + "<|im_end|>\n<|im_start|>assistant\n"
            + (trail or ""),
    },
    # Google Gemma / Gemma2 (no dedicated system role: merged into user turn)
    "gemma": {
        "end_token": "<end_of_turn>",
        "stop_tokens": ["<end_of_turn>", "<start_of_turn>"],
        "chat_bos": "<bos>",
        "format_chat_message": _gemma_chat_msg,
        "chat_assistant_header": "<start_of_turn>model\n",
        "analysis_prefix": lambda system, user_trail:
            f"<bos><start_of_turn>user\n{system}\n\n{user_trail}",
        "analysis_to_question": lambda analysis_text, question, trail:
            analysis_text + "\n\n" + question
            + "<end_of_turn>\n<start_of_turn>model\n"
            + (trail or ""),
    },
    # Microsoft Phi-3 / Phi-4
    "phi": {
        "end_token": "<|end|>",
        "stop_tokens": ["<|end|>", "<|user|>", "<|system|>"],
        "chat_bos": "",
        "format_chat_message": lambda role, content:
            f"<|{role}|>\n{content}<|end|>\n",
        "chat_assistant_header": "<|assistant|>\n",
        "analysis_prefix": lambda system, user_trail:
            f"<|system|>\n{system}<|end|>\n<|user|>\n{user_trail}",
        "analysis_to_question": lambda analysis_text, question, trail:
            analysis_text + "\n\n" + question
            + "<|end|>\n<|assistant|>\n"
            + (trail or ""),
    },
    # DeepSeek V2 / V3 / R1 style
    "deepseek": {
        "end_token": "<｜end▁of▁sentence｜>",
        "stop_tokens": ["<｜end▁of▁sentence｜>", "<｜User｜>"],
        "chat_bos": "<｜begin▁of▁sentence｜>",
        "format_chat_message": _deepseek_chat_msg,
        "chat_assistant_header": "<｜Assistant｜>",
        "analysis_prefix": lambda system, user_trail:
            f"<｜begin▁of▁sentence｜>{system}<｜User｜>{user_trail}",
        "analysis_to_question": lambda analysis_text, question, trail:
            analysis_text + "\n\n" + question
            + "<｜Assistant｜>"
            + (trail or ""),
    },
    # Classic Alpaca instruction format
    "alpaca": {
        "end_token": "</s>",
        "stop_tokens": ["</s>", "### Instruction:"],
        "chat_bos": "",
        "format_chat_message": _alpaca_chat_msg,
        "chat_assistant_header": "### Response:\n",
        "analysis_prefix": lambda system, user_trail:
            f"{system}\n\n### Instruction:\n{user_trail}",
        "analysis_to_question": lambda analysis_text, question, trail:
            analysis_text + "\n\n" + question
            + "\n\n### Response:\n"
            + (trail or ""),
    },
}

DEFAULT_MODE = "llama3"


def _get_mode(config: dict) -> dict:
    name = config.get("mode") or DEFAULT_MODE
    mode = MODES.get(name)
    if mode is None:
        raise ValueError(
            f"Unsupported mode '{name}'. Supported modes: {', '.join(MODES.keys())}"
        )
    return mode


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

def load_config(config_path: str, model_path_override: str | None = None) -> str:
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
    
    if (CONFIG.get("mode") is not None) and CONFIG["mode"] not in MODES:
        raise ValueError(
            f"Invalid config: mode must be one of {', '.join(MODES.keys())} if provided"
        )
    
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

    return {"end_token": _get_mode(CONFIG)["end_token"]}


# ── Sampling-params builder helpers ───────────────────────────────────────

def _build_stop_tokens(config: dict, extra_stops: list[str] | None = None) -> list[str]:
    """Return the list of stop strings depending on chat-template mode."""
    stops = list(_get_mode(config)["stop_tokens"])
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
    return _get_mode(config)["analysis_prefix"](system, user_trail)


def _format_question_prompt(config: dict, analysis_text: str, question: str, trail: str | None) -> str:
    return _get_mode(config)["analysis_to_question"](analysis_text, question, trail)


def _format_chat_prompt(config: dict, messages: list[dict], trail: str | None) -> str:
    mode = _get_mode(config)
    prompt = mode["chat_bos"]
    for msg in messages:
        prompt += mode["format_chat_message"](msg["role"], msg["content"])
    prompt += mode["chat_assistant_header"]
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
        raise ValueError("Invalid system format or missing")
    if not isinstance(data.get("userTrail"), str):
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

    regex_stop_after = [
        re.compile(rf"(^|[.,;])\s*{escape_regexp(s)}\s*([.,;]|$)", re.IGNORECASE)
        for s in data["stopAfter"]
    ]

    prompt = _format_question_prompt(CONFIG, ANALYSIS_TEXT, data["question"], data.get("trail"))

    stop_tokens = _build_stop_tokens(CONFIG, data.get("stopAt"))

    sampling_settings = CONFIG["analyze"]
    if (data.get("gear") == "cardtype-gen"):
        sampling_settings = CONFIG["standard"]

    sampling = _make_sampling_params(sampling_settings, stop_tokens, grammar=data.get("grammar"))

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
    if data.get("trail") is not None and not isinstance(data["trail"], str):
        raise ValueError("Invalid trail format")
    if not isinstance(data.get("stopAfter"), list):
        raise ValueError("Invalid stopAfter format")
    if data.get("grammar") is not None and not isinstance(data["grammar"], str):
        raise ValueError("Invalid grammar format")

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
    grammar = data.get("grammar")
    sampling = _make_sampling_params(CONFIG["standard"], stop_tokens, grammar)

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

    try:
        request_id = _next_request_id()
        yield {"request_id": request_id}
        generation = MODEL.generate(prompt, sampling_params=sampling, request_id=request_id)

        accumulated_text = ""
        accumulated_counting = ""
        has_begun_counting = True
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