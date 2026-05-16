/**
 * USAGE:
 * 
 * node local-llama.js <path to config json>
 * 
 * Example:
 * Windows:
 * node .\local-llama.js .\testing\model.json
 * Unix/Linux/Mac:
 * node ./local-llama.js ./testing/model.json
 * 
 * Example (debug mode):
 * Windows:
 * $env:DEBUG=1; node .\local-llama.js .\testing\model.json
 * Unix/Linux/Mac:
 * DEBUG=1 node ./local-llama.js ./testing/model.json
 * 
 * Remember in Windows
 * Remove-Item Env:DEBUG
 * 
 * JSON File settings example
 * 
 * {
 *  // the path of the model relative to the json file
 *   "modelPath": "./model.json", *   // chat template: mistral | llama3 | chatml | gemma | phi | deepseek | alpaca *   "mode": "mistral",
 *   // standard generation used in roleplay contexts
 *   "standard": {
 *       // temperature base
 *       "temperature": 1.0,
 *       "maxTokens": 512,
 *       // dynamic temperature range, if given it will vary temperature between these values
 *       "dynamicTemperature": [0.8, 1.05],
 *       // minimum probability for dry run detection
 *       "minP": 0.025,
 *       // dry sampler settings
 *       "dry": {
 *           "multiplier": 0.8,
 *           "base": 1.74,
 *           "length": 5
 *       },
 *       // xtc sampler settings (should probably not use both dry and xtc at the same time)
 *   },
 *   "analyze": {
 *       // analysis generation settings
 *       "temperature": 0.4,
 *       "topP": 0.8,
 *       "topK": 40,
 *       "repeatPenalty": 1.1,
 *       "frequencyPenalty": 0.0,
 *       "presencePenalty": 0.0,
 *       "maxTokens": 512,
 *   }
 * }
 */

import fs from 'fs';
const { LlamaCompletion, getLlama } = await import('node-llama-cpp');
import path from 'path';

/**
 * Chat-template registry. Each mode describes how to format prompts and
 * which strings should act as stop triggers for the given model family.
 *
 * To add support for another model type, add a new entry to this object.
 *
 * Fields:
 *   - endToken: token returned by loadConfig() over the wire to indicate
 *               the end of an assistant turn for this template.
 *   - stopTokens: hard stop strings appended to the user-supplied stopAt.
 *   - chatBos: prefix prepended once at the start of a chat-style prompt.
 *   - formatChatMessage(role, content): serializes one chat message.
 *   - chatAssistantHeader: opens the trailing assistant turn for chat.
 *   - analysisPrefix(system, userTrail): opens the user turn for analysis,
 *               leaving it open so a question can be appended later.
 *   - analysisToQuestion(analysisText, question, trail): closes the analysis
 *               user turn, appends the question and opens the assistant turn.
 *
 * @type {Record<string, {
 *   endToken: string,
 *   stopTokens: string[],
 *   chatBos: string,
 *   formatChatMessage: (role: string, content: string) => string,
 *   chatAssistantHeader: string,
 *   analysisPrefix: (system: string, userTrail: string) => string,
 *   analysisToQuestion: (analysisText: string, question: string, trail: string | null) => string,
 * }>}
 */
export const MODES = {
    mistral: {
        endToken: "</s>",
        stopTokens: ["</s>", "[INST]"],
        chatBos: "<s>",
        formatChatMessage: (role, content) =>
            role === "system"
                ? `[SYSTEM_PROMPT] ${content}[/SYSTEM_PROMPT][INST]`
                : `\n\n${content}`,
        chatAssistantHeader: "[/INST]\n\n",
        analysisPrefix: (system, userTrail) =>
            `<s>[SYSTEM_PROMPT] ${system}[/SYSTEM_PROMPT][INST] ${userTrail}`,
        analysisToQuestion: (analysisText, question, trail) =>
            analysisText + "\n\n" + question + "\n[/INST]\n\n" + (trail || ""),
    },
    llama3: {
        endToken: "<|eot_id|>",
        stopTokens: ["<|eot_id|>", "<|start_header_id|>"],
        chatBos: "",
        formatChatMessage: (role, content) =>
            `<|start_header_id|>${role}<|end_header_id|>\n\n${content}<|eot_id>`,
        chatAssistantHeader: "\n<|start_header_id|>assistant<|end_header_id|>\n\n",
        analysisPrefix: (system, userTrail) =>
            `<|start_header_id|>system<|end_header_id|>\n\n${system}<|eot_id><|start_header_id|>user<|end_header_id|>\n\n${userTrail}`,
        analysisToQuestion: (analysisText, question, trail) =>
            analysisText + "\n" + question
            + `\n<|start_header_id|>assistant<|end_header_id|>\n\n`
            + (trail || ""),
    },
    // Qwen, Hermes, Yi, generic ChatML
    chatml: {
        endToken: "<|im_end|>",
        stopTokens: ["<|im_end|>", "<|im_start|>"],
        chatBos: "",
        formatChatMessage: (role, content) =>
            `<|im_start|>${role}\n${content}<|im_end|>\n`,
        chatAssistantHeader: "<|im_start|>assistant\n",
        analysisPrefix: (system, userTrail) =>
            `<|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${userTrail}`,
        analysisToQuestion: (analysisText, question, trail) =>
            analysisText + "\n\n" + question
            + "<|im_end|>\n<|im_start|>assistant\n"
            + (trail || ""),
    },
    // Google Gemma / Gemma2 (no dedicated system role: merged into first user turn)
    gemma: {
        endToken: "<end_of_turn>",
        stopTokens: ["<end_of_turn>", "<start_of_turn>"],
        chatBos: "<bos>",
        formatChatMessage: (role, content) => {
            if (role === "system") {
                return `<start_of_turn>user\n${content}<end_of_turn>\n`;
            }
            const r = role === "assistant" ? "model" : "user";
            return `<start_of_turn>${r}\n${content}<end_of_turn>\n`;
        },
        chatAssistantHeader: "<start_of_turn>model\n",
        analysisPrefix: (system, userTrail) =>
            `<bos><start_of_turn>user\n${system}\n\n${userTrail}`,
        analysisToQuestion: (analysisText, question, trail) =>
            analysisText + "\n\n" + question
            + "<end_of_turn>\n<start_of_turn>model\n"
            + (trail || ""),
    },
    // Microsoft Phi-3 / Phi-4
    phi: {
        endToken: "<|end|>",
        stopTokens: ["<|end|>", "<|user|>", "<|system|>"],
        chatBos: "",
        formatChatMessage: (role, content) =>
            `<|${role}|>\n${content}<|end|>\n`,
        chatAssistantHeader: "<|assistant|>\n",
        analysisPrefix: (system, userTrail) =>
            `<|system|>\n${system}<|end|>\n<|user|>\n${userTrail}`,
        analysisToQuestion: (analysisText, question, trail) =>
            analysisText + "\n\n" + question
            + "<|end|>\n<|assistant|>\n"
            + (trail || ""),
    },
    // DeepSeek V2 / V3 / R1 style
    deepseek: {
        endToken: "<｜end▁of▁sentence｜>",
        stopTokens: ["<｜end▁of▁sentence｜>", "<｜User｜>"],
        chatBos: "<｜begin▁of▁sentence｜>",
        formatChatMessage: (role, content) => {
            if (role === "system") return content;
            if (role === "user") return `<｜User｜>${content}`;
            return `<｜Assistant｜>${content}<｜end▁of▁sentence｜>`;
        },
        chatAssistantHeader: "<｜Assistant｜>",
        analysisPrefix: (system, userTrail) =>
            `<｜begin▁of▁sentence｜>${system}<｜User｜>${userTrail}`,
        analysisToQuestion: (analysisText, question, trail) =>
            analysisText + "\n\n" + question
            + "<｜Assistant｜>"
            + (trail || ""),
    },
    // Classic Alpaca instruction format
    alpaca: {
        endToken: "</s>",
        stopTokens: ["</s>", "### Instruction:"],
        chatBos: "",
        formatChatMessage: (role, content) => {
            if (role === "system") return `${content}\n\n`;
            if (role === "user") return `### Instruction:\n${content}\n\n`;
            return `### Response:\n${content}\n\n`;
        },
        chatAssistantHeader: "### Response:\n",
        analysisPrefix: (system, userTrail) =>
            `${system}\n\n### Instruction:\n${userTrail}`,
        analysisToQuestion: (analysisText, question, trail) =>
            analysisText + "\n\n" + question
            + "\n\n### Response:\n"
            + (trail || ""),
    },
};

const DEFAULT_MODE = "llama3";

/**
 * @param {string | undefined} mode
 * @returns {string}
 */
function resolveModeName(mode) {
    return mode || DEFAULT_MODE;
}

/**
 * @param {string | undefined} mode
 */
function getMode(mode) {
    const name = resolveModeName(mode);
    const m = MODES[name];
    if (!m) {
        throw new Error(`Unsupported mode '${name}'. Supported modes: ${Object.keys(MODES).join(", ")}`);
    }
    return m;
}

/**
 * @type {import('node-llama-cpp').LlamaModel}
 */
export let MODEL = /** @type {any} */ (null);
let LLAMA = await getLlama();
export let MODEL_PATH = "";

/**
 * @param {string} string 
 * @returns 
 */
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

/**
 * @type {{
 *    modelPath: string;
 *    mode: keyof typeof MODES;
 *    standard: {temperature: number; temperatureRange?: [number, number]; topP?: number; minP?: number; repeatPenalty?: number; frequencyPenalty?: number; presencePenalty?: number; maxTokens: number;},
 *    analyze: {temperature: number; temperatureRange?: [number, number]; topP?: number; minP?: number; repeatPenalty?: number; frequencyPenalty?: number; presencePenalty?: number; maxTokens: number;},
 * }}
 */
let CONFIG = /** @type {any} */ (null);
let CONFIG_PATH = "";

/**
 * @param {*} config 
 */
function checkConfigValidity(config) {
    // implement any additional checks if needed
    if (typeof config.maxTokens !== "number") {
        throw new Error("Invalid config: maxTokens must be a number");
    }
    if (typeof config.temperature !== "number") {
        throw new Error("Invalid config: temperature must be a number");
    }
    if (config.temperatureRange !== undefined) {
        if (!Array.isArray(config.temperatureRange) || config.temperatureRange.length !== 2 ||
            typeof config.temperatureRange[0] !== "number" || typeof config.temperatureRange[1] !== "number") {
            throw new Error("Invalid config: temperatureRange must be an array of two numbers");
        }
    }
    if (config.topP !== undefined && typeof config.topP !== "number") {
        throw new Error("Invalid config: topP must be a number");
    }
    if (config.repeatPenalty !== undefined && typeof config.repeatPenalty !== "number") {
        throw new Error("Invalid config: repeatPenalty must be a number");
    }
    if (config.frequencyPenalty !== undefined && typeof config.frequencyPenalty !== "number") {
        throw new Error("Invalid config: frequencyPenalty must be a number");
    }
    if (config.presencePenalty !== undefined && typeof config.presencePenalty !== "number") {
        throw new Error("Invalid config: presencePenalty must be a number");
    }
    if (config.minP !== undefined && typeof config.minP !== "number") {
        throw new Error("Invalid config: minP must be a number");
    }
    if (config.dry !== undefined) {
        if (typeof config.dry !== "object") {
            throw new Error("Invalid config: dry must be an object");
        }
        if (typeof config.dry.multiplier !== "number") {
            throw new Error("Invalid config: dry.multiplier must be a number");
        }
        if (typeof config.dry.base !== "number") {
            throw new Error("Invalid config: dry.base must be a number");
        }
        if (typeof config.dry.length !== "number") {
            throw new Error("Invalid config: dry.length must be a number");
        }
    }
    if (config.xtc !== undefined) {
        if (typeof config.xtc !== "object") {
            throw new Error("Invalid config: xtc must be an object");
        }
        // TODO: add xtc specific checks
    }
}

/**
 * @type {AbortController | null}
 */
export let CONTROLLER = null;

/**
 * @param {string} configPath
 * @return {Promise<{endToken: string}>} The end token to use for the current model, based on the config mode
 */
export async function loadConfig(configPath) {
    console.log("Loading config:", configPath);

    const configContent = await fs.promises.readFile(configPath, 'utf-8');
    CONFIG = JSON.parse(configContent);
    CONFIG_PATH = configPath;

    // check that everything lines up
    if (!CONFIG.standard || !CONFIG.analyze) {
        console.log(CONFIG);
        throw new Error("Invalid config file, missing standard or analyze sections");
    }
    checkConfigValidity(CONFIG.standard);
    checkConfigValidity(CONFIG.analyze);

    console.log("Config loaded successfully");

    if (CONFIG.mode !== undefined && !MODES[CONFIG.mode]) {
        throw new Error(
            `Invalid config: mode must be one of ${Object.keys(MODES).join(", ")} if provided`
        );
    }

    if (MODEL_PATH !== CONFIG.modelPath) {
        // use relative path from config file
        const baseDir = path.dirname(configPath);
        const modelFullPath = path.resolve(baseDir, CONFIG.modelPath);
        await loadModel(modelFullPath);
    }

    return { endToken: getMode(CONFIG.mode).endToken };
}

/**
 * @param {string} model 
 * @returns 
 */
async function loadModel(model) {
    console.log("Loading model:", model);
    if (MODEL_PATH === model && MODEL !== null) {
        console.log('Model already loaded');
        return;
    }

    if (MODEL !== null) {
        console.log('Unloading previous model');
        await MODEL.dispose();
        MODEL = /** @type {any} */ (null);
        MODEL_PATH = "";
    }

    console.log('GPU Support:', LLAMA.gpu || 'Unknown');

    const LLAMA_MODEL = await LLAMA.loadModel({
        modelPath: model,
        gpuLayers: "auto",
        defaultContextFlashAttention: true,
    });
    MODEL = LLAMA_MODEL
    MODEL_PATH = model;

    // Create a simple HTTP server that takes a prompt and returns a response
    console.log('Model loaded successfully');
}

const DEBUG = process.env.DEBUG === "1";

console.log("DEBUG mode:", DEBUG);

/**
 * @type {import('node-llama-cpp').Token[] | null}
 */
//let ANALYSIS_TOKENS = null;
/**
 * @type {string | null}
 */
let ANALYSIS_TEXT = null;

/**
 * @param {number} minTemp 
 * @param {number} maxTemp 
 */
function getDynamicTemperature(minTemp, maxTemp) {
    return Math.random() * (maxTemp - minTemp) + minTemp;
}

/**
 * 
 * @param {{system: string, userTrail: string}} data 
 * @param {() => void} onDone 
 * @param {(error: Error) => void} onError 
 */
export async function prepareAnalysis(data, onDone, onError) {
    if (!MODEL) {
        throw new Error("Model not loaded");
    }
    if (!CONFIG) {
        throw new Error("Config not loaded");
    }
    if (!data.system || typeof data.system !== "string") {
        throw new Error("Invalid system format or missing");
    }
    if (typeof data.userTrail !== "string") {
        throw new Error("Invalid userTrail format");
    }
    try {
        //const context = await MODEL.createContext();
        //const contextSequence = context.getSequence();
        //contextSequence.eraseContextTokenRanges

        // TODO optimize this, for now just retokenize every time
        ANALYSIS_TEXT = getMode(CONFIG.mode).analysisPrefix(data.system, data.userTrail);

        if (DEBUG) {
            console.log("Prepared analysis text:", ANALYSIS_TEXT);
        }
        onDone();
    } catch (e) {
        // @ts-ignore
        onError(e);
    }
}

/**
 * 
 * @param {{
 * question: string;
 * stopAt: Array<string>;
 * stopAfter: Array<string>;
 * maxParagraphs: number;
 * maxCharacters: number;
 * trail: string | null;
 * grammar: string | null;
 * }} data
 * @param {(v: string) => void} onAnswer 
 * @param {(err: Error) => void} onError 
 */
export async function runQuestion(data, onAnswer, onError) {
    if (CONTROLLER) {
        throw new Error("Another generation is already in progress");
    }
    if (!MODEL) {
        throw new Error("Model not loaded");
    }
    if (!CONFIG) {
        throw new Error("Config not loaded");
    }
    if (!ANALYSIS_TEXT) {
        throw new Error("Analysis not prepared");
    }

    if (!data.question || typeof data.question !== "string") {
        throw new Error("Invalid question format");
    }

    if (!Array.isArray(data.stopAt)) {
        throw new Error("Invalid stopAt format");
    }

    if (!Array.isArray(data.stopAfter)) {
        throw new Error("Invalid stopAfter format");
    }

    if (typeof data.maxParagraphs !== "number" || isNaN(data.maxParagraphs) || data.maxParagraphs < 0) {
        throw new Error("Invalid maxParagraphs format");
    }

    if (typeof data.maxCharacters !== "number" || isNaN(data.maxCharacters) || data.maxCharacters < 0) {
        throw new Error("Invalid maxCharacters format");
    }

    if (data.trail !== null && typeof data.trail !== "string") {
        throw new Error("Invalid trail format");
    }

    if (data.grammar !== null && typeof data.grammar !== "string") {
        throw new Error("Invalid grammar format");
    }

    const regexStopAfter = data.stopAfter.map(s => new RegExp(`(^|[.,;])\\s*${escapeRegExp(s)}\\s*([.,;]|$)`, 'i'));

    const modeImpl = getMode(CONFIG.mode);
    let prompt = modeImpl.analysisToQuestion(ANALYSIS_TEXT, data.question, data.trail);
    let context = null
    let completion = null;
    let answer = "";
    CONTROLLER = new AbortController();
    try {
        const grammar = data.grammar ? await LLAMA.createGrammar({
            grammar: data.grammar,
        }) : undefined;
        // Create context and completion for raw text
        context = await MODEL.createContext();
        completion = new LlamaCompletion({
            contextSequence: context.getSequence(),
        });

        const CONFIG_TO_USE = data.gear === "cardtype-gen" ? CONFIG.standard : CONFIG.analyze;

        const basicConfig = {
            temperature: CONFIG_TO_USE.temperature,
            topP: CONFIG_TO_USE.topP,
            minP: CONFIG_TO_USE.minP,
            repeatPenalty: {
                penalty: CONFIG_TO_USE.repeatPenalty,
                frequencyPenalty: CONFIG_TO_USE.frequencyPenalty,
                presencePenalty: CONFIG_TO_USE.presencePenalty,
            },
            customStopTriggers: modeImpl.stopTokens.concat(data.stopAt || []),
            maxTokens: CONFIG_TO_USE.maxTokens || 512,
        }
        if (CONFIG_TO_USE.temperatureRange) {
            basicConfig.temperature = getDynamicTemperature(CONFIG_TO_USE.temperatureRange[0], CONFIG_TO_USE.temperatureRange[1]);
        }
        if (typeof data.maxParagraphs === "number" && DEBUG) {
            console.log("Max paragraphs limit set to:", data.maxParagraphs);
        }
        if (typeof data.maxCharacters === "number" && DEBUG) {
            console.log("Max characters limit set to:", data.maxCharacters);
        }
        // TODO add XTC and dry sampling options from config

        let accumulatedText = "";

        if (DEBUG) {
            console.log("Generation config:", basicConfig);
            console.log("Prompt:", prompt);
            console.log("Using grammar:", data.grammar);
        }

        await completion.generateCompletion(prompt, {
            ...basicConfig,
            signal: CONTROLLER.signal,
            stopOnAbortSignal: true,
            grammar,
            onTextChunk(textSrc) {
                try {
                    const text = textSrc;
                    accumulatedText += text;

                    if (DEBUG) {
                        // use this weird character to denote token boundaries
                        process.stdout.write(text + "§");
                    }

                    if (typeof data.maxParagraphs === "number" && data.maxParagraphs > 0) {
                        // For the non prototype this can be optimized better but for now it's fine
                        // count paragraphs
                        let paragraphCount = 0;

                        for (let i = 0; i < accumulatedText.length; i++) {
                            if (accumulatedText[i] === '\n' && accumulatedText[i + 1] === '\n') {
                                paragraphCount += 1;
                            }
                            //console.log("Current paragraph count:", paragraphCount);

                            // this should hit exactly at paragraph end
                            if (paragraphCount >= data.maxParagraphs) {
                                //console.log("Max paragraphs reached:", paragraphCount, "stopping completion early.");
                                // I think newlines are whole tokens, but just in case the text contains some text too
                                const potentialPartBeforeNew = text.split("\n")[0]
                                if (potentialPartBeforeNew.length > 0) {
                                    answer += potentialPartBeforeNew;
                                }
                                console.log("\nAborting completion due to max paragraphs limit.");
                                CONTROLLER?.abort();
                                CONTROLLER = null;
                                return;
                            }
                        }
                    }
                    if (typeof data.maxCharacters === "number" && data.maxCharacters > 0) {
                        const characterCount = accumulatedText.length;

                        //console.log("Current character count:", characterCount);

                        if (characterCount >= data.maxCharacters) {
                            //console.log("Trying to abort but no paragraph end found yet.");
                            // let's find if our text is finally finishing a paragraph
                            if (text.indexOf('\n') !== -1) {
                                //console.log("Max characters reached:", characterCount, "stopping completion at this paragraph end.");
                                const potentialPartBeforeNew = text.split("\n")[0]
                                if (potentialPartBeforeNew.length > 0) {
                                    answer += potentialPartBeforeNew;
                                }
                                console.log("\nAborting completion due to max characters limit.");
                                CONTROLLER?.abort();
                                CONTROLLER = null;
                                return;
                            }
                        }
                    }

                    answer += text;

                    if (regexStopAfter.length > 0) {
                        for (const stopRegex of regexStopAfter) {
                            if (stopRegex.test(answer)) {
                                console.log("\nAborting completion due to stopAfter trigger matched:", stopRegex);
                                CONTROLLER?.abort();
                                CONTROLLER = null;
                                return;
                            }
                        }
                    }
                } catch (e) {
                    // @ts-ignore
                    console.log("\nError in onToken callback:", e.message);
                    throw e;
                }
            }
        });
    } catch (e) {
        console.log("");
        // @ts-ignore
        console.log(e.message);
        // @ts-ignore
        onError(e);
    }

    if (context) {
        await context.dispose();
        context = null;
    }

    console.log("");

    // For the love of god stop adding newlines at the end of the answer
    while (answer[answer.length - 1] === '\n') {
        answer = answer.slice(0, -1);
    }

    onAnswer(answer);
    CONTROLLER = null;
}


/**
 * 
 * @param {{messages: Array<{role: string, content: string}>, stopAt: Array<string>, stopAfter: Array<string>, maxParagraphs: number, maxCharacters: number, trail: string | null, gear: string}} data 
 * @param {(text: string) => void} onToken 
 * @param {() => void} onDone 
 * @param {(error: Error) => void} onError 
 */
export async function generateCompletion(data, onToken, onDone, onError) {
    if (CONTROLLER) {
        throw new Error("Another generation is already in progress");
    }

    if (!MODEL) {
        throw new Error("Model not loaded");
    }

    if (!CONFIG) {
        throw new Error("Config not loaded");
    }

    if (!Array.isArray(data.messages)) {
        throw new Error("Invalid messages format");
    }

    if (!Array.isArray(data.stopAt)) {
        throw new Error("Invalid stopAt format");
    } else if (data.stopAt.some(s => typeof s !== "string")) {
        throw new Error("Invalid stopAt format, all stops must be strings");
    }

    if (typeof data.maxParagraphs !== "number" || isNaN(data.maxParagraphs) || data.maxParagraphs < 0) {
        throw new Error("Invalid maxParagraphs format");
    }

    if (typeof data.maxCharacters !== "number" || isNaN(data.maxCharacters) || data.maxCharacters < 0) {
        throw new Error("Invalid maxCharacters format");
    }

    if (data.trail !== null && typeof data.trail !== "string") {
        throw new Error("Invalid trail format");
    }

    if (!Array.isArray(data.stopAfter)) {
        throw new Error("Invalid stopAfter format");
    }

    if (data.grammar !== null && typeof data.grammar !== "string") {
        throw new Error("Invalid grammar format");
    }

    // clear previous analysis
    ANALYSIS_TEXT = null;

    const modeImpl = getMode(CONFIG.mode);
    let prompt = modeImpl.chatBos;
    for (const msg of data.messages) {
        if (typeof msg.content !== "string") {
            throw new Error("Invalid message content");
        } else if (typeof msg.role !== "string") {
            throw new Error("Invalid message role");
        } else if (!["user", "assistant", "system"].includes(msg.role)) {
            throw new Error("Invalid message role: " + msg.role);
        }
        prompt += modeImpl.formatChatMessage(msg.role, msg.content);
    }
    prompt += modeImpl.chatAssistantHeader;

    if (data.trail) {
        prompt += data.trail;
    }

    const grammar = data.grammar ? await LLAMA.createGrammar({
        grammar: data.grammar,
    }) : undefined;

    let context = null
    let completion = null;
    CONTROLLER = new AbortController();
    try {
        // Create context and completion for raw text
        context = await MODEL.createContext();
        completion = new LlamaCompletion({
            contextSequence: context.getSequence()
        });

        const basicConfig = {
            temperature: CONFIG.standard.temperature,
            topP: CONFIG.standard.topP,
            minP: CONFIG.standard.minP,
            repeatPenalty: {
                penalty: CONFIG.standard.repeatPenalty,
                frequencyPenalty: CONFIG.standard.frequencyPenalty,
                presencePenalty: CONFIG.standard.presencePenalty,
            },
            customStopTriggers: modeImpl.stopTokens.concat(data.stopAt || []),
            maxTokens: CONFIG.standard.maxTokens || 512,
        }
        if (CONFIG.standard.temperatureRange) {
            basicConfig.temperature = getDynamicTemperature(CONFIG.standard.temperatureRange[0], CONFIG.standard.temperatureRange[1]);
        }
        // TODO add XTC and dry sampling options from config
        if (typeof data.maxParagraphs === "number" && DEBUG) {
            console.log("Max paragraphs limit set to:", data.maxParagraphs);
        }
        if (typeof data.maxCharacters === "number" && DEBUG) {
            console.log("Max characters limit set to:", data.maxCharacters);
        }

        let hasBegunCounting = true
        let accumulatedText = "";
        let accumulatedTextForCounting = "";

        if (DEBUG) {
            console.log("Generation config:", basicConfig);
            console.log("Prompt:", prompt);
        }

        const regexStopAfter = data.stopAfter.map(s => new RegExp(`(^|[.,;])\\s*${escapeRegExp(s)}\\s*([.,;]|$)`, 'i'));

        await completion.generateCompletion(prompt, {
            ...basicConfig,
            signal: CONTROLLER.signal,
            stopOnAbortSignal: true,
            grammar,
            onTextChunk(textSrc) {
                try {
                    const text = textSrc;
                    accumulatedText += text;
                    if (DEBUG) {
                        // use this weird character to denote token boundaries
                        process.stdout.write(text + "§");
                    }
                    // Always accumulate text if we need to track limits
                    if (hasBegunCounting) {
                        accumulatedTextForCounting += text;
                    }

                    if (typeof data.maxParagraphs === "number" && data.maxParagraphs > 0) {
                        // For the non prototype this can be optimized better but for now it's fine
                        // count paragraphs
                        let paragraphCount = 0;

                        for (let i = 0; i < accumulatedTextForCounting.length; i++) {
                            if (accumulatedTextForCounting[i] === '\n' && accumulatedTextForCounting[i + 1] === '\n') {
                                paragraphCount += 1;
                            }
                            //console.log("Current paragraph count:", paragraphCount);

                            // this should hit exactly at paragraph end
                            if (paragraphCount >= data.maxParagraphs) {
                                //console.log("Max paragraphs reached:", paragraphCount, "stopping completion early.");
                                // I think newlines are whole tokens, but just in case the text contains some text too
                                const potentialPartBeforeNew = text.split("\n")[0]
                                if (potentialPartBeforeNew.length > 0) {
                                    onToken(potentialPartBeforeNew);
                                }
                                console.log("\nAborting completion due to max paragraphs limit.");
                                CONTROLLER?.abort();
                                CONTROLLER = null;
                                return;
                            }
                        }
                    }
                    if (typeof data.maxCharacters === "number" && data.maxCharacters > 0) {
                        const characterCount = accumulatedText.length;

                        //console.log("Current character count:", characterCount);

                        if (characterCount >= data.maxCharacters) {
                            //console.log("Trying to abort but no paragraph end found yet.");
                            // let's find if our text is finally finishing a paragraph
                            if (text.indexOf('\n') !== -1) {
                                //console.log("Max characters reached:", characterCount, "stopping completion at this paragraph end.");
                                const potentialPartBeforeNew = text.split("\n")[0]
                                if (potentialPartBeforeNew.length > 0) {
                                    onToken(potentialPartBeforeNew);
                                }
                                console.log("\nAborting completion due to max characters limit.");
                                CONTROLLER?.abort();
                                CONTROLLER = null;
                                return;
                            }
                        }
                    }

                    onToken(text);

                    if (regexStopAfter.length > 0) {
                        for (const stopRegex of regexStopAfter) {
                            if (stopRegex.test(accumulatedTextForCounting)) {
                                console.log("\nAborting completion due to stopAfter trigger matched:", stopRegex);
                                CONTROLLER?.abort();
                                CONTROLLER = null;
                                return;
                            }
                        }
                    }

                } catch (e) {
                    // @ts-ignore
                    console.log("\nError in onToken callback:", e.message);
                    throw e;
                }
            }
        });
    } catch (e) {
        console.log("");
        // @ts-ignore
        console.log(e.message);
        // @ts-ignore
        onError(e);
    }
    if (context) {
        await context.dispose();
        context = null;
    }
    console.log("");
    onDone();
    CONTROLLER = null;
}