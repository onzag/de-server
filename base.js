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
 *       "temperatureRange": [0.8, 1.05],
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

// @ts-ignore
import fs from 'fs';
const { LlamaCompletion, getLlama } = await import('node-llama-cpp');
// @ts-ignore
import path from 'path';

// @ts-ignore
let CONTEXT_WINDOW_SIZE = 2048 * 4; // 8k context default
// @ts-ignore
if (process.env.CONTEXT_WINDOW_SIZE) {
    // @ts-ignore
    const envSize = parseInt(process.env.CONTEXT_WINDOW_SIZE);
    if (!isNaN(envSize) && envSize > 0) {
        CONTEXT_WINDOW_SIZE = envSize;
    }
}
console.log("Context window size:", CONTEXT_WINDOW_SIZE);

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
            analysisText + "\n\n" + question + "\n<|eot_id>"
            + `\n<|start_header_id|>assistant<|end_header_id|>\n\n`
            + (trail || ""),
    },
    gemma4: {
        endToken: "<turn|>",
        stopTokens: ["<turn|>", "<channel|>", "<|turn|>", "<|channel|>", "<|turn>", "</turn>", "<|channel>"],
        chatBos: "",
        formatChatMessage: (role, content) => {
            if (role === "system") {
                return `<|turn>system\n${content}<turn|>\n`;
            }
            const r = role === "assistant" ? "model" : "user";
            return `<|turn>${r}\n${content}<turn|>\n`;
        },
        chatAssistantHeader: "<|turn>model\n",
        analysisPrefix: (system, userTrail) =>
            `<|turn>system\n${system}<turn|>\n<|turn>user\n${userTrail}`,
        analysisToQuestion: (analysisText, question, trail) =>
            analysisText + "\n" + question + "<turn|>\n<|turn>model\n" + (trail || ""),
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
 * @type {{model: import('node-llama-cpp').LlamaModel | null}}
 */
export let MODEL = {
    model: null,
};
let LLAMA = await getLlama();
/**
 * @type {{path: string | null}}
 */
export let MODEL_PATH = {
    path: null,
};

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
 *    supportedLanguages?: string[];
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
 * @type {{ctrl: AbortController | null}}
 */
export let CONTROLLER = {ctrl: null};

/**
 * @param {string} configPath
 * @return {Promise<{endToken: string, supportedLanguages: string[]}>} The end token to use for the current model, based on the config mode
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

    if (MODEL_PATH.path !== CONFIG.modelPath) {
        // use relative path from config file
        const baseDir = path.dirname(configPath);
        const modelFullPath = path.resolve(baseDir, CONFIG.modelPath);
        await loadModel(modelFullPath);
    }

    console.log("Supported languages:", CONFIG.supportedLanguages || []);

    return { endToken: getMode(CONFIG.mode).endToken, supportedLanguages: CONFIG.supportedLanguages || [] };
}

/**
 * @param {string} model 
 * @returns 
 */
export async function loadModel(model) {
    console.log("Loading model:", model);
    if (MODEL_PATH.path === model && MODEL.model !== null) {
        console.log('Model already loaded');
        return;
    }

    if (MODEL.model !== null) {
        console.log('Unloading previous model');
        await MODEL.model.dispose();
        MODEL.model = null;
        MODEL_PATH.path = null;
    }

    console.log('GPU Support:', LLAMA.gpu || 'Unknown');

    const LLAMA_MODEL = await LLAMA.loadModel({
        modelPath: model,
        gpuLayers: "auto",
        defaultContextFlashAttention: true,
    });
    MODEL.model = LLAMA_MODEL;
    MODEL_PATH.path = model;

    // Create a simple HTTP server that takes a prompt and returns a response
    console.log('Model loaded successfully');
}

// @ts-ignore
const DEBUG = process.env.DEBUG === "1";
// @ts-ignore
const DEBUG_MODE = process.env.DEBUG_MODE || "default";

console.log("DEBUG:", DEBUG);
console.log("DEBUG_MODE:", DEBUG_MODE);

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

        if (DEBUG && DEBUG_MODE === "default") {
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
 * maxCharactersCutOnDot: boolean;
 * maxSafetyCharacters: number;
 * trail: string | null;
 * grammar: string | null;
 * gear: string;
 * }} data
 * @param {(v: string) => void} onAnswer 
 * @param {(err: Error) => void} onError 
 */
export async function runQuestion(data, onAnswer, onError) {
    if (CONTROLLER.ctrl) {
        throw new Error("Another generation is already in progress");
    }
    if (!MODEL.model) {
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

    if (typeof data.maxCharactersCutOnDot !== "boolean") {
        throw new Error("Invalid maxCharactersCutOnDot format");
    }

    if (typeof data.maxSafetyCharacters !== "number" || isNaN(data.maxSafetyCharacters) || data.maxSafetyCharacters < 0) {
        throw new Error("Invalid maxSafetyCharacters format");
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
    CONTROLLER.ctrl = new AbortController();
    try {
        const grammar = data.grammar ? await LLAMA.createGrammar({
            grammar: data.grammar,
        }) : undefined;
        // Create context and completion for raw text
        context = await MODEL.model.createContext();
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
        if (typeof data.maxParagraphs === "number" && DEBUG && DEBUG_MODE === "default") {
            console.log("Max paragraphs limit set to:", data.maxParagraphs);
        }
        if (typeof data.maxCharacters === "number" && DEBUG && DEBUG_MODE === "default") {
            console.log("Max characters limit set to:", data.maxCharacters);
        }
        if (typeof data.maxCharactersCutOnDot === "boolean" && DEBUG && DEBUG_MODE === "default") {
            console.log("Max characters cut on dot set to:", data.maxCharactersCutOnDot);
        }
        // TODO add XTC and dry sampling options from config

        let accumulatedText = "";

        if (DEBUG && DEBUG_MODE === "default") {
            console.log("Generation config:", basicConfig);
            console.log("Prompt:", prompt);
            console.log("Using grammar:", data.grammar);
        }

        await completion.generateCompletion(prompt, {
            ...basicConfig,
            signal: CONTROLLER.ctrl.signal,
            stopOnAbortSignal: true,
            grammar,
            onTextChunk(textSrc) {
                try {
                    const text = textSrc;
                    accumulatedText += text;

                    if (DEBUG && (DEBUG_MODE === "default" || DEBUG_MODE === "output")) {
                        // use this weird character to denote token boundaries
                        // @ts-ignore
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
                                CONTROLLER.ctrl?.abort();
                                CONTROLLER.ctrl = null;
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
                                CONTROLLER.ctrl?.abort();
                                CONTROLLER.ctrl = null;
                                return;
                            } else if (data.maxCharactersCutOnDot && text.indexOf('.') !== -1) {
                                const potentialPartBeforeDot = text.split('.')[0];
                                if (potentialPartBeforeDot.length > 0) {
                                    answer += potentialPartBeforeDot;
                                }
                                console.log("\nAborting completion due to max characters limit (cut on dot).");
                                CONTROLLER.ctrl?.abort();
                                CONTROLLER.ctrl = null;
                                return;
                            }
                        }
                    }

                    answer += text;

                    if (regexStopAfter.length > 0) {
                        for (const stopRegex of regexStopAfter) {
                            if (stopRegex.test(answer)) {
                                console.log("\nAborting completion due to stopAfter trigger matched:", stopRegex);
                                CONTROLLER.ctrl?.abort();
                                CONTROLLER.ctrl = null;
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
    CONTROLLER.ctrl = null;
}

// TODO implement wordRejection, where rejectedWordsInNarration is expected to be "you" "your" etc... and delimiter - or emdash.

/**
 * @typedef {Object} WordRejectionSettings
 * @property {Array<string>} rejectedWordsInNarration - The words to reject in narration.
 * @property {string | null} postRejectedWordInNarrationGrammar - The grammar to use after a rejected word in narration.
 * @property {Array<string>} rejectedWordsInDialogue - The words to reject in dialogue.
 * @property {string | null} postRejectedWordInDialogueGrammar - The grammar to use after a rejected word in dialogue.
 * @property {Array<string>} delimiters - The delimiters to use for splitting the text into words.
 * @property {boolean} startsInDialogue - Whether the text starts in dialogue or not.
 */

/**
 * @typedef {Object} GenerationData
 * @property {Array<{role: string, content: string}>} messages - The chat messages to generate a completion for.
 * @property {Array<string>} stopAt - The strings to stop generation at.
 * @property {Array<string>} stopAfter - The strings to stop generation after.
 * @property {number} maxParagraphs - The maximum number of paragraphs to generate.
 * @property {number} maxCharacters - The maximum number of characters to generate.
 * @property {number} maxSafetyCharacters - The maximum number of characters to generate before stopping for safety.
 * @property {boolean} maxCharactersCutOnDot - Whether to cut on dot when max characters is reached, not just newline
 * @property {string | null} trail - The trailing text to append to the prompt.
 * @property {string} gear - The gear to use for generation (standard or analyze).
 * @property {string | null} grammar - The grammar to use for generation.
 * @property {WordRejectionSettings} wordRejection - The word rejection settings.
 */

/**
 * @param {GenerationData} data 
 * @param {(text: string) => void} onToken 
 * @param {() => void} onDone 
 * @param {(error: Error) => void} onError 
 */
export async function generateCompletion(data, onToken, onDone, onError) {
    if (CONTROLLER.ctrl) {
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

    if (typeof data.maxCharactersCutOnDot !== "boolean") {
        throw new Error("Invalid maxCharactersCutOnDot format");
    }

    if (typeof data.maxSafetyCharacters !== "number" || isNaN(data.maxSafetyCharacters) || data.maxSafetyCharacters < 0) {
        throw new Error("Invalid maxSafetyCharacters format");
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
    if (typeof data.maxParagraphs === "number" && DEBUG && DEBUG_MODE === "default") {
        console.log("Max paragraphs limit set to:", data.maxParagraphs);
    }
    if (typeof data.maxCharacters === "number" && DEBUG && DEBUG_MODE === "default") {
        console.log("Max characters limit set to:", data.maxCharacters);
    }

    if (typeof data.maxCharactersCutOnDot === "boolean" && DEBUG && DEBUG_MODE === "default") {
        console.log("Max characters cut on dot set to:", data.maxCharactersCutOnDot);
    }

    return await runPrompt(prompt, data, basicConfig, onToken, onDone, onError);
}

/**
 * TODO implement bad words in python version too
 * 
 * @param {string} prompt 
 * @param {GenerationData} data 
 * @param {*} basicConfig
 * @param {(text: string) => void} onToken 
 * @param {() => void} onDone 
 * @param {(error: Error) => void} onError 
 */
async function runPrompt(
    prompt,
    data,
    basicConfig,
    onToken,
    onDone,
    onError,
) {
    if (!MODEL.model) {
        throw new Error("Model not loaded");
    }

    let bufferedText = "";
    let producedText = "";
    const BUFFERED_SIZE = 20; // buffer 20 characters

    let inDialoge = data.wordRejection ? data.wordRejection.startsInDialogue : false;

    const grammar = data.grammar ? await LLAMA.createGrammar({ grammar: data.grammar }) : undefined;

    const regexStopAfter = data.stopAfter.map(s => new RegExp(`(^|[.,;])\\s*${escapeRegExp(s)}\\s*([.,;]|$)`, 'i'));

    /**
     * @type {{
     *    prompt: string,
     *    data: GenerationData,
     * } | null}
     */
    let failedDueToBadWordReprocessArgs = null;
    let earlyAbort = false;

    /**
     * @param {string} text 
     */
    const increaseBufferedText = (text) => {
        bufferedText += text;
        producedText += text;

        if (DEBUG && (DEBUG_MODE === "default" || DEBUG_MODE === "output")) {
            // use this weird character to denote token boundaries
            // @ts-ignore
            process.stdout.write(text + "§");
        }

        if (data.wordRejection) {
            for (const delimiter of data.wordRejection.delimiters) {
                while (bufferedText.includes(delimiter)) {
                    const firstPart = bufferedText.split(delimiter)[0];
                    const secondPart = bufferedText.slice(firstPart.length + delimiter.length);
                    inDialoge = !inDialoge;
                    bufferedText = secondPart;
                    onToken(firstPart + delimiter);
                }
            }
        }

        // For the non prototype this can be optimized better but for now it's fine
        // count paragraphs
        let paragraphCount = 0;

        if (typeof data.maxParagraphs === "number" && data.maxParagraphs > 0) {
            for (let i = 0; i < producedText.length; i++) {
                if (producedText[i] === '\n' && producedText[i + 1] === '\n') {
                    paragraphCount += 1;
                }
                //console.log("Current paragraph count:", paragraphCount);

                // this should hit exactly at paragraph end
                if (paragraphCount >= data.maxParagraphs) {
                    //console.log("Max paragraphs reached:", paragraphCount, "stopping completion early.");
                    // I think newlines are whole tokens, but just in case the text contains some text too
                    const potentialPartBeforeNew = bufferedText.split("\n")[0]
                    if (potentialPartBeforeNew.length > 0) {
                        onToken(potentialPartBeforeNew);
                    }
                    console.log("\nAborting completion due to max paragraphs limit.");
                    CONTROLLER.ctrl?.abort();
                    CONTROLLER.ctrl = null;
                    earlyAbort = true;
                    return;
                }
            }
        }

        if (typeof data.maxCharacters === "number" && data.maxCharacters > 0) {
            const characterCount = producedText.length;

            //console.log("Current character count:", characterCount);

            if (characterCount >= data.maxCharacters) {
                //console.log("Trying to abort but no paragraph end found yet.");
                // let's find if our text is finally finishing a paragraph
                if (text.indexOf('\n') !== -1) {
                    //console.log("Max characters reached:", characterCount, "stopping completion at this paragraph end.");
                    const potentialPartBeforeNew = bufferedText.split("\n")[0]
                    if (potentialPartBeforeNew.length > 0) {
                        onToken(potentialPartBeforeNew);
                    }
                    console.log("\nAborting completion due to max characters limit.");
                    CONTROLLER.ctrl?.abort();
                    CONTROLLER.ctrl = null;
                    earlyAbort = true;
                    return;
                }
            }
        }
        if (typeof data.maxSafetyCharacters === "number" && data.maxSafetyCharacters > 0) {
            const characterCount = producedText.length;
            //console.log("Current character count:", characterCount);

            if (characterCount >= data.maxSafetyCharacters) {
                console.log("\nAborting completion due to max safety characters limit.");
                onToken(bufferedText);
                CONTROLLER.ctrl?.abort();
                CONTROLLER.ctrl = null;
                earlyAbort = true;
                return;
            }
        }

        if (regexStopAfter.length > 0) {
            for (const stopRegex of regexStopAfter) {
                if (stopRegex.test(producedText)) {
                    console.log("\nAborting completion due to stopAfter trigger matched:", stopRegex);
                    CONTROLLER.ctrl?.abort();
                    CONTROLLER.ctrl = null;
                    earlyAbort = true;
                    return;
                }
            }
        }

        if (bufferedText.length >= BUFFERED_SIZE) {
            // get the first n characters to make the buffered text exactly BUFFERED_SIZE
            const toSend = bufferedText.slice(0, BUFFERED_SIZE);
            bufferedText = bufferedText.slice(BUFFERED_SIZE);
            onToken(toSend);
        }

        if (data.wordRejection) {
            const forbiddenWords = inDialoge ? data.wordRejection.rejectedWordsInDialogue : data.wordRejection.rejectedWordsInNarration;
            const postGrammar = inDialoge ? data.wordRejection.postRejectedWordInDialogueGrammar : data.wordRejection.postRejectedWordInNarrationGrammar;
            // check using regex for words
            for (const word of forbiddenWords) {
                const regex = new RegExp(`\\b${escapeRegExp(word)}\\b`, 'i');
                if (regex.test(bufferedText)) {
                    const indexBadWordFound = bufferedText.search(regex);
                    const textBeforeBadWord = bufferedText.slice(0, indexBadWordFound);

                    onToken(textBeforeBadWord);

                    const textAfterBadWord = bufferedText.slice(indexBadWordFound + word.length);

                    console.log(`\nAborting completion due to forbidden word detected: ${word}`);
                    CONTROLLER.ctrl?.abort();
                    CONTROLLER.ctrl = null;

                    const newData = { ...data };
                    newData.grammar = postGrammar;
                    if (data.maxCharacters !== 0) {
                        newData.maxCharacters = data.maxCharacters - producedText.length + textAfterBadWord.length;
                        if (newData.maxCharacters <= 0) {
                            console.log("\nAborting completion due to max characters limit reached after forbidden word.");
                            onDone();
                            return;
                        }
                    }
                    if (data.maxSafetyCharacters !== 0) {
                        newData.maxSafetyCharacters = data.maxSafetyCharacters - producedText.length + textAfterBadWord.length;
                    }
                    newData.wordRejection.startsInDialogue = inDialoge;
                    newData.maxParagraphs = data.maxParagraphs - paragraphCount;

                    const producedTextWithoutTheBadWord = producedText.slice(0, producedText.length - textAfterBadWord.length);
                    prompt += producedTextWithoutTheBadWord;

                    failedDueToBadWordReprocessArgs = {
                        prompt,
                        data: newData,
                    };
                    earlyAbort = true;
                    return false;
                }
            }
        }

        return true;
    }

    let context = null;
    let completion = null;
    CONTROLLER.ctrl = new AbortController();
    try {
        // Create context and completion for raw text
        context = await MODEL.model.createContext();
        completion = new LlamaCompletion({
            contextSequence: context.getSequence()
        });

        if (DEBUG && DEBUG_MODE === "default") {
            console.log("Generation config:", basicConfig);
            console.log("Prompt:", prompt);
        }

        await completion.generateCompletion(prompt, {
            ...basicConfig,
            signal: CONTROLLER.ctrl.signal,
            stopOnAbortSignal: true,
            grammar,
            onTextChunk(textSrc) {
                if (failedDueToBadWordReprocessArgs) {
                    return;
                }
                try {
                    increaseBufferedText(textSrc);
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

    if (failedDueToBadWordReprocessArgs) {
        CONTROLLER.ctrl = null;
        console.log("\nReprocessing due to forbidden word detected...");
        await runPrompt(
            // @ts-ignore typescript is wrong
            failedDueToBadWordReprocessArgs.prompt,
            // @ts-ignore typescript is wrong
            failedDueToBadWordReprocessArgs.data,
            basicConfig,
            onToken,
            onDone,
            onError
        );
        return;
    } else if (!earlyAbort && bufferedText.length > 0) {
        // send the last buffered text left
        onToken(bufferedText);
    }

    onDone();
    CONTROLLER.ctrl = null;
}