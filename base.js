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
 *   "modelPath": "./model.json",
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
 * @type {import('node-llama-cpp').LlamaModel}
 */
export let MODEL = /** @type {any} */ (null);
let LLAMA = await getLlama();
let MODEL_PATH = "";

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
 *    mode?: "mistral" | "llama3";
 *    standard: {temperature: number; temperatureRange?: [number, number]; topP?: number; minP?: number; repeatPenalty?: number; frequencyPenalty?: number; presencePenalty?: number; maxTokens: number;},
 *    analyze: {temperature: number; temperatureRange?: [number, number]; topP?: number; minP?: number; repeatPenalty?: number; frequencyPenalty?: number; presencePenalty?: number; maxTokens: number;},
 * }}
 */
let CONFIG = /** @type {any} */ (null);
let CONFIG_PATH = "";

/**
 * @param {string} text 
 * @param {number} minLength 
 * @param {number} maxLength
 * @return {{
 *      repetitionAt: string,
 *      amount: number,
 *   } | null
 * }
 */
function patternRepetitionChecker(text, minLength, maxLength) {
    if (text.length < minLength) {
        return null;
    }
    for (let length = minLength; length <= Math.min(maxLength, text.length); length++) {
        const pattern = text.slice(0, length);
        const splitted = text.split(pattern);
        if (splitted.every(s => s === "")) {
            const occurrences = splitted.length - 1;
            if (occurrences > 1) {
                return {
                    repetitionAt: pattern,
                    amount: occurrences,
                };
            }
        }
    }
    return null;
}

/**
 * @param {string} text
 * @return {boolean}
 */
function aggressiveListRepetitionChecker(text) {
    const splitted = text.split(",").map(s => s.trim()).filter(s => s.length > 0);
    if (splitted.length === 1) {
        return false;
    }
    let hasHadAnotherItemDifferentFromFirst = false;
    for (let i = 1; i < splitted.length; i++) {
        const item = splitted[i];
        if (item === splitted[0] && hasHadAnotherItemDifferentFromFirst) {
            return true;
        } else if (item !== splitted[0]) {
            hasHadAnotherItemDifferentFromFirst = true;
        }
    }
    return false;
}

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
 */
async function loadConfig(configPath) {
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

    if (CONFIG.mode !== "mistral" && CONFIG.mode !== "llama3" && CONFIG.mode !== undefined) {
        throw new Error("Invalid config: mode must be 'mistral' or 'llama3' if provided");
    }

    if (MODEL_PATH !== CONFIG.modelPath) {
        // use relative path from config file
        const baseDir = path.dirname(configPath);
        const modelFullPath = path.resolve(baseDir, CONFIG.modelPath);
        await loadModel(modelFullPath);
    }
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

const argv = process.argv.slice(2);
if (argv.length < 1) {
    console.error("Please provide a model path as the first argument.");
    process.exit(1);
}

const DEBUG = process.env.DEBUG === "1";

console.log("DEBUG mode:", DEBUG);

await loadConfig(argv[0]);

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
        throw new Error("Invalid system format");
    }
    if (!data.userTrail || typeof data.userTrail !== "string") {
        throw new Error("Invalid userTrail format");
    }
    try {
        //const context = await MODEL.createContext();
        //const contextSequence = context.getSequence();
        //contextSequence.eraseContextTokenRanges

        // TODO optimize this, for now just retokenize every time
        if (CONFIG.mode === "mistral") {
            ANALYSIS_TEXT = `<s>[SYSTEM_PROMPT] ${data.system}[/SYSTEM_PROMPT][INST] ${data.userTrail}`;
        } else {
            ANALYSIS_TEXT = `<|start_header_id|>system<|end_header_id|>\n\n${data.system}<|eot_id><|start_header_id|>user<|end_header_id|>\n\n${data.userTrail}`;
        }

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
 * repetitionBuster?: boolean;
 * aggressiveListRepetitionBuster?: boolean;
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
    CONTROLLER = new AbortController();

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

    if (data.repetitionBuster !== undefined && typeof data.repetitionBuster !== "boolean") {
        throw new Error("Invalid repetitionBuster format");
    }

    if (data.aggressiveListRepetitionBuster !== undefined && typeof data.aggressiveListRepetitionBuster !== "boolean") {
        throw new Error("Invalid aggressiveListRepetitionBuster format");
    }

    const regexStopAfter = data.stopAfter.map(s => new RegExp(`(^|[.,;])\\s*${escapeRegExp(s)}\\s*([.,;]|$)`, 'i'));

    let prompt = "";
    if (CONFIG.mode === "mistral") {
        prompt = ANALYSIS_TEXT + "\n\n" + data.question + `\n[/INST]\n\n` + (data.trail || "");
    } else {
        prompt = ANALYSIS_TEXT + "\n" + data.question + `\n<|start_header_id|>assistant<|end_header_id|>\n\n` + (data.trail || "");
    }
    let context = null
    let completion = null;
    let answer = "";
    try {
        const grammar = data.grammar ? await LLAMA.createGrammar({
            grammar: data.grammar,
        }) : undefined;
        // Create context and completion for raw text
        context = await MODEL.createContext();
        completion = new LlamaCompletion({
            contextSequence: context.getSequence(),
        });

        const basicConfig = {
            temperature: CONFIG.analyze.temperature,
            topP: CONFIG.analyze.topP,
            minP: CONFIG.analyze.minP,
            repeatPenalty: {
                penalty: CONFIG.analyze.repeatPenalty,
                frequencyPenalty: CONFIG.analyze.frequencyPenalty,
                presencePenalty: CONFIG.analyze.presencePenalty,
            },
            customStopTriggers: (CONFIG.mode === "mistral" ? ["</s>", "[INST]"] : ["<|eot_id|>", "<|start_header_id|>"]).concat(data.stopAt || []),
            maxTokens: CONFIG.analyze.maxTokens || 512,
        }
        if (CONFIG.analyze.temperatureRange) {
            basicConfig.temperature = getDynamicTemperature(CONFIG.analyze.temperatureRange[0], CONFIG.analyze.temperatureRange[1]);
        }
        if (typeof data.maxParagraphs === "number") {
            console.log("Max paragraphs limit set to:", data.maxParagraphs);
        }
        if (typeof data.maxCharacters === "number") {
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

                    if (data.repetitionBuster) {
                        const repetition = patternRepetitionChecker(answer, 5, 300);
                        if (repetition && repetition.amount >= 3) {
                            console.log("\nAborting completion due to repetition detected:", repetition);
                            CONTROLLER?.abort();
                            CONTROLLER = null;
                            return;
                        }
                    }

                    if (data.aggressiveListRepetitionBuster && aggressiveListRepetitionChecker(answer)) {
                        console.log("\nAborting completion due to aggressive list repetition detected");
                        CONTROLLER?.abort();
                        CONTROLLER = null;
                        return;
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
 * @param {{messages: Array<{role: string, content: string}>, stopAt: Array<string>, stopAfter: Array<string>, maxParagraphs: number, maxCharacters: number, startCountingFromToken: string | null, trail: string | null}} data 
 * @param {(text: string) => void} onToken 
 * @param {() => void} onDone 
 * @param {(error: Error) => void} onError 
 */
export async function generateCompletion(data, onToken, onDone, onError) {
    if (CONTROLLER) {
        throw new Error("Another generation is already in progress");
    }
    CONTROLLER = new AbortController();

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

    if (data.startCountingFromToken !== null && typeof data.startCountingFromToken !== "string") {
        throw new Error("Invalid startCountingFromToken format");
    }

    if (data.trail !== null && typeof data.trail !== "string") {
        throw new Error("Invalid trail format");
    }

    if (!Array.isArray(data.stopAfter)) {
        throw new Error("Invalid stopAfter format");
    }

    // clear previous analysis
    ANALYSIS_TEXT = null;

    let prompt = "";
    if (CONFIG.mode === "mistral") {
        prompt += "<s>";
    }
    for (const msg of data.messages) {
        if (typeof msg.content !== "string") {
            throw new Error("Invalid message content");
        } else if (typeof msg.role !== "string") {
            throw new Error("Invalid message role");
        } else if (!["user", "assistant", "system"].includes(msg.role)) {
            throw new Error("Invalid message role: " + msg.role);
        }
        if (CONFIG.mode === "mistral") {
            if (msg.role === "system") {
                prompt += `[SYSTEM_PROMPT] ${msg.content}[/SYSTEM_PROMPT][INST]`;
            } else {
                prompt += "\n\n" + msg.content
            }
        } else {
            prompt += `<|start_header_id|>${msg.role}<|end_header_id|>\n\n${msg.content}<|eot_id>`;
        }

    }
    if (CONFIG.mode === "mistral") {
        prompt += "[/INST]\n\n";
    } else {
        prompt += "\n<|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    if (data.trail) {
        prompt += data.trail;
    }

    let context = null
    let completion = null;
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
            customStopTriggers: (CONFIG.mode === "mistral" ? ["</s>", "[INST]"] : ["<|eot_id|>", "<|start_header_id|>"]).concat(data.stopAt || []),
            maxTokens: CONFIG.standard.maxTokens || 512,
        }
        if (CONFIG.standard.temperatureRange) {
            basicConfig.temperature = getDynamicTemperature(CONFIG.standard.temperatureRange[0], CONFIG.standard.temperatureRange[1]);
        }
        // TODO add XTC and dry sampling options from config
        if (typeof data.maxParagraphs === "number") {
            console.log("Max paragraphs limit set to:", data.maxParagraphs);
        }
        if (typeof data.maxCharacters === "number") {
            console.log("Max characters limit set to:", data.maxCharacters);
        }

        let hasBegunCounting = data.startCountingFromToken === null ? true : false;
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
            onTextChunk(textSrc) {
                try {
                    const text = textSrc;
                    accumulatedText += text;
                    if (DEBUG) {
                        // use this weird character to denote token boundaries
                        process.stdout.write(text + "§");
                    }
                    if (!hasBegunCounting && data.startCountingFromToken && accumulatedText.includes(data.startCountingFromToken)) {
                        hasBegunCounting = true;
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