// this is an extremely simple websocket server that runs a llama.cpp instance locally
// that can only handle one request at a time

// this is very good for consumer grade hardware that can't run anything in parallell
// and the model runs locally in the same machine

import { WebSocketServer } from "ws";
import { CONTROLLER, MODEL, MODEL_PATH, generateCompletion, prepareAnalysis, runQuestion, loadConfig } from "./base.js";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { randomBytes } from "crypto";
import { URL } from "url";
import { createServer as createHttpsServer } from "https";
import { createServer as createHttpServer } from "http";

const DEV = process.env.DEV === "1";
const SSL = process.env.SSL === "1";

const argv = process.argv.slice(2);
if (argv.length < 1) {
    console.error("Please provide a model path as the first argument.");
    process.exit(1);
}

const END_TOKEN = (await loadConfig(argv[0])).endToken;

let expectedSecret;
if (!DEV) {
    if (existsSync("./secret")) {
        expectedSecret = readFileSync("./secret", "utf-8").trim();
        console.log("Using Secret key from ./secret for authentication");
    } else {
        expectedSecret = randomBytes(64).toString("hex");
        writeFileSync("./secret", expectedSecret);
        console.log(`Generated new Secret key and saved to ./secret: ${expectedSecret}`);
    }
} else {
    expectedSecret = "dev-secret-12345678900abcdef";
}

console.log("DEV mode:", DEV);
console.log("SSL mode:", SSL);
console.log(`Starting Local LLaMA WebSocket Server, listening on ${SSL ? "wss" : "ws"}://0.0.0.0:8765`);

const verifyClient = (info) => {
    const url = new URL(info.req.url, `http://${info.req.headers.host}`);
    const secret = url.searchParams.get("secret");
    if (secret !== expectedSecret) {
        console.log("Unauthorized connection attempt with invalid secret");
        return false;
    }
    console.log("Client authenticated successfully");
    return true;
};

let server;
if (SSL) {
    try {
        server = createHttpsServer({
            cert: readFileSync("./cert.pem"),
            key: readFileSync("./key.pem"),
        });
    } catch (e) {
        console.error("Failed to create SSL server:", e.message);
        process.exit(1);
    }
} else {
    server = createHttpServer();
}

let CONTEXT_WINDOW_SIZE = 2048 * 4; // 8k context
if (process.env.CONTEXT_WINDOW_SIZE) {
    const envSize = parseInt(process.env.CONTEXT_WINDOW_SIZE);
    if (!isNaN(envSize) && envSize > 0) {
        CONTEXT_WINDOW_SIZE = envSize;
    }
}

// ── Static info page ──────────────────────────────────────────────────────
// Load the HTML template once at startup. Visiting http(s)://host:8765/ in a
// browser shows basic server status; this also gives users a way to manually
// accept the self-signed certificate so subsequent wss:// connections work.
import { fileURLToPath } from "url";
import { dirname, join } from "path";
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const INDEX_HTML_TEMPLATE = readFileSync(join(__dirname, "index.html"), "utf-8");
const SERVER_START_TIME = Date.now();
const ARG_CONFIG_PATH = argv[0];

/**
 * @param {Record<string, string>} replacements
 */
function renderIndexHtml(replacements) {
    return INDEX_HTML_TEMPLATE.replace(/\{\{(\w+)\}\}/g, (_, key) =>
        Object.prototype.hasOwnProperty.call(replacements, key)
            ? String(replacements[key])
            : `{{${key}}}`,
    );
}

/**
 * @param {number} ms
 */
function formatUptime(ms) {
    const s = Math.floor(ms / 1000);
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const parts = [];
    if (d) parts.push(`${d}d`);
    if (h || d) parts.push(`${h}h`);
    if (m || h || d) parts.push(`${m}m`);
    parts.push(`${sec}s`);
    return parts.join(" ");
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

server.on("request", (req, res) => {
    // Only the root path serves the info page; everything else is 404.
    const url = new URL(req.url || "/", `http://${req.headers.host}`);
    if (url.pathname !== "/" && url.pathname !== "/index.html") {
        res.statusCode = 404;
        res.setHeader("Content-Type", "text/plain; charset=utf-8");
        res.end("Not found");
        return;
    }

    // MODEL_PATH and MODEL are live ES module bindings, so we always read
    // their current values here without needing to re-import.
    const html = renderIndexHtml({
        PROTOCOL: SSL ? "wss" : "ws",
        DEV_MODE: DEV ? "DEV (insecure secret)" : "production",
        SSL_MODE: SSL ? "enabled" : "disabled",
        MODEL_LOADED: MODEL ? "yes" : "no",
        MODEL_PATH: escapeHtml(MODEL_PATH || "(none)"),
        CONFIG_PATH: escapeHtml(ARG_CONFIG_PATH || "(none)"),
        CONFIG_MODE: escapeHtml(process.env.CONFIG_MODE || "(see config file)"),
        END_TOKEN: escapeHtml(END_TOKEN || ""),
        CONTEXT_WINDOW: String(CONTEXT_WINDOW_SIZE),
        GPU: escapeHtml(process.env.GPU || "auto"),
        UPTIME: formatUptime(Date.now() - SERVER_START_TIME),
        PROGRAM: "Node.js Local LLaMA Server",
    });
    res.statusCode = 200;
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    res.setHeader("Cache-Control", "no-store");
    res.end(html);
});

const wss = new WebSocketServer({ server, verifyClient });
server.listen(8765, '0.0.0.0');

/**
 * @type {Promise<void>}
 */
let lastGenerationPromise = Promise.resolve();

wss.on('connection', (ws) => {
    console.log('Client connected');

    ws.send(JSON.stringify({
        type: 'ready',
        message: 'Model is ready',
        context_window_size: CONTEXT_WINDOW_SIZE,
        supports_parallel_requests: false,
        end_token: END_TOKEN,
    }));

    ws.on('message', async (message) => {
        let rid = "no-rid";
        try {
            // @ts-ignore
            const data = JSON.parse(message);
            rid = data.rid || "no-rid";

            // Handle different actions
            if (data.action === 'infer') {
                if (!data.payload) {
                    throw new Error("Invalid payload for infer");
                }
                lastGenerationPromise = lastGenerationPromise.catch(() => {}).then(() => {
                    return generateCompletion(data.payload, (text) => {
                        ws.send(JSON.stringify({ type: 'token', rid, text }));
                    }, () => {
                        ws.send(JSON.stringify({ type: 'done', rid }));
                    }, (error) => {
                        ws.send(JSON.stringify({ type: 'error', rid, message: error.message }));
                    });
                });
                await lastGenerationPromise;
            } else if (data.action === 'analyze-prepare') {
                if (!data.payload) {
                    throw new Error("Invalid payload for analyze-prepare");
                }
                lastGenerationPromise = lastGenerationPromise.catch(() => {}).then(() => {
                    return prepareAnalysis(data.payload, () => {
                        ws.send(JSON.stringify({ type: 'analyze-ready', rid }));
                    }, (error) => {
                        ws.send(JSON.stringify({ type: 'error', rid, message: error.message }));
                    });
                });
                await lastGenerationPromise;
            } else if (data.action === 'analyze-question') {
                if (!data.payload) {
                    throw new Error("Invalid payload for analyze-question");
                }
                lastGenerationPromise = lastGenerationPromise.catch(() => {}).then(() => {
                    return runQuestion(data.payload, (text) => {
                        ws.send(JSON.stringify({ type: 'answer', rid, text }));
                    }, (error) => {
                        ws.send(JSON.stringify({ type: 'error', rid, message: error.message }));
                    });
                });
                await lastGenerationPromise;
            } else if (data.action === 'count-tokens') {
                if (!data.payload || typeof data.payload.text !== "string") {
                    throw new Error("Invalid payload for count-tokens");
                }
                const text = data.payload.text;
                const tokens = MODEL.tokenize(text);
                ws.send(JSON.stringify({ type: 'count', rid, n_tokens: tokens.length }));
            } else if (data.action === 'cancel') {
                if (!data.rid) {
                    throw new Error("Missing rid for cancel action");
                }
                if (data.rid === "no-rid") {
                    throw new Error("Cannot cancel request with no rid");
                }
                if (CONTROLLER) {
                    CONTROLLER.abort();
                    // Cancel the request with the given rid
                    // This will depend on how you track active requests and their IDs
                    ws.send(JSON.stringify({ type: "cancelled", rid: data.rid }));
                } else {
                    ws.send(JSON.stringify({ type: "error", rid, message: "No active controller to cancel" }));
                }
            } else {
                throw new Error("Unknown action: " + data.action);
            }
        } catch (e) {
            // @ts-ignore
            console.log(e.message);
            // @ts-ignore
            ws.send(JSON.stringify({ type: 'error', rid, message: e.message }));
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected');
        if (CONTROLLER) {
            CONTROLLER.abort();
        }
    });
});