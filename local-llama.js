// this is an extremely simple websocket server that runs a llama.cpp instance locally
// that can only handle one request at a time

// this is very good for consumer grade hardware that can't run anything in parallell
// and the model runs locally in the same machine

import { WebSocketServer } from "ws";
import { CONTROLLER, MODEL, generateCompletion, prepareAnalysis, runQuestion } from "./base.js";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { randomBytes } from "crypto";
import { URL } from "url";
import { createServer as createHttpsServer } from "https";
import { createServer as createHttpServer } from "http";

const DEV = process.env.DEV === "1";
const SSL = process.env.SSL === "1";

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

const wss = new WebSocketServer({ server, verifyClient });
server.listen(8765, '0.0.0.0');

let CONTEXT_WINDOW_SIZE = 2048 * 4; // 8k context
if (process.env.CONTEXT_WINDOW_SIZE) {
    const envSize = parseInt(process.env.CONTEXT_WINDOW_SIZE);
    if (!isNaN(envSize) && envSize > 0) {
        CONTEXT_WINDOW_SIZE = envSize;
    }
}

/**
 * @type {Promise<void>}
 */
let lastGenerationPromise = Promise.resolve();

wss.on('connection', (ws) => {
    console.log('Client connected');

    ws.send(JSON.stringify({ type: 'ready', message: 'Model is ready', context_window_size: CONTEXT_WINDOW_SIZE, supports_parallel_requests: false }));

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