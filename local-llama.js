// this is an extremely simple websocket server that runs a llama.cpp instance locally
// that can only handle one request at a time

// this is very good for consumer grade hardware that can't run anything in parallell
// and the model runs locally in the same machine

import { WebSocketServer } from "ws";
import { CONTROLLER, MODEL, generateCompletion, prepareAnalysis, runQuestion } from "./base.js";

const wss = new WebSocketServer({ port: 8765, host: '0.0.0.0' });

let CONTEXT_WINDOW_SIZE = 2048 * 4; // 8k context
if (process.env.CONTEXT_WINDOW_SIZE) {
    const envSize = parseInt(process.env.CONTEXT_WINDOW_SIZE);
    if (!isNaN(envSize) && envSize > 0) {
        CONTEXT_WINDOW_SIZE = envSize;
    }
}

wss.on('connection', (ws) => {
    console.log('Client connected');

    ws.send(JSON.stringify({ type: 'ready', message: 'Model is ready', context_window_size: CONTEXT_WINDOW_SIZE, supports_parallel_requests: false }));

    ws.on('message', async (message) => {
        try {
            // @ts-ignore
            const data = JSON.parse(message);
            const rid = data.rid || "no-rid";

            // Handle different actions
            if (data.action === 'infer') {
                if (!data.payload) {
                    throw new Error("Invalid payload for infer");
                }
                await generateCompletion(data.payload, (text) => {
                    ws.send(JSON.stringify({ type: 'token', rid, text }));
                }, () => {
                    ws.send(JSON.stringify({ type: 'done', rid }));
                }, (error) => {
                    ws.send(JSON.stringify({ type: 'error', rid, message: error.message }));
                });
            } else if (data.action === 'analyze-prepare') {
                if (!data.payload) {
                    throw new Error("Invalid payload for analyze-prepare");
                }
                await prepareAnalysis(data.payload, () => {
                    ws.send(JSON.stringify({ type: 'analyze-ready', rid }));
                }, (error) => {
                    ws.send(JSON.stringify({ type: 'error', rid, message: error.message }));
                });
            } else if (data.action === 'analyze-question') {
                if (!data.payload) {
                    throw new Error("Invalid payload for analyze-question");
                }
                await runQuestion(data.payload, (text) => {
                    ws.send(JSON.stringify({ type: 'answer', rid, text }));
                }, (error) => {
                    ws.send(JSON.stringify({ type: 'error', rid, message: error.message }));
                });
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