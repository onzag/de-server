import { WebSocketServer } from "ws";
import { CONTROLLER, MODEL, generateCompletion, prepareAnalysis, runQuestion } from "./base.js";

const wss = new WebSocketServer({ port: 8765, host: '0.0.0.0' });

wss.on('connection', (ws) => {
    console.log('Client connected');

    ws.send(JSON.stringify({ type: 'ready', message: 'Model is ready' }));

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