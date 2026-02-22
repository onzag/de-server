import asyncio
import json
import sys
import websockets
from base import DEBUG, generate_completion, load_config, prepare_analysis, run_question
import base

PORT = 8765
HOST = '0.0.0.0'

async def handle_client(websocket):
    print('Client connected')
    await websocket.send(json.dumps({"type": "ready", "message": "Model is ready"}))
    loop = asyncio.get_event_loop()
    executor = None  # Use default executor (main thread)
    try:
        internalrids = set()  # Keep track of internal request IDs for this connection
        async for message in websocket:
            try:
                data = json.loads(message)
                rid = data.get('rid', 'no-rid')  # Use provided rid or default to 'no-rid'
                internalid = None  # To track the internal request ID for this action, if applicable
                # Handle different actions
                if data.get('action') == 'infer':
                    if not data.get('payload'):
                        raise ValueError("Invalid payload for infer")
                    def on_token(text):
                        loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "token", "rid": rid, "text": text})))
                    def on_done():
                        loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "done", "rid": rid})))
                        nonlocal internalid
                        if internalid is not None:
                            internalrids.discard(internalid)  # Remove from active requests on error
                    def on_error(error):
                        loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "error", "rid": rid, "message": str(error)})))
                        nonlocal internalid
                        if internalid is not None:
                            internalrids.discard(internalid)  # Remove from active requests on error
                    def on_request_id(req_id):
                        nonlocal internalid
                        internalid = req_id
                        internalrids.add(req_id)
                    await loop.run_in_executor(executor, base.generate_completion, data['payload'], on_request_id, on_token, on_done, on_error)
                elif data.get('action') == 'analyze-prepare':
                    if not data.get('payload'):
                        raise ValueError("Invalid payload for analyze-prepare")
                    def on_done():
                        loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "analyze-ready", "rid": rid})))
                        nonlocal internalid
                        if internalid is not None:
                            internalrids.discard(internalid)  # Remove from active requests on error
                    def on_error(error):
                        loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "error", "rid": rid, "message": str(error)})))
                        nonlocal internalid
                        if internalid is not None:
                            internalrids.discard(internalid)  # Remove from active requests on error
                    await loop.run_in_executor(executor, base.prepare_analysis, data['payload'], on_done, on_error)
                elif data.get('action') == 'analyze-question':
                    if not data.get('payload'):
                        raise ValueError("Invalid payload for analyze-question")
                    def on_answer(text):
                        loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "answer", "rid": rid, "text": text})))
                    def on_error(error):
                        loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "error", "rid": rid, "message": str(error)})))
                        nonlocal internalid
                        if internalid is not None:
                            internalrids.discard(internalid)  # Remove from active requests on error
                    def on_request_id(req_id):
                        nonlocal internalid
                        internalid = req_id
                        internalrids.add(req_id)
                    await loop.run_in_executor(executor, base.run_question, data['payload'], on_request_id, on_answer, on_error)
                elif data.get('action') == 'count-tokens':
                    if not data.get('payload') or not isinstance(data['payload'].get('text'), str):
                        raise ValueError("Invalid payload for count-tokens")
                    text = data['payload']['text']
                    tokens = base.MODEL.tokenizer.encode(text)
                    await websocket.send(json.dumps({"type": "count", "rid": rid, "n_tokens": len(tokens)}))
            except Exception as e:
                print(str(e))
                await websocket.send(json.dumps({"type": "error", "rid": rid, "message": str(e)}))
    except websockets.ConnectionClosed:
        print('Client disconnected')
        for internalrid in internalrids:
            base.MODEL.abort_request(internalrid)  # Stop the requests if the client disconnects

async def main():
    async with websockets.serve(handle_client, HOST, PORT):
        print(f"WebSocket server started on ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 1:
        print("Please provide a config path as the first argument.", file=sys.stderr)
        sys.exit(1)

    print("DEBUG mode:", DEBUG)
    load_config(argv[0])
    asyncio.run(main())
