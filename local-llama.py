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
                    
                    def run_generator(payload):
                        for result in base.generate_completion(payload):
                            if 'error' in result:
                                loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "error", "rid": rid, "message": str(result['error'])})))
                                nonlocal internalid
                                if internalid is not None:
                                    internalrids.discard(internalid)  # Remove from active requests on error
                            elif 'token' in result:
                                loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "token", "rid": rid, "text": result['token']})))
                            elif 'done' in result and result['done']:
                                loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "done", "rid": rid})))
                                nonlocal internalid
                                if internalid is not None:
                                    internalrids.discard(internalid)  # Remove from active requests on done
                            elif 'request_id' in result:
                                nonlocal internalid
                                internalid = result['request_id']
                                internalrids.add(internalid)  # Track active request ID
                    await loop.run_in_executor(executor, run_generator, data['payload'])
                elif data.get('action') == 'analyze-prepare':
                    if not data.get('payload'):
                        raise ValueError("Invalid payload for analyze-prepare")
                    
                    def run_generator(payload):
                        for result in base.prepare_analysis(payload):
                            if 'error' in result:
                                loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "error", "rid": rid, "message": str(result['error'])})))
                                nonlocal internalid
                                if internalid is not None:
                                    internalrids.discard(internalid)  # Remove from active requests on error
                            elif 'done' in result and result['done']:
                                loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "analyze-ready", "rid": rid})))
                                nonlocal internalid
                                if internalid is not None:
                                    internalrids.discard(internalid)  # Remove from active requests on done

                    await loop.run_in_executor(executor, run_generator, data['payload'])
                elif data.get('action') == 'analyze-question':
                    if not data.get('payload'):
                        raise ValueError("Invalid payload for analyze-question")
                    
                    def run_generator(payload):
                        for result in base.run_question(payload):
                            if 'error' in result:
                                loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "error", "rid": rid, "message": str(result['error'])})))
                                nonlocal internalid
                                if internalid is not None:
                                    internalrids.discard(internalid)  # Remove from active requests on error
                            elif 'answer' in result:
                                loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "answer", "rid": rid, "text": result['answer']})))
                                if result.get('done'):
                                    nonlocal internalid
                                    if internalid is not None:
                                        internalrids.discard(internalid)  # Remove from active requests on done
                            elif 'request_id' in result:
                                nonlocal internalid
                                internalid = result['request_id']
                                internalrids.add(internalid)  # Track active request ID

                    await loop.run_in_executor(executor, run_generator, data['payload'])
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
