# This spins up a local vllm server that is more production ready than the js one and is meant
# to be used in strong compute hardware when the entire model fits on VRAM
# It supports multiple concurrent requests, cancellation, and has a more robust protocol for error handling and responses. It also includes a simple secret key
# authentication mechanism to prevent unauthorized access.

# While this is more or less complete, it doesn't really have robust security measures in place for multiple users
# that are not trusted, so it should only be used with friends or a known group that has access to the "secret" file that contains
# the password for access the server

# this can also be used for production but it will need a proxy to handle proper security

import asyncio
import json
import sys
import websockets
import base

PORT = 8765
HOST = '0.0.0.0'

async def handle_client(websocket):
    print('Client connected')

    await websocket.send(json.dumps({"type": "ready", "message": "Model is ready", "context_window_size": base.CONTEXT_WINDOW_SIZE, "supports_parallel_requests": True}))

    websocket

    loop = asyncio.get_event_loop()
    executor = None  # Use default executor (main thread)
    try:
        requestid_to_rid = {}  # Map internal request IDs to client rids for cancellation
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
                                    requestid_to_rid.pop(internalid, None)  # Remove from active requests on error
                            elif 'token' in result:
                                loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "token", "rid": rid, "text": result['token']})))
                            elif 'done' in result and result['done']:
                                loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "done", "rid": rid})))
                                nonlocal internalid
                                if internalid is not None:
                                    requestid_to_rid.pop(internalid, None)  # Remove from active requests on done
                            elif 'request_id' in result:
                                nonlocal internalid
                                internalid = result['request_id']
                                requestid_to_rid[internalid] = rid  # Track active request ID
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
                                    requestid_to_rid.pop(internalid, None)  # Remove from active requests on error
                            elif 'done' in result and result['done']:
                                loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "analyze-ready", "rid": rid})))
                                nonlocal internalid
                                if internalid is not None:
                                    requestid_to_rid.pop(internalid, None)  # Remove from active requests on done

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
                                    requestid_to_rid.pop(internalid, None)  # Remove from active requests on error
                            elif 'answer' in result:
                                loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "answer", "rid": rid, "text": result['answer']})))
                                if result.get('done'):
                                    nonlocal internalid
                                    if internalid is not None:
                                        requestid_to_rid.pop(internalid, None)  # Remove from active requests on done
                            elif 'request_id' in result:
                                nonlocal internalid
                                internalid = result['request_id']
                                requestid_to_rid[internalid] = rid  # Track active request ID

                    await loop.run_in_executor(executor, run_generator, data['payload'])
                elif data.get('action') == 'count-tokens':
                    if not data.get('payload') or not isinstance(data['payload'].get('text'), str):
                        raise ValueError("Invalid payload for count-tokens")
                    text = data['payload']['text']
                    tokens = base.MODEL.tokenizer.encode(text)
                    await websocket.send(json.dumps({"type": "count", "rid": rid, "n_tokens": len(tokens)}))
                elif data.get('action') == 'cancel':
                    if not data.get('rid'):
                        raise ValueError("Missing rid for cancel action")
                    # Cancel the request with the given rid
                    cancel_rid = data['rid']
                    if (cancel_rid == 'no-rid'):
                        await websocket.send(json.dumps({"type": "error", "rid": rid, "message": "Cannot cancel request without a valid rid"}))
                        continue
                    internalid = None
                    for key, value in requestid_to_rid.items():
                        if value == cancel_rid:
                            internalid = key
                            break
                    if internalid is not None:
                        base.MODEL.abort_request(internalid)
                        requestid_to_rid.pop(internalid, None)  # Remove from active requests
            except Exception as e:
                print(str(e))
                await websocket.send(json.dumps({"type": "error", "rid": rid, "message": str(e)}))
    except websockets.ConnectionClosed:
        print('Client disconnected')
        for internalrid in requestid_to_rid.keys():
            base.MODEL.abort_request(internalrid)  # Stop the requests if the client disconnects

async def main():
    async with websockets.serve(handle_client, HOST, PORT, process_request=process_request):
        print(f"WebSocket server started on ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever

async def process_request(path, request_headers):
    api_key = request_headers.get("secret")

    # read the local ./secret file for the expected API key
    try:
        with open("./secret", "r") as f:
            expected_api_key = f.read().strip()
    except Exception as e:
        # create a new random API secret and save it to the file
        import secrets
        expected_api_key = secrets.token_hex(512)
        with open("./secret", "w") as f:
            f.write(expected_api_key)
        print(f"Generated new Secret key and saved to ./secret: {expected_api_key}")

    if api_key != expected_api_key:
        print(f"Unauthorized connection attempt with invalid secret")
        return (401, [], b'Unauthorized')  # Return 401 Unauthorized if the secret key is incorrect

    # Returning None continues the WebSocket handshake
    return None

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 1:
        print("Please provide a config path as the first argument.", file=sys.stderr)
        sys.exit(1)

    print("DEBUG mode:", base.DEBUG)
    base.load_config(argv[0])
    asyncio.run(main())
