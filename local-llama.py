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
import secrets
import ssl
import sys
import websockets
import base
from urllib.parse import urlparse, parse_qs
import os

PORT = 8765
HOST = '0.0.0.0'

# get the DEV=1 environment variable to make a simpler secret for development
DEV = os.getenv("DEV", "0") == "1"

async def handle_client(websocket):
    print('Client connected')

    await websocket.send(json.dumps({"type": "ready", "message": "Model is ready", "context_window_size": base.CONTEXT_WINDOW_SIZE, "supports_parallel_requests": True}))

    loop = asyncio.get_event_loop()
    executor = None  # Use default executor (main thread)
    try:
        requestid_to_rid = {}  # Map internal request IDs to client rids for cancellation
        async for message in websocket:
            try:
                data = json.loads(message)
                rid = data.get('rid', 'no-rid')  # Use provided rid or default to 'no-rid'
                internal_request_id = None  # To track the internal request ID for this action, if applicable

                if (base.DEBUG):
                    print(f"Received message from client: {data}")

                payload = data['payload'] if 'payload' in data else None

                # Handle different actions
                if data.get('action') == 'infer':
                    if not payload:
                        raise ValueError("Invalid payload for infer")
                    
                    async for result in base.generate_completion(payload):
                        if 'error' in result:
                            loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "error", "rid": rid, "message": str(result['error'])})))
                            if internal_request_id is not None:
                                requestid_to_rid.pop(internal_request_id, None)  # Remove from active requests on error
                        elif 'token' in result:
                            loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "token", "rid": rid, "text": result['token']})))
                        elif 'done' in result and result['done']:
                            loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "done", "rid": rid})))
                            if internal_request_id is not None:
                                requestid_to_rid.pop(internal_request_id, None)  # Remove from active requests on done
                        elif 'request_id' in result:
                            internal_request_id = result['request_id']
                            requestid_to_rid[internal_request_id] = rid  # Track active request ID
                elif data.get('action') == 'analyze-prepare':
                    if not payload:
                        raise ValueError("Invalid payload for analyze-prepare")

                    async for result in base.prepare_analysis(payload):
                        if 'error' in result:
                            loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "error", "rid": rid, "message": str(result['error'])})))
                            if internal_request_id is not None:
                                requestid_to_rid.pop(internal_request_id, None)  # Remove from active requests on error
                        elif 'done' in result and result['done']:
                            loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "analyze-ready", "rid": rid})))
                            if internal_request_id is not None:
                                requestid_to_rid.pop(internal_request_id, None)  # Remove from active requests on done
                    
                elif data.get('action') == 'analyze-question':
                    if not payload:
                        raise ValueError("Invalid payload for analyze-question")
                    
                    async for result in base.run_question(payload):
                        if 'error' in result:
                            loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "error", "rid": rid, "message": str(result['error'])})))
                            if internal_request_id is not None:
                                requestid_to_rid.pop(internal_request_id, None)  # Remove from active requests on error
                        elif 'answer' in result:
                            loop.call_soon_threadsafe(asyncio.create_task, websocket.send(json.dumps({"type": "answer", "rid": rid, "text": result['answer']})))
                            if result.get('done'):
                                if internal_request_id is not None:
                                    requestid_to_rid.pop(internal_request_id, None)  # Remove from active requests on done
                        elif 'request_id' in result:
                            internal_request_id = result['request_id']
                            requestid_to_rid[internal_request_id] = rid  # Track active request ID
                elif data.get('action') == 'count-tokens':
                    if not payload or not isinstance(payload.get('text'), str):
                        raise ValueError("Invalid payload for count-tokens")
                    text = payload['text']
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

                    for key, value in requestid_to_rid.items():
                        if value == cancel_rid:
                            internal_request_id = key
                            break

                    if internal_request_id is not None:
                        base.MODEL.abort_request(internal_request_id)
                        requestid_to_rid.pop(internal_request_id, None)  # Remove from active requests
            except Exception as e:
                print(str(e))
                await websocket.send(json.dumps({"type": "error", "rid": rid, "message": str(e)}))
    except websockets.ConnectionClosedOK:
        print('Client disconnected normally')
        for internalrid in requestid_to_rid.keys():
            base.MODEL.abort_request(internalrid)  # Stop the requests if the client disconnects
    except websockets.ConnectionClosedError as e:
        print(f'Client disconnected abnormally: code={e.code} reason={e.reason}')
        for internalrid in requestid_to_rid.keys():
            base.MODEL.abort_request(internalrid)  # Stop the requests if the client disconnects

async def main():
    ssl_context = None
    try:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
        print("SSL context created with cert.pem and key.pem")
    except Exception as e:
        print(f"Failed to create SSL context: {e}")
        sys.exit(1)
        
    server = await websockets.serve(handle_client, HOST, PORT, process_request=process_request, ssl=ssl_context)
    await server.serve_forever()

async def process_request(connection, request):
    parsed = urlparse(request.path)
    query_params = parse_qs(parsed.query)
    secret = query_params.get('secret', [None])[0]

    if not DEV:
        # read the local ./secret file for the expected API key
        try:
            with open("./secret", "r") as f:
                expected_secret = f.read().strip()
        except Exception as e:
            # create a new random API secret and save it to the file
            import secrets
            expected_secret = secrets.token_hex(64)
            with open("./secret", "w") as f:
                f.write(expected_secret)
            print(f"Generated new Secret key and saved to ./secret: {expected_secret}")
    else:
        expected_secret = "dev-secret-12345678900abcdef"  # Simple secret for development mode

    if secret != expected_secret:
        print(f"Unauthorized connection attempt with invalid secret")
        return (401, [], b'Unauthorized')  # Return 401 Unauthorized if the secret key is incorrect

    # Returning None continues the WebSocket handshake
    return None

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 1:
        print("Please provide a config path as the first argument.", file=sys.stderr)
        sys.exit(1)

    if not DEV:
        try:
            with open("./secret", "r") as f:
                print("Using Secret key from ./secret for authentication:")
        except Exception as e:
            print("No existing secret key found. A new one will be generated and saved to ./secret.")
            
            expected_secret = secrets.token_hex(64)
            with open("./secret", "w") as f:
                f.write(expected_secret)
            print(f"Generated new Secret key and saved to ./secret: {expected_secret}")

    print("DEBUG mode:", base.DEBUG)
    print("DEV mode:", DEV)
    base.load_config(argv[0])
    asyncio.run(main())
