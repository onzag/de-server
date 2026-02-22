# Dreamengine Servers

These servers are meant for single users, either locally or running on a remote server, they do not have any security by default nevertheless.
    
## local-llama.js

Best for machines that cannot fit the entire model in VRAM, it runs a simple nodejs server with llama.cpp

Prepare by calling `npm install`

Test that GPU inference works by doing `node test.js [path-to-config]`

Run the server with `node local-llama.js [path-to-config]`

## local-llama.py

Best for machines that can fit the entire model in VRAM, in fact it will not work if it cannot, as it uses vllm as the backend.

Prepare by calling `python -m pip install -r requirements.txt` or `pip install -r requirements.txt`

Test that it works by doing `python test.py [path-to-config]`

If you get a warning message about missing the tokenizer because of using a gguf file, you can get the tokenizer by calling `save-tokenizer.py [path-to-gguf]` and add that to the
config `tokenizerPath` that should speed thigns up

vllm expect `enforceEager` to be specified, keep it at false for faster inference, true for faster cold starts.

## runpod-serverless.py

Made to be run with runpod.