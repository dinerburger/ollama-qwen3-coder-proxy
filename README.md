# Ollama Qwen3-Coder Proxy

This proxy enables tool calling functionality with the Qwen3 Coder model by converting
Qwen3-Coder-style tool call formats to Ollama-compatible formats.

## Why This Project?

The Qwen3 Coder model uses a specific XML-based tool call format that isn't directly
compatible with Ollama's standard tool calling interface (OpenAI compatible). This proxy translates between these
formats, enabling seamless tool usage with the Qwen3 Coder model.

## Getting Started

To run the proxy:

1. Install dependencies: `pip install -r requirements.txt` (or use `nix-shell`)
2. Start the server: `uvicorn main:app --host 0.0.0.0 --port 8000`
3. Set `OLLAMA_BASE_URL` to point at your Ollama server

## Usage

The proxy works by intercepting requests to Ollama's API endpoints, parsing Qwen-style tool
calls from the responses, and converting them to standard OpenAI tool call format that can be
consumed by clients like [opencode.ai](https://github.com/sst/opencode).

## Features

- Converts Qwen XML-style tool calls to Ollama-compatible JSON format
- Handles both streaming and non-streaming responses
- Supports chat/completions and generate endpoints
- Works seamlessly with opencode.ai for Qwen3 Coder model integration

This proxy is particularly useful when using the Qwen3 Coder model in development
environments that expect standard OpenAI tool calling conventions.

## Example usage

1. Make sure Ollama is running
2. Pull the qwen3-coder model: `ollama pull hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_XL`
3. Launch the proxy
4. Connect opencode to the proxy. Here is an example of config:

```json
  "provider": {
    "ollama-qwen-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "models": {
          "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_XL": {
              "name": "Qwen3-Coder (unsloth)"
          },
      },
      "options": {
        "baseURL": "http://localhost:8000/v1"
      }
    },
```
