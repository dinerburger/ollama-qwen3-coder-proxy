import os
import json
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import logging; 

logger=logging.getLogger("proxy")

def _log_pair(tag, path, before, after): logger.debug("%s %s | before=%r after=%r", tag, path, before, after)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TIMEOUT = float(os.getenv("PROXY_TIMEOUT", "600"))
LOG_STREAM_MAX = 100

app = FastAPI(title="Ollama Tool-Parsing Proxy", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Qwen-style <tool_call> parser → Ollama-compatible tool_calls
# ----------------------------------------------------------------------------
_TOOL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
_FUNC_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)


def _normalize_tool_obj(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize a parsed tool object to Ollama tool_calls shape.

    Expected outputs:
      {"function": {"name": str, "arguments": dict}}
    """
    if raw is None:
        return None

    # Handle {"name": ..., "arguments": {...}}
    if isinstance(raw, dict) and "name" in raw:
        args = raw.get("arguments", {})
        # Some models emit arguments as JSON-string
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        return {"function": {"name": raw["name"], "arguments": args}}

    # Handle {"function": {"name":..., "arguments":{...}}}
    if isinstance(raw, dict) and "function" in raw and isinstance(raw["function"], dict):
        fn = raw["function"]
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        return {"function": {"name": fn.get("name"), "arguments": args}}

    return None


def _parse_qwen_xml_body(xml_text: str) -> List[Dict[str, Any]]:
    """Parse a <function=...><parameter=...>...</function> style body into tool_calls.
    Returns a list in Ollama-compatible tool_calls shape.
    """
    out: List[Dict[str, Any]] = []
    for fn_name, fn_body in _FUNC_RE.findall(xml_text):
        params: Dict[str, Any] = {}
        for p_name, p_val in _PARAM_RE.findall(fn_body):
            # Try to coerce JSON values if possible
            val = p_val.strip()
            try:
                params[p_name] = json.loads(val)
            except Exception:
                # fallback to string
                params[p_name] = val
        out.append({"function": {"name": fn_name.strip(), "arguments": params}})
    return out


def parse_tool_calls_and_strip(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract <tool_call> blocks and return (text_without_blocks, tool_calls).

    Each <tool_call> may contain either JSON (preferred by Ollama/Qwen templates)
    or the XML-ish <function=...><parameter=...>...</function> format.
    """
    tool_calls: List[Dict[str, Any]] = []

    def _replace(match: re.Match) -> str:
        inner = match.group(1).strip()
        parsed_any: List[Dict[str, Any]] = []

        # First try JSON (single object or array of objects)
        try:
            obj = json.loads(inner)
            if isinstance(obj, list):
                for item in obj:
                    norm = _normalize_tool_obj(item)
                    if norm:
                        parsed_any.append(norm)
            else:
                norm = _normalize_tool_obj(obj)
                if norm:
                    parsed_any.append(norm)
        except Exception:
            # Fallback to XML-ish function/parameter tags
            parsed_any.extend(_parse_qwen_xml_body(inner))

        tool_calls.extend(parsed_any)
        # strip the whole tool_call block from content
        return ""

    new_text = _TOOL_BLOCK_RE.sub(_replace, text)

    return new_text, tool_calls


# ----------------------------------------------------------------------------
# Streaming transformer
class ToolStreamTransformer:
    """Stateful transformer that removes <tool_call> blocks from streaming
    content and collects parsed tool_calls to attach to outgoing chunks.
    """

    def __init__(self) -> None:
        self._in_block = False
        self._buffer = ""  # buffer for content inside <tool_call> ... </tool_call>
        self._carry = ""   # carry content across chunks
        self.collected: List[Dict[str, Any]] = []

    def process_text_chunk(self, chunk: str) -> Tuple[str, List[Dict[str, Any]]]:
        text = self._carry + (chunk or "")
        emitted: List[str] = []
        i = 0
        while i < len(text):
            if not self._in_block:
                start = text.find("<tool_call>", i)
                if start == -1:
                    # no tag ahead, emit the rest
                    emitted.append(text[i:])
                    break
                # emit text before tag and enter block
                emitted.append(text[i:start])
                i = start + len("<tool_call>")
                self._in_block = True
                self._buffer = ""
            else:
                end = text.find("</tool_call>", i)
                if end == -1:
                    # consume remainder into buffer and wait for next chunk
                    self._buffer += text[i:]
                    i = len(text)
                else:
                    # got a full block
                    self._buffer += text[i:end]
                    # parse full block
                    _, calls = parse_tool_calls_and_strip(f"<tool_call>{self._buffer}</tool_call>")
                    self.collected.extend(calls)
                    # advance past end tag and leave block state
                    i = end + len("</tool_call>")
                    self._in_block = False
                    self._buffer = ""
        # nothing to carry; all consumed
        self._carry = ""
        clean = "".join(emitted)
        return clean, self.flush_new_calls()

    def flush_new_calls(self) -> List[Dict[str, Any]]:
        # Return all collected calls (idempotent per emission is acceptable)
        calls = self.collected
        self.collected = []
        return calls


# ----------------------------------------------------------------------------
# Low-level HTTP helpers
# ----------------------------------------------------------------------------
async def _forward_stream(
    upstream: httpx.Response,
    mutator: Optional[callable] = None,
) -> AsyncGenerator[bytes, None]:
    """(Deprecated in streaming paths where context lifetime matters.)
    Forward a streaming NDJSON response, optionally mutating each JSON line.

    `mutator` receives a Python dict (one JSON object) and must return a dict.
    """
    async for raw_line in upstream.aiter_lines():
        if not raw_line:
            # Forward keep-alive/empty lines
            yield b"\n"
            continue
        try:
            obj = json.loads(raw_line)
        except Exception:
            # Not JSON? forward as-is
            yield (raw_line + "\n").encode()
            continue
        if mutator:
            try:
                obj = mutator(obj)
            except Exception:
                # On any mutation error, fall back to the original object
                pass
        yield (json.dumps(obj, ensure_ascii=False) + "\n").encode()


def _dedupe_tool_calls(existing: Optional[List[Dict[str, Any]]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing = existing or []
    seen = {(tc.get("function", {}).get("name"), json.dumps(tc.get("function", {}).get("arguments", {}), sort_keys=True)) for tc in existing}
    for tc in new:
        sig = (tc.get("function", {}).get("name"), json.dumps(tc.get("function", {}).get("arguments", {}), sort_keys=True))
        if sig not in seen:
            existing.append(tc)
            seen.add(sig)
    return existing


def _to_openai_tool_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in calls or []:
        fn = c.get("function", {})
        name = fn.get("name")
        args = fn.get("arguments", {})
        # OpenAI expects arguments as a JSON string
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except Exception:
                args = str(args)
        out.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
    return out


# ----------------------------------------------------------------------------


def _to_openai_tool_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in calls or []:
        fn = c.get("function", {})
        name = fn.get("name")
        args = fn.get("arguments", {})
        # OpenAI expects arguments as a JSON string
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except Exception:
                args = str(args)
        out.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
    return out


# ----------------------------------------------------------------------------
# Core proxy logic for chat/generate (with response parsing)
# ----------------------------------------------------------------------------
async def _proxy_and_parse(request: Request, path: str) -> Response:
    # Read original body (pass through unchanged)
    body_bytes = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    # Detect streaming flag without mutating payload
    stream_flag = False
    try:
        payload = json.loads(body_bytes or b"{}")
        stream_flag = bool(payload.get("stream"))
    except Exception:
        pass

    url = f"{OLLAMA_BASE_URL}{path}"

    # --- Streaming branch: keep client context inside generator ----------------
    if stream_flag:
        is_sse = path.startswith("/v1/")

        if is_sse:
            async def sse_stream():
                async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                    async with client.stream("POST", url, content=body_bytes, headers=headers) as upstream:
                        transformers: Dict[int, ToolStreamTransformer] = {}
                        log_count = 0
                        async for line in upstream.aiter_lines():
                            if not line:
                                yield b":\n\n"
                                continue
                            if not line.startswith("data: "):
                                yield (line + "\n\n").encode()
                                continue
                            payload_line = line[len("data: ") :]
                            if payload_line.strip() == "[DONE]":
                                yield b"data: [DONE]\n\n"
                                continue
                            try:
                                obj = json.loads(payload_line)
                            except Exception:
                                yield (line + "\n\n").encode()
                                continue
                            # log/intercept before
                            before_text = None
                            choices = obj.get("choices")
                            if isinstance(choices, list):
                                for ch in choices:
                                    idx = int(ch.get("index", 0))
                                    tr = transformers.setdefault(idx, ToolStreamTransformer())
                                    delta = ch.get("delta", {})
                                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                                        if before_text is None:
                                            before_text = delta["content"]
                                        cleaned, new_calls = tr.process_text_chunk(delta["content"]) 
                                        delta["content"] = cleaned
                                        if new_calls:
                                            delta.setdefault("tool_calls", []).extend(_to_openai_tool_calls(new_calls))
                                        ch["delta"] = delta
                                    if ch.get("finish_reason"):
                                        leftover = tr.flush_new_calls()
                                        if leftover:
                                            ch.setdefault("delta", {}).setdefault("tool_calls", []).extend(_to_openai_tool_calls(leftover))
                            if before_text is not None and log_count < LOG_STREAM_MAX:
                                after_text = choices[0].get("delta", {}).get("content", "") if isinstance(choices, list) and choices else ""
                                _log_pair("SSE chunk", path, before_text, after_text)
                                log_count += 1
                            yield ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()
            return StreamingResponse(sse_stream(), media_type="text/event-stream")

        async def ndjson_stream():
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                async with client.stream("POST", url, content=body_bytes, headers=headers) as upstream:
                    transformer = ToolStreamTransformer()
                    async for raw_line in upstream.aiter_lines():
                        if not raw_line:
                            yield b"\n\n"
                            continue
                        try:
                            obj = json.loads(raw_line)
                        except Exception:
                            yield (raw_line + "\n").encode()
                            continue

                        # Chat streaming: message.content
                        msg = obj.get("message")
                        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                            cleaned, new_calls = transformer.process_text_chunk(msg["content"]) 
                            msg["content"] = cleaned
                            if new_calls:
                                msg["tool_calls"] = _dedupe_tool_calls(msg.get("tool_calls"), new_calls)
                            obj["message"] = msg
                            if obj.get("done"):
                                leftover = transformer.flush_new_calls()
                                if leftover:
                                    msg["tool_calls"] = _dedupe_tool_calls(msg.get("tool_calls"), leftover)
                            yield (json.dumps(obj, ensure_ascii=False) + "\n").encode()
                            continue

                        # Generate streaming: response
                        if isinstance(obj.get("response"), str):
                            cleaned, new_calls = transformer.process_text_chunk(obj["response"]) 
                            obj["response"] = cleaned
                            if new_calls:
                                obj["tool_calls"] = _dedupe_tool_calls(obj.get("tool_calls"), new_calls)
                            if obj.get("done"):
                                leftover = transformer.flush_new_calls()
                                if leftover:
                                    obj["tool_calls"] = _dedupe_tool_calls(obj.get("tool_calls"), leftover)
                            yield (json.dumps(obj, ensure_ascii=False) + "\n").encode()
                            continue

                        yield (json.dumps(obj, ensure_ascii=False) + "\n").encode()
        return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")

    # --- Non-streaming branch --------------------------------------------------
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(url, content=body_bytes, headers=headers)
        try:
            data = resp.json()
        except Exception:
            content = await resp.aread()
            return Response(content=content, status_code=resp.status_code, headers=dict(resp.headers))

    # OpenAI-chat style: choices[].message.content
    if isinstance(data, dict) and isinstance(data.get("choices"), list):
        for ch in data["choices"]:
            msg = ch.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                cleaned, calls = parse_tool_calls_and_strip(msg["content"]) 
                msg["content"] = cleaned
                if calls:
                    msg["tool_calls"] = _to_openai_tool_calls(calls)
                ch["message"] = msg
        return JSONResponse(data, status_code=resp.status_code, headers={k: v for k, v in resp.headers.items() if k.lower() == "x-request-id"})

    # Chat-style response (Ollama /api/chat)
    if isinstance(data, dict) and isinstance(data.get("message"), dict) and isinstance(data["message"].get("content"), str):
        cleaned, calls = parse_tool_calls_and_strip(data["message"]["content"])
        data["message"]["content"] = cleaned
        if calls:
            data["message"]["tool_calls"] = _dedupe_tool_calls(data["message"].get("tool_calls"), calls)
        return JSONResponse(data, status_code=resp.status_code, headers={k: v for k, v in resp.headers.items() if k.lower() == "x-request-id"})

    # Generate-style response (Ollama /api/generate)
    if isinstance(data, dict) and isinstance(data.get("response"), str):
        cleaned, calls = parse_tool_calls_and_strip(data["response"])
        data["response"] = cleaned
        if calls:
            data["tool_calls"] = _dedupe_tool_calls(data.get("tool_calls"), calls)
        return JSONResponse(data, status_code=resp.status_code, headers={k: v for k, v in resp.headers.items() if k.lower() == "x-request-id"})

    # Not a recognized JSON payload; forward as-is
    return JSONResponse(data, status_code=resp.status_code, headers=dict(resp.headers))

# ----------------------------------------------------------------------------
# Explicit Ollama API routes (requests passed through unmodified)
# Only responses for /api/chat and /api/generate are parsed for tools.
# ----------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(request: Request) -> Response:
    return await _proxy_and_parse(request, "/api/chat")


@app.post("/api/generate")
async def generate(request: Request) -> Response:
    return await _proxy_and_parse(request, "/api/generate")


# --- OpenAI-compatible endpoint ----------------------------------------------
@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request) -> Response:
    return await _proxy_and_parse(request, "/v1/chat/completions")


# --- Model management & other endpoints (pass-through) -----------------------

async def _passthrough(request: Request, path: str, method: str = "POST") -> Response:
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    async def stream_passthrough():
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream(method, f"{OLLAMA_BASE_URL}{path}", content=body if method != "GET" else None, headers=headers) as resp:
                async for chunk in resp.aiter_raw():
                    yield chunk

    # Use generic octet-stream; the client will parse NDJSON/SSE as needed
    return StreamingResponse(stream_passthrough(), media_type="application/octet-stream")


@app.get("/v1/models")
async def list_models_oai(request: Request) -> Response:
    return await _passthrough(request, "/v1/models", method="GET")

@app.get("/api/tags")
async def list_models(request: Request) -> Response:
    return await _passthrough(request, "/api/tags", method="GET")


@app.get("/api/ps")
async def list_running(request: Request) -> Response:
    return await _passthrough(request, "/api/ps", method="GET")


@app.get("/api/version")
async def version(request: Request) -> Response:
    return await _passthrough(request, "/api/version", method="GET")


@app.post("/api/create")
async def create_model(request: Request) -> Response:
    return await _passthrough(request, "/api/create")


@app.post("/api/show")
async def show_model(request: Request) -> Response:
    return await _passthrough(request, "/api/show")


@app.post("/api/copy")
async def copy_model(request: Request) -> Response:
    return await _passthrough(request, "/api/copy")


@app.post("/api/pull")
async def pull_model(request: Request) -> Response:
    return await _passthrough(request, "/api/pull")


@app.post("/api/push")
async def push_model(request: Request) -> Response:
    return await _passthrough(request, "/api/push")


@app.delete("/api/delete")
async def delete_model(request: Request) -> Response:
    return await _passthrough(request, "/api/delete", method="DELETE")


# Accept POST as well to accommodate clients that cannot send bodies with DELETE
@app.post("/api/delete")
async def delete_model_post(request: Request) -> Response:
    return await _passthrough(request, "/api/delete", method="POST")


# Embeddings endpoints: both legacy and current
@app.post("/api/embed")
async def embed(request: Request) -> Response:
    return await _passthrough(request, "/api/embed")


@app.post("/api/embeddings")
async def embeddings(request: Request) -> Response:
    return await _passthrough(request, "/api/embeddings")


# Catch-all passthrough for any other /api/* routes (kept last)
@app.api_route("/api/{remainder:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def catchall(request: Request, remainder: str) -> Response:
    method = request.method.upper()
    path = f"/api/{remainder}"
    return await _passthrough(request, path, method=method)


# ----------------------------------------------------------------------------
# Run: uvicorn ollama_tool_proxy:app --host 0.0.0.0 --port 8000
# Optionally set OLLAMA_BASE_URL to point at your Ollama server.
# ----------------------------------------------------------------------------
