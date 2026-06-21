"""
render_proxy.py — Reverse-proxy server for Render.com (free tier).

WHY THIS EXISTS
---------------
The ZAI sandbox exposes one public URL `https://preview-chat-<chatId>.space-z.ai/`
via the platform's edge router. That edge router ONLY keeps routing while the
chat WebSocket (the user's browser tab) is alive. When the user closes the
tab (or the laptop sleeps, or the browser crashes), the edge router tears down
the route, the public URL dies, and Hermes on the laptop can no longer reach
the GLM proxy — even though the proxy is still running inside the sandbox.

THE FIX
-------
Deploy this file on Render.com (free tier, always-on with a persistent
WebSocket keeping it warm). The sandbox runs `sandbox_connector.py`, which
establishes an OUTBOUND WebSocket to this Render app. That outbound
connection survives the chat-tab-close because it's initiated from inside
the sandbox and the sandbox has unrestricted outbound HTTPS.

The laptop hits Render's public HTTPS URL with normal OpenAI-compatible
requests. Render forwards each request through the WebSocket tunnel to the
sandbox, which proxies it to the local GLM proxy on port 3000 and sends the
response back through the tunnel.

LAPTOP (Hermes)  ──HTTPS──▶  RENDER (this file)  ──WebSocket──▶  SANDBOX  ──HTTP──▶  GLM proxy:3000

DEPLOYMENT (Render.com free tier, ~3 minutes)
---------------------------------------------
1. Push this single file to a GitHub repo (or use Render's "from public
   Git URL" with a Dockerless Python environment).

2. Create a new "Web Service" on Render, pick the repo, set:
     Runtime:        Python 3
     Build Command:  pip install fastapi uvicorn[standard] websockets
     Start Command:  uvicorn render_proxy:app --host 0.0.0.0 --port $PORT
     Instance Type:  Free

3. Add one Environment Variable:
     TUNNEL_SECRET   =   some-long-random-string-you-make-up

4. Deploy. Render gives you a URL like:
     https://your-app-name.onrender.com

5. Tell the sandbox connector the same secret + URL. Done.

USAGE (from the laptop)
-----------------------
Point Hermes at the Render URL exactly like you used to point it at the
preview-*.space-z.ai URL:

    baseURL = https://your-app-name.onrender.com/v1
    apiKey  = any
    model   = glm-5.2

PROTOCOL
--------
- Sandbox opens: wss://your-app.onrender.com/tunnel?secret=<TUNNEL_SECRET>
- Laptop POSTs:  https://your-app.onrender.com/v1/chat/completions  (and /health, /v1/models, /admin/*)
- Render forwards each HTTP request through the tunnel as a JSON message:
    {"type":"req","id":"<uuid>","method":"POST","path":"/v1/chat/completions",
     "headers":{...},"body_b64":"..."}
- Sandbox replies with one or more messages:
    {"type":"res","id":"<uuid>","status":200,"headers":{...},"body_b64":"..."}
  For streaming responses, sandbox sends multiple "chunk" messages then one "end":
    {"type":"chunk","id":"<uuid>","body_b64":"..."}
    {"type":"chunk","id":"<uuid>","body_b64":"..."}
    {"type":"end", "id":"<uuid>"}

SECURITY
--------
- The tunnel is authenticated with TUNNEL_SECRET (Render env var, never in git).
- The laptop endpoint accepts any Bearer token (the upstream GLM proxy does
  the real auth). For extra safety, set LAPTOP_API_KEY on Render to require
  a specific key from your Hermes client.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TUNNEL_SECRET = os.environ.get("TUNNEL_SECRET", "")
LAPTOP_API_KEY = os.environ.get("LAPTOP_API_KEY", "")  # optional; if empty, accept any
PORT = int(os.environ.get("PORT", "10000"))
REQUEST_TIMEOUT_SEC = float(os.environ.get("REQUEST_TIMEOUT_SEC", "120"))
HEARTBEAT_SEC = float(os.environ.get("HEARTBEAT_SEC", "25"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [render_proxy] %(message)s",
)
log = logging.getLogger("render_proxy")

# ---------------------------------------------------------------------------
# Tunnel registry — only one sandbox can be connected at a time
# ---------------------------------------------------------------------------
class TunnelRegistry:
    def __init__(self) -> None:
        self._ws: Optional[WebSocket] = None
        self._lock = asyncio.Lock()
        # pending request_id -> asyncio.Future
        # For non-streaming: future resolves to {"status","headers","body_b64"}
        # For streaming: future is the queue and we set "stream": True
        self._pending: Dict[str, asyncio.Future] = {}
        self._stream_queues: Dict[str, asyncio.Queue] = {}
        self._last_seen: float = 0.0

    async def register(self, ws: WebSocket) -> None:
        async with self._lock:
            # Close any old tunnel (sandbox reconnect)
            if self._ws is not None:
                try:
                    await self._ws.close(code=1000, reason="replaced")
                except Exception:
                    pass
            self._ws = ws
            self._last_seen = time.time()
        log.info("sandbox tunnel registered")

    async def unregister(self, ws: WebSocket) -> None:
        async with self._lock:
            if self._ws is ws:
                self._ws = None
                # fail any pending requests
                for rid, fut in list(self._pending.items()):
                    if not fut.done():
                        fut.set_exception(RuntimeError("tunnel disconnected"))
                    self._pending.pop(rid, None)
                for rid, q in list(self._stream_queues.items()):
                    await q.put(None)  # signal end
                    self._stream_queues.pop(rid, None)
                log.warning("sandbox tunnel disconnected")

    @property
    def connected(self) -> bool:
        return self._ws is not None

    @property
    def last_seen(self) -> float:
        return self._last_seen

    async def call(self, method: str, path: str, headers: Dict[str, str],
                   body: bytes, stream: bool, timeout: float):
        """Send a request through the tunnel. Returns either a dict (non-stream)
        or an async generator yielding bytes (stream)."""
        ws = self._ws
        if ws is None:
            raise RuntimeError("no tunnel connected")

        rid = uuid.uuid4().hex
        if stream:
            q: asyncio.Queue = asyncio.Queue()
            self._stream_queues[rid] = q
        else:
            fut: asyncio.Future = asyncio.Future()
            self._pending[rid] = fut

        msg = {
            "type": "req",
            "id": rid,
            "method": method.upper(),
            "path": path,
            "headers": headers,
            "body_b64": base64.b64encode(body).decode("ascii"),
            "stream": stream,
        }
        await ws.send_text(json.dumps(msg))
        self._last_seen = time.time()

        if stream:
            return self._stream_generator(rid, q, timeout)
        else:
            try:
                return await asyncio.wait_for(fut, timeout=timeout)
            except asyncio.TimeoutError:
                self._pending.pop(rid, None)
                raise RuntimeError("tunnel response timeout")

    async def _stream_generator(self, rid: str, q: asyncio.Queue, timeout: float):
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    log.warning("stream chunk timeout rid=%s", rid)
                    return
                if item is None:
                    return  # end signal
                yield item
        finally:
            self._stream_queues.pop(rid, None)

    async def on_message(self, msg: Dict[str, Any]) -> None:
        """Called when the tunnel sends us a message."""
        t = msg.get("type")
        rid = msg.get("id")
        if t == "res":
            fut = self._pending.pop(rid, None)
            if fut and not fut.done():
                fut.set_result({
                    "status": int(msg.get("status", 502)),
                    "headers": msg.get("headers", {}),
                    "body_b64": msg.get("body_b64", ""),
                })
        elif t == "chunk":
            q = self._stream_queues.get(rid)
            if q is not None:
                body_b64 = msg.get("body_b64", "")
                await q.put(base64.b64decode(body_b64) if body_b64 else b"")
        elif t == "end":
            q = self._stream_queues.pop(rid, None)
            if q is not None:
                await q.put(None)
        elif t == "error":
            err = msg.get("error", "unknown")
            fut = self._pending.pop(rid, None)
            if fut and not fut.done():
                fut.set_exception(RuntimeError(err))
            q = self._stream_queues.pop(rid, None)
            if q is not None:
                await q.put(None)
        elif t == "heartbeat":
            self._last_seen = time.time()


registry = TunnelRegistry()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="GLM Tunnel Proxy", docs_url=None, redoc_url=None)


@app.get("/")
async def root():
    return {
        "service": "render-tunnel-proxy",
        "tunnel_connected": registry.connected,
        "last_seen_ago_sec": (time.time() - registry.last_seen) if registry.last_seen else None,
        "endpoints": ["/health", "/v1/models", "/v1/chat/completions", "/admin/*"],
    }


@app.get("/tunnel-status")
async def tunnel_status():
    return {
        "connected": registry.connected,
        "last_seen_ago_sec": (time.time() - registry.last_seen) if registry.last_seen else None,
        "pending_requests": len(registry._pending),
        "active_streams": len(registry._stream_queues),
    }


# Tunnel endpoint — sandbox connects here
@app.websocket("/tunnel")
async def tunnel_endpoint(ws: WebSocket):
    # Auth via query string secret
    secret = ws.query_params.get("secret", "")
    if not TUNNEL_SECRET:
        await ws.close(code=1008, reason="TUNNEL_SECRET not set on server")
        return
    if secret != TUNNEL_SECRET:
        log.warning("tunnel auth failed")
        await ws.close(code=1008, reason="bad secret")
        return

    await ws.accept()
    await registry.register(ws)

    # Heartbeat task — periodically ping the sandbox
    async def heartbeat():
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_SEC)
                await ws.send_text(json.dumps({"type": "heartbeat", "ts": time.time()}))
        except Exception:
            return

    hb_task = asyncio.create_task(heartbeat())

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            await registry.on_message(msg)
    except WebSocketDisconnect:
        log.info("sandbox WebSocket disconnected")
    except Exception as e:
        log.warning("tunnel error: %s", e)
    finally:
        hb_task.cancel()
        await registry.unregister(ws)


# Laptop-facing endpoints — everything is forwarded to the sandbox GLM proxy
async def _forward(request: Request, path: str, stream: bool = False):
    # Auth check (optional)
    if LAPTOP_API_KEY:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {LAPTOP_API_KEY}" and auth != LAPTOP_API_KEY:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    if not registry.connected:
        return JSONResponse(
            {"error": "no sandbox tunnel connected", "hint": "start sandbox_connector.py on the sandbox"},
            status_code=503,
        )

    # Build headers to forward (strip hop-by-hop and auth we already checked)
    fwd_headers = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "transfer-encoding", "connection",
                  "authorization", "accept-encoding"):
            continue
        fwd_headers[k] = v
    # Always set Authorization: Bearer any (the GLM proxy accepts any)
    fwd_headers["authorization"] = "Bearer any"

    body = await request.body()

    try:
        if stream:
            gen = await registry.call(
                method=request.method,
                path=path,
                headers=fwd_headers,
                body=body,
                stream=True,
                timeout=REQUEST_TIMEOUT_SEC,
            )

            async def stream_sse():
                try:
                    async for chunk in gen:
                        if chunk:
                            yield chunk
                except Exception as e:
                    log.warning("stream error: %s", e)

            return StreamingResponse(stream_sse(), media_type="text/event-stream")

        else:
            result = await registry.call(
                method=request.method,
                path=path,
                headers=fwd_headers,
                body=body,
                stream=False,
                timeout=REQUEST_TIMEOUT_SEC,
            )
            body_bytes = base64.b64decode(result["body_b64"]) if result["body_b64"] else b""
            resp_headers = {k: v for k, v in result["headers"].items()
                            if k.lower() not in ("content-length", "transfer-encoding",
                                                  "content-encoding", "connection")}
            return Response(content=body_bytes, status_code=result["status"], headers=resp_headers)

    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    except Exception as e:
        log.exception("forwarding error")
        return JSONResponse({"error": "internal", "detail": str(e)}, status_code=500)


# Laptop-facing routes (mirror the GLM proxy's routes)
@app.get("/health")
async def health(request: Request):
    return await _forward(request, "/health")


@app.get("/v1/models")
async def list_models(request: Request):
    return await _forward(request, "/v1/models")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # Peek at the body to detect streaming
    body = await request.body()
    try:
        j = json.loads(body) if body else {}
        stream = bool(j.get("stream", False))
    except Exception:
        stream = False
    # Re-wrap body into a fresh Request-like object — easier: pass body through
    # We'll re-use _forward by re-injecting body via a tiny shim
    return await _forward_with_body(request, "/v1/chat/completions", body, stream)


async def _forward_with_body(request: Request, path: str, body: bytes, stream: bool):
    if LAPTOP_API_KEY:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {LAPTOP_API_KEY}" and auth != LAPTOP_API_KEY:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    if not registry.connected:
        return JSONResponse(
            {"error": "no sandbox tunnel connected", "hint": "start sandbox_connector.py on the sandbox"},
            status_code=503,
        )

    fwd_headers = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "transfer-encoding", "connection",
                  "authorization", "accept-encoding"):
            continue
        fwd_headers[k] = v
    fwd_headers["authorization"] = "Bearer any"
    if request.method.upper() in ("POST", "PUT", "PATCH"):
        fwd_headers["content-type"] = request.headers.get("content-type", "application/json")

    method = request.method.upper()
    try:
        if stream:
            gen = await registry.call(
                method=method, path=path, headers=fwd_headers, body=body,
                stream=True, timeout=REQUEST_TIMEOUT_SEC,
            )

            async def stream_sse():
                try:
                    async for chunk in gen:
                        if chunk:
                            yield chunk
                except Exception as e:
                    log.warning("stream error: %s", e)

            return StreamingResponse(stream_sse(), media_type="text/event-stream")
        else:
            result = await registry.call(
                method=method, path=path, headers=fwd_headers, body=body,
                stream=False, timeout=REQUEST_TIMEOUT_SEC,
            )
            body_bytes = base64.b64decode(result["body_b64"]) if result["body_b64"] else b""
            resp_headers = {k: v for k, v in result["headers"].items()
                            if k.lower() not in ("content-length", "transfer-encoding",
                                                  "content-encoding", "connection")}
            return Response(content=body_bytes, status_code=result["status"], headers=resp_headers)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    except Exception as e:
        log.exception("forwarding error")
        return JSONResponse({"error": "internal", "detail": str(e)}, status_code=500)


# Admin passthrough (everything under /admin/* and /v1/sessions/*)
@app.api_route("/admin/{rest:path}", methods=["GET", "POST", "DELETE"])
async def admin_passthrough(request: Request, rest: str):
    body = await request.body()
    try:
        j = json.loads(body) if body else {}
        stream = bool(j.get("stream", False))
    except Exception:
        stream = False
    return await _forward_with_body(request, f"/admin/{rest}", body, stream)


@app.api_route("/v1/sessions/{rest:path}", methods=["GET", "DELETE"])
async def sessions_passthrough(request: Request, rest: str):
    body = await request.body()
    return await _forward_with_body(request, f"/v1/sessions/{rest}", body, stream=False)


if __name__ == "__main__":
    log.info("starting render_proxy on port %s", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
