import asyncio
import json
import time
from pathlib import Path

from aiohttp import WSMsgType, web

_ROLE = {"user_transcript": "user", "model_transcript": "model"}

_PAGE = """<!doctype html>
<meta charset="utf-8"><title>Aria x Gemini</title>
<style>
 body{background:#111;color:#eee;font-family:-apple-system,sans-serif;margin:0}
 #wrap{display:flex;height:100vh}
 #left{flex:1;min-width:0;overflow-y:auto;padding:0 16px 40px}
 #left h3{position:sticky;top:0;background:#111;padding:10px 0;margin:0}
 #right{width:min(48vw,680px);background:#0c0c0c;padding:10px 14px;display:flex;flex-direction:column;align-items:center}
 #cam{width:100%;border-radius:8px;display:none}
 .b{padding:8px 12px;border-radius:10px;margin:6px 0;white-space:pre-wrap;line-height:1.4}
 .user{background:#1e4620;margin-left:15%}
 .model{background:#123a5c;margin-right:15%}
 .status{color:#888;font-size:12px;text-align:center}
 h3{text-align:center;color:#aaa;font-weight:normal;margin:6px 0}
 @media (max-width:700px){#wrap{flex-direction:column-reverse;height:auto}#right{width:auto}}
</style>
<div id="wrap">
 <div id="left"><h3>Live transcript</h3><div id="log"></div></div>
 <div id="right"><h3>Aria live preview</h3><img id="cam"></div>
</div>
<script>
const log=document.getElementById('log'),left=document.getElementById('left');
let last=null,lastRole=null;
function add(role,text,replace){
  if(replace&&lastRole===role&&last){last.textContent=text;}
  else{last=document.createElement('div');last.className='b '+role;last.textContent=text;log.appendChild(last);lastRole=role;}
  left.scrollTop=left.scrollHeight;
}
const ws=new WebSocket(`ws://${location.host}/ws`);
ws.onmessage=e=>{const m=JSON.parse(e.data);
  if(m.kind==='line')add(m.role,(m.role==='user'?'You: ':'AI: ')+m.text,m.merged);
  else if(m.kind==='status')add('status',m.text,false);};
ws.onclose=()=>add('status','[disconnected]',false);
const cam=document.getElementById('cam');
function refreshCam(){
  const i=new Image();
  i.onload=()=>{cam.src=i.src;cam.style.display='inline';setTimeout(refreshCam,150);};
  i.onerror=()=>{cam.style.display='none';setTimeout(refreshCam,2000);};
  i.src='/frame.jpg?t='+Date.now();
}
refreshCam();
</script>"""


class TranscriptHub:
    """Collects events; stitches transcript lines; fans out to jsonl + websocket clients."""

    def __init__(self, session_dir: Path | None = None):
        self.t0 = time.monotonic()
        self.frame_provider = None  # set by app when a camera source exists
        self.lines: list[tuple[str, str]] = []
        self._open_role: str | None = None
        self._clients: set[web.WebSocketResponse] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._events_f = None
        if session_dir is not None:
            Path(session_dir).mkdir(parents=True, exist_ok=True)
            self._events_f = open(Path(session_dir) / "events.jsonl", "a")

    # -- events ------------------------------------------------------------
    def emit(self, type: str, text: str = "", detail: dict | None = None) -> None:
        ev = {
            "t_wall": time.time(),
            "t_session": round(time.monotonic() - self.t0, 3),
            "type": type,
            "text": text,
        }
        if detail:
            ev["detail"] = detail
        if self._events_f:
            self._events_f.write(json.dumps(ev) + "\n")
            self._events_f.flush()

        role = _ROLE.get(type)
        if role:
            merged = self._open_role == role and bool(self.lines)
            if merged:
                prev_role, prev_text = self.lines[-1]
                self.lines[-1] = (prev_role, prev_text + text)
            else:
                self.lines.append((role, text))
            self._open_role = role
            self._broadcast({"kind": "line", "role": role, "text": self.lines[-1][1], "merged": merged})
        elif type == "turn_complete":
            self._open_role = None
        elif type in ("session_event", "error"):
            self._open_role = None
            self._broadcast({"kind": "status", "text": text})
        # frame_sent etc: jsonl only

    def save_txt(self, path: Path) -> None:
        with open(path, "w") as f:
            for role, text in self.lines:
                f.write(("You: " if role == "user" else "AI: ") + text + "\n")

    def close(self) -> None:
        if self._events_f:
            self._events_f.close()
            self._events_f = None

    # -- web ---------------------------------------------------------------
    def _broadcast(self, msg: dict) -> None:
        if not self._clients or self._loop is None:
            return
        data = json.dumps(msg)
        for ws in list(self._clients):
            self._loop.create_task(ws.send_str(data))

    async def _ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        # replay existing lines to late joiners
        for role, text in self.lines:
            await ws.send_str(json.dumps({"kind": "line", "role": role, "text": text, "merged": False}))
        self._clients.add(ws)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)
        return ws

    async def _frame_handler(self, request):
        jpeg = self.frame_provider() if self.frame_provider else None
        if not jpeg:
            raise web.HTTPNotFound()
        return web.Response(body=jpeg, content_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

    async def start_server(self, port: int = 8899) -> None:
        self._loop = asyncio.get_running_loop()
        app = web.Application()
        app.router.add_get("/", lambda r: web.Response(text=_PAGE, content_type="text/html"))
        app.router.add_get("/ws", self._ws_handler)
        app.router.add_get("/frame.jpg", self._frame_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
