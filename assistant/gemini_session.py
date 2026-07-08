import asyncio
import traceback

from google import genai
from google.genai import types


class GeminiSession:
    """Owns the Live API connection: sends mic audio and camera frames,
    receives model audio + transcriptions, survives reconnects via session
    resumption + sliding-window compression."""

    def __init__(self, api_key, model, system_prompt, hub, audio_io,
                 get_frame_jpeg=None, on_model_audio=None, voice="Kore"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.voice = voice
        self.hub = hub
        self.audio_io = audio_io
        self.get_frame_jpeg = get_frame_jpeg
        self.on_model_audio = on_model_audio
        self.resume_handle = None
        self.running = True

    def _config(self):
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=self.system_prompt,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice),
                ),
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            session_resumption=types.SessionResumptionConfig(handle=self.resume_handle),
        )

    async def run(self):
        while self.running:
            try:
                async with self.client.aio.live.connect(model=self.model, config=self._config()) as session:
                    self.hub.emit("session_event", text="Gemini connected")
                    send_audio = asyncio.create_task(self._send_audio(session))
                    send_frames = (
                        asyncio.create_task(self._send_frames(session))
                        if self.get_frame_jpeg else None
                    )
                    try:
                        await self._receive(session)  # returns when server ends the connection
                    finally:
                        send_audio.cancel()
                        if send_frames:
                            send_frames.cancel()
                if self.running:
                    self.hub.emit("session_event", text="connection ended; resuming session")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self.running:
                    break
                traceback.print_exc()
                self.hub.emit("error", text=f"session error: {e!r}; reconnecting in 1s")
                await asyncio.sleep(1)

    def stop(self):
        self.running = False

    async def _send_audio(self, session):
        while True:
            chunk = await self.audio_io.mic_queue.get()
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
            )

    async def _send_frames(self, session):
        while True:
            jpeg = self.get_frame_jpeg()
            if jpeg is not None:
                await session.send_realtime_input(
                    video=types.Blob(data=jpeg, mime_type="image/jpeg")
                )
                self.hub.emit("frame_sent")
            await asyncio.sleep(1.0)

    async def _receive(self, session):
        # receive() is a per-turn generator: it ends at each turn_complete.
        # Loop it forever on the same session; a dead connection raises out of
        # here into run()'s reconnect logic.
        while True:
            async for msg in session.receive():
                sru = getattr(msg, "session_resumption_update", None)
                if sru is not None and getattr(sru, "resumable", False) and getattr(sru, "new_handle", None):
                    self.resume_handle = sru.new_handle
                if getattr(msg, "go_away", None) is not None:
                    self.hub.emit("session_event", text="server GoAway; will reconnect")

                sc = getattr(msg, "server_content", None)
                if sc is not None:
                    if getattr(sc, "interrupted", None):
                        self.audio_io.flush_playback()
                        self.hub.emit("session_event", text="[interrupted]")
                    it = getattr(sc, "input_transcription", None)
                    if it is not None and it.text:
                        self.hub.emit("user_transcript", it.text)
                    ot = getattr(sc, "output_transcription", None)
                    if ot is not None and ot.text:
                        self.hub.emit("model_transcript", ot.text)
                    if getattr(sc, "turn_complete", None):
                        self.hub.emit("turn_complete")

                data = getattr(msg, "data", None)
                if data:
                    self.audio_io.play(data)
                    if self.on_model_audio:
                        self.on_model_audio(data)
