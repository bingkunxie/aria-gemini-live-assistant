import argparse
import asyncio
import datetime
import os
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

from assistant.audio_io import AudioIO
from assistant.gemini_session import GeminiSession
from assistant.prompts import build_prompt
from assistant.transcript_hub import TranscriptHub

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(prog="assistant")
    p.add_argument("--audio-test", action="store_true", help="voice only, no glasses")
    p.add_argument("--interface", choices=["usb", "wifi"], default="usb")
    p.add_argument("--device-ip", help="glasses IP for wifi streaming")
    p.add_argument("--record", action="store_true")
    p.add_argument("--task", default="lemonade", choices=["lemonade", "chat"])
    p.add_argument("--audio-device", default="AirPods")
    p.add_argument("--echo-guard", type=float, default=0.0, metavar="SECONDS",
                   help="mute mic to Gemini while the speaker plays, plus this hangover "
                        "(e.g. 0.4). Use with built-in speakers to stop Gemini hearing "
                        "itself; leave 0 with headphones so barge-in still works.")
    return p.parse_args()


async def run(args):
    load_dotenv(ROOT / ".env")
    api_key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview")
    loop = asyncio.get_running_loop()

    session_dir = None
    recorder = None
    if args.record:
        from assistant.recorder import Recorder

        session_dir = ROOT / "sessions" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        recorder = Recorder(session_dir)

    hub = TranscriptHub(session_dir=session_dir)
    await hub.start_server()
    print("transcript window: http://localhost:8899")
    webbrowser.open("http://localhost:8899")

    audio = AudioIO(loop, prefer=args.audio_device,
                    on_mic_chunk=recorder.add_mic if recorder else None,
                    echo_guard=args.echo_guard)
    audio.start()

    aria_src = None
    if not args.audio_test:
        from assistant.aria_source import AriaSource

        aria_src = AriaSource(
            interface=args.interface, device_ip=args.device_ip,
            on_status=lambda m: hub.emit("session_event", text=m),
        )
        aria_src.start()
        hub.frame_provider = aria_src.latest_jpeg

    task = args.task if not args.audio_test else "chat"
    gem = GeminiSession(
        api_key, model, build_prompt(task), hub, audio,
        get_frame_jpeg=aria_src.latest_jpeg if aria_src else None,
        on_model_audio=recorder.add_model_audio if recorder else None,
        voice=os.environ.get("GEMINI_VOICE", "Kore"),
    )

    tasks = [asyncio.create_task(gem.run())]
    if recorder and aria_src:
        async def frame_loop():
            # composite/encode work runs off the event loop; lines snapshot is
            # taken on the loop thread before handing off
            while True:
                await loop.run_in_executor(
                    None, recorder.add_frame, aria_src.latest_frame(), list(hub.lines)
                )
                await asyncio.sleep(0.5 / recorder.FPS)

        tasks.append(asyncio.create_task(frame_loop()))

    print("Running — press Ctrl-C to stop.")
    try:
        await asyncio.gather(*tasks)
    finally:
        gem.stop()
        for t in tasks:
            t.cancel()
        # each cleanup step is isolated: a dead WiFi socket in aria stop must
        # never prevent the recording from being finalized
        steps = [audio.stop]
        if aria_src:
            steps.append(aria_src.stop)
        if recorder:
            steps.append(recorder.finalize)
            steps.append(lambda: hub.save_txt(session_dir / "transcript.txt"))
        for step in steps:
            try:
                step()
            except Exception as e:
                print(f"cleanup error (continuing): {e!r}")
        if recorder:
            print(f"session artifacts: {session_dir}")
        hub.close()


def main():
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nstopped.")
