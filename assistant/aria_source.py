import threading

import aria.sdk as aria
import cv2
import numpy as np


class AriaSource:
    """Connects to Aria Gen1, subscribes to the RGB stream, exposes the latest
    frame (rotated upright, RGB channel order) and a downscaled JPEG."""

    def __init__(self, interface="usb", device_ip=None, profile="profile18", on_status=None):
        self.interface = interface
        self.device_ip = device_ip
        self.profile = profile
        self.on_status = on_status or (lambda msg: None)
        self._lock = threading.Lock()
        self._frame = None
        self._device = None
        self._client = None
        self._streaming = False
        self._subscribed = False

    # observer callbacks arrive on SDK threads
    def on_image_received(self, image: np.ndarray, record) -> None:
        if record.camera_id == aria.CameraId.Rgb:
            with self._lock:
                self._frame = np.rot90(image, -1)

    def on_streaming_client_failure(self, reason, message: str) -> None:
        self.on_status(f"Aria stream failure: {reason}: {message}")

    def start(self) -> None:
        aria.set_log_level(aria.Level.Warning)
        self._client = aria.DeviceClient()
        cfg = aria.DeviceClientConfig()
        if self.device_ip:
            cfg.ip_v4_address = self.device_ip
        self._client.set_client_config(cfg)
        self._device = self._client.connect()

        mgr = self._device.streaming_manager
        s_cfg = aria.StreamingConfig()
        s_cfg.profile_name = self.profile
        if self.interface == "usb":
            s_cfg.streaming_interface = aria.StreamingInterface.Usb
        s_cfg.security_options.use_ephemeral_certs = True
        mgr.streaming_config = s_cfg

        sub = mgr.streaming_client.subscription_config
        sub.subscriber_data_type = aria.StreamingDataType.Rgb
        sub.message_queue_size[aria.StreamingDataType.Rgb] = 1
        sub.security_options.use_ephemeral_certs = True
        mgr.streaming_client.subscription_config = sub
        mgr.streaming_client.set_streaming_client_observer(self)

        try:
            mgr.stop_streaming()  # clear stale session left by an unclean shutdown
        except Exception:
            pass
        mgr.start_streaming()
        self._streaming = True
        mgr.streaming_client.subscribe()
        self._subscribed = True
        self.on_status(f"Aria streaming via {self.interface}")

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def latest_jpeg(self, max_dim: int = 768, quality: int = 80) -> bytes | None:
        frame = self.latest_frame()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def stop(self) -> None:
        # best-effort teardown: each step may fail independently (e.g. the
        # WiFi socket is already gone) without blocking the others
        if self._device is None:
            return
        if self._subscribed:
            try:
                self._device.streaming_manager.streaming_client.unsubscribe()
            except Exception:
                pass
        if self._streaming:
            try:
                self._device.streaming_manager.stop_streaming()
            except Exception:
                pass
        try:
            self._client.disconnect(self._device)
        except Exception:
            pass
