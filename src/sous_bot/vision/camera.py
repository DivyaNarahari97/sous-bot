"""Image capture from webcam or static files."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class CameraCapture:
    """Captures images from webcam or loads from files."""

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._cap: cv2.VideoCapture | None = None

    def capture_frame(self) -> bytes:
        """Capture a single JPEG frame from the webcam."""
        if self._cap is None:
            self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {self._device_index}"
            )
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")
        return self._encode_jpeg(frame)

    def release(self) -> None:
        """Release the webcam resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @staticmethod
    def load_image(path: str | Path) -> bytes:
        """Load an image file and return JPEG-encoded bytes."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not decode image: {path}")
        return CameraCapture._encode_jpeg(frame)

    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> bytes:
        """Encode a CV2 frame as JPEG bytes."""
        success, buf = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")
        return buf.tobytes()
