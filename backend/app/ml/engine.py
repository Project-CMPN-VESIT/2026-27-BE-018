"""Model loading and large-scene inference (tiled, overlap-blended, CPU or GPU)."""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .architecture import GeoLitterNet
from .preprocess import build_input_stack

log = logging.getLogger(__name__)

MARIDA_CLASSES = [
    "Marine Debris", "Dense Sargassum", "Sparse Sargassum", "Natural Organic",
    "Ship", "Clouds", "Marine Water", "Sediment Water",
    "Foam", "Turbid Water", "Shallow Water", "Waves",
    "Cloud Shadow", "Wakes", "Mixed Water",
]
DEBRIS_CLASS = 0
TILE = 256  # training patch size; also a multiple of 16 as the encoder requires

ProgressFn = Callable[[float, str], None]


@dataclass
class InferenceResult:
    debris_prob: np.ndarray   # (H, W) float32, P(Marine Debris)
    class_map: np.ndarray     # (H, W) uint8, argmax class index
    class_conf: np.ndarray    # (H, W) float32, max softmax probability
    seconds: float
    tiles: int


def _blend_window(size: int) -> np.ndarray:
    """2-D cosine window so overlapping tile predictions fade smoothly into each other."""
    w = np.hanning(size + 2)[1:-1].astype(np.float32)
    w = np.maximum(w, 1e-3)
    return np.outer(w, w)


class ModelEngine:
    def __init__(self, weights_path: Path, device: str = "cpu", threads: int = 0, batch_size: int = 4,
                 overlap: int = 64):
        self.weights_path = Path(weights_path)
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        if threads > 0:
            torch.set_num_threads(threads)
        self.batch_size = max(1, batch_size)
        if not 0 <= overlap < TILE // 2 + 1:
            raise ValueError("overlap must be in [0, 128]")
        self.overlap = overlap
        self._lock = threading.Lock()  # one inference at a time per process (CPU-bound)
        self.model: Optional[GeoLitterNet] = None
        self.sha256 = ""
        self.n_params = 0

    def load(self) -> None:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"model weights not found: {self.weights_path}")
        self.sha256 = hashlib.sha256(self.weights_path.read_bytes()).hexdigest()
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model = GeoLitterNet(num_classes=len(MARIDA_CLASSES))
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device)
        self.model = model
        self.n_params = int(sum(p.numel() for p in model.parameters()))
        log.info("GeoLitterNet loaded: %s params, sha256=%s, device=%s", self.n_params, self.sha256[:12], self.device)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, reflectance_11: np.ndarray, progress: Optional[ProgressFn] = None) -> InferenceResult:
        """reflectance_11: (11, H, W) surface reflectance with NaN for no-data. Any H, W >= 1."""
        if self.model is None:
            raise RuntimeError("model not loaded")
        t0 = time.time()
        x = build_input_stack(reflectance_11)                       # (16, H, W)
        _, H, W = x.shape
        stride = TILE - self.overlap
        # pad so the tile grid covers the whole image (reflect keeps edge statistics natural)
        ph = max(TILE, int(np.ceil(max(H - TILE, 0) / stride)) * stride + TILE)
        pw = max(TILE, int(np.ceil(max(W - TILE, 0) / stride)) * stride + TILE)
        pad_mode = "reflect" if (H > 1 and W > 1 and ph - H < H and pw - W < W) else "constant"
        xp = np.pad(x, ((0, 0), (0, ph - H), (0, pw - W)), mode=pad_mode)

        n_cls = len(MARIDA_CLASSES)
        acc = np.zeros((n_cls, ph, pw), dtype=np.float32)
        wsum = np.zeros((ph, pw), dtype=np.float32)
        win = _blend_window(TILE) if self.overlap > 0 else np.ones((TILE, TILE), np.float32)

        coords = [(r, c) for r in range(0, ph - TILE + 1, stride) for c in range(0, pw - TILE + 1, stride)]
        total = len(coords)
        with self._lock, torch.inference_mode():
            for i in range(0, total, self.batch_size):
                chunk = coords[i:i + self.batch_size]
                batch = np.stack([xp[:, r:r + TILE, c:c + TILE] for r, c in chunk])
                logits = self.model(torch.from_numpy(batch).to(self.device))
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                for (r, c), p in zip(chunk, probs):
                    acc[:, r:r + TILE, c:c + TILE] += p * win
                    wsum[r:r + TILE, c:c + TILE] += win
                if progress:
                    progress(min(i + len(chunk), total) / total, f"tile {min(i + len(chunk), total)}/{total}")

        probs = acc[:, :H, :W] / np.maximum(wsum[:H, :W], 1e-6)
        return InferenceResult(
            debris_prob=probs[DEBRIS_CLASS].astype(np.float32),
            class_map=probs.argmax(0).astype(np.uint8),
            class_conf=probs.max(0).astype(np.float32),
            seconds=time.time() - t0,
            tiles=total,
        )
