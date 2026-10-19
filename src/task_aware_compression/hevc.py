from __future__ import annotations
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .compressor import CompressionResult

def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable: raise RuntimeError("HEVC baseline requires ffmpeg on PATH")
    return executable

def encode_hevc(image: np.ndarray, output: str | Path, crf: int = 35, preset: str = "medium") -> Path:
    executable = _ffmpeg(); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source.png"; Image.fromarray(image[..., :3].astype(np.uint8)).save(source)
        command = [executable, "-y", "-loglevel", "error", "-i", str(source), "-c:v", "libx265", "-preset", preset, "-x265-params", f"crf={int(crf)}", "-frames:v", "1", str(output)]
        subprocess.run(command, check=True)
    return output

def decode_hevc(path: str | Path) -> np.ndarray:
    executable = _ffmpeg()
    command = [executable, "-loglevel", "error", "-i", str(path), "-f", "image2pipe", "-vcodec", "png", "-"]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    from io import BytesIO
    with Image.open(BytesIO(completed.stdout)) as image: return np.asarray(image.convert("RGB"), dtype=np.uint8)


def compress_hevc(image: np.ndarray, crf: int = 35, preset: str = "fast") -> CompressionResult:
    """Encode and decode one image as a single-frame HEVC baseline."""
    image = np.ascontiguousarray(image[..., :3].astype(np.uint8))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "frame.hevc"
        start = time.perf_counter()
        encode_hevc(image, path, crf, preset)
        encode_ms = (time.perf_counter() - start) * 1000
        encoded = path.read_bytes()
        start = time.perf_counter()
        decoded = decode_hevc(path)
        decode_ms = (time.perf_counter() - start) * 1000
    pixels = image.shape[0] * image.shape[1]
    return CompressionResult(decoded, encoded, "hevc", len(encoded), image.nbytes, len(encoded) * 8 / pixels, image.nbytes / max(1, len(encoded)), encode_ms, decode_ms, [crf], None, {"crf": crf, "preset": preset})
