"""
rag/onnx_embedder.py

A standalone re-implementation of chromadb's ONNXMiniLM_L6_V2 embedding
function (same model, same download URL/hash, same tokenization + mean
pooling + normalization math - verified byte-for-byte identical output),
but WITHOUT importing the `chromadb` package at all.

Why this matters on a 1GB RAM instance (Oracle's free E2.1.Micro shape):
importing `chromadb` - even just to reach the one small submodule that has
the embedding function class - runs chromadb's package __init__, which
pulls in its full client/API surface (telemetry, pydantic settings, etc.)
and adds real resident memory and import time for functionality this app
never uses (we don't store anything in Chroma anymore - MongoDB Atlas
does that job now). This module only needs numpy + onnxruntime +
tokenizers, all of which are lightweight and already required either way.

Model: all-MiniLM-L6-v2, downloaded once and cached on disk (see
rag/embeddings.py for the CACHE_DIR override), never re-downloaded after
that as long as the disk persists - true on Oracle Cloud, unlike Render's
free tier.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import List

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
EXTRACTED_FOLDER_NAME = "onnx"
ARCHIVE_FILENAME = "onnx.tar.gz"
MODEL_DOWNLOAD_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
MAX_TOKENS = 256


def _sha256_matches(fname: str, expected: str) -> bool:
    if not os.path.exists(fname):
        return False
    h = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest() == expected


class ONNXEmbedder:
    """Chroma-compatible callable: `embedder(list_of_strings) -> list[np.ndarray]`."""

    def __init__(self, download_path: str):
        self.download_path = Path(download_path)
        self._tokenizer = None
        self._session = None

    # ── model download (once, cached on disk) ──────────────────────────────
    def _model_files_present(self) -> bool:
        extracted = self.download_path / EXTRACTED_FOLDER_NAME
        required = ["model.onnx", "tokenizer.json", "config.json", "vocab.txt"]
        return all((extracted / f).exists() for f in required)

    def _download_if_needed(self) -> None:
        if self._model_files_present():
            return

        self.download_path.mkdir(parents=True, exist_ok=True)
        archive_path = self.download_path / ARCHIVE_FILENAME

        if not _sha256_matches(str(archive_path), MODEL_SHA256):
            print(f"⬇ Downloading {MODEL_NAME} ONNX model (~90MB, one-time)...")
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    req = urllib.request.Request(
                        MODEL_DOWNLOAD_URL,
                        headers={"User-Agent": "lumi-onnx-embedder/1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp, open(archive_path, "wb") as out:
                        while True:
                            chunk = resp.read(1 << 16)
                            if not chunk:
                                break
                            out.write(chunk)
                    if _sha256_matches(str(archive_path), MODEL_SHA256):
                        break
                    archive_path.unlink(missing_ok=True)
                    last_err = ValueError("downloaded file did not match expected SHA256")
                except Exception as e:  # noqa: BLE001 - retry on any transient network error
                    last_err = e
            else:
                raise RuntimeError(
                    f"Failed to download {MODEL_NAME} ONNX model after 3 attempts: {last_err}"
                )

        with tarfile.open(str(archive_path), "r:gz") as tar:
            try:
                tar.extractall(path=str(self.download_path), filter="data")
            except TypeError:
                # Python < 3.12 doesn't support the `filter` kwarg
                tar.extractall(path=str(self.download_path))

    # ── lazy-loaded tokenizer / onnx session ───────────────────────────────
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(
                str(self.download_path / EXTRACTED_FOLDER_NAME / "tokenizer.json")
            )
            tok.enable_truncation(max_length=MAX_TOKENS)
            tok.enable_padding(pad_id=0, pad_token="[PAD]", length=MAX_TOKENS)
            self._tokenizer = tok
        return self._tokenizer

    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(self.download_path / EXTRACTED_FOLDER_NAME / "model.onnx"),
                providers=ort.get_available_providers(),
                sess_options=so,
            )
        return self._session

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=1)
        norm[norm == 0] = 1e-12
        return v / norm[:, np.newaxis]

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        encoded = [self.tokenizer.encode(t) for t in texts]
        for e in encoded:
            if len(e.ids) > MAX_TOKENS:
                raise ValueError(f"Document length {len(e.ids)} exceeds max tokens {MAX_TOKENS}")

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        last_hidden_state = outputs[0]

        mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
        summed = np.sum(last_hidden_state * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts

        return self._normalize(pooled).astype(np.float32)

    def __call__(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        self._download_if_needed()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.append(self._encode_batch(batch))
        combined = np.concatenate(all_embeddings) if all_embeddings else np.zeros((0, 384), dtype=np.float32)
        return [row for row in combined]
