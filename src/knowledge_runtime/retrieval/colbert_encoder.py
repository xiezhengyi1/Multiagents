from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

BGE_MODEL_NAME = "BAAI/bge-m3"
_HF_HOME = Path(
    str(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")).strip()
).expanduser()
_HF_HUB_ROOT = _HF_HOME / "hub"


def _resolve_local_path(model_name: str) -> Optional[str]:
    repo_id = model_name.replace("/", "--")
    snapshots_dir = _HF_HUB_ROOT / f"models--{repo_id}" / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0])
    return None


class ColBERTEncoder:
    """Generate per-token embeddings for ColBERT-style late interaction retrieval.

    Uses the underlying transformer model directly to extract last_hidden_state,
    then L2-normalizes each token vector — matching FlagEmbedding's colbert_vecs output.
    """

    def __init__(
        self,
        model_name: str = BGE_MODEL_NAME,
        device: Optional[str] = None,
    ):
        from transformers import AutoModel, AutoTokenizer

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        local = _resolve_local_path(model_name)
        source = local if local else model_name
        self.model = AutoModel.from_pretrained(source).to(self._device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self._model_name = model_name

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def model_name(self) -> str:
        return self._model_name

    @torch.no_grad()
    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        results: List[np.ndarray] = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            tokens = {k: v.to(self._device) for k, v in tokens.items()}
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state.float()
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            for j in range(len(batch)):
                mask = tokens["attention_mask"][j].bool()
                emb = embeddings[j][mask]
                emb = emb[1:-1]
                if emb.shape[0] == 0:
                    emb = embeddings[j][mask][:1]
                results.append(emb.cpu().numpy().astype(np.float32))
        return results

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_queries([query], batch_size=1)[0]

    @torch.no_grad()
    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        results: List[np.ndarray] = []
        indices = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm

            indices = tqdm(
                list(indices),
                desc="Encoding documents",
            )

        for i in indices:
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            tokens = {k: v.to(self._device) for k, v in tokens.items()}
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state.float()
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            for j in range(len(batch)):
                mask = tokens["attention_mask"][j].bool()
                emb = embeddings[j][mask]
                emb = emb[1:-1]
                if emb.shape[0] == 0:
                    emb = embeddings[j][mask][:1]
                results.append(emb.cpu().numpy().astype(np.float32))
        return results

    def encode_single_document(self, text: str) -> np.ndarray:
        return self.encode_documents([text], show_progress=False)[0]


@lru_cache(maxsize=1)
def get_colbert_encoder() -> ColBERTEncoder:
    return ColBERTEncoder()
