"""Tests for local embedder progress handling."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from clean.core.config import EmbedderConfig
from clean.embedding.local import SentenceTransformerEmbedder


class _FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


def test_embedder_disables_progress_env_before_model_import(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )

    embedder = SentenceTransformerEmbedder(EmbedderConfig(show_progress_bar=False))

    model = embedder._get_model()

    assert isinstance(model, _FakeSentenceTransformer)
    assert model.model_name == "all-MiniLM-L6-v2"
    assert (
        sys.modules["sentence_transformers"].SentenceTransformer
        is _FakeSentenceTransformer
    )
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TQDM_DISABLE"] == "1"
