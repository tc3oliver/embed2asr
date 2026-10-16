"""Embed2ASR: speech recognition from frozen EmbeddingGemma 2 temporal audio features + a GRU-CTC reader."""

__version__ = "0.1.0"


def __getattr__(name):  # lazy: `import embed2asr.metrics` must not pull in torch
    if name in ("Embed2ASR", "Transcript"):
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
