"""Offline voice-native pipeline interfaces and adapters.

The adapters intentionally accept text or bytes and return deterministic values. They
are seams for replacing with a local model later; they never make a network request.
"""

from dataclasses import dataclass
from typing import Protocol

from .retrieval import Hit, Retriever


class Transcriber(Protocol):
    def transcribe(self, audio: bytes | str) -> str: ...


class Synthesizer(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class OfflineTranscriber:
    """A fixture adapter: bytes are decoded as UTF-8 instead of speech-recognized."""

    def transcribe(self, audio: bytes | str) -> str:
        if isinstance(audio, bytes):
            return audio.decode("utf-8").strip()
        if isinstance(audio, str):
            return audio.strip()
        raise TypeError("audio must be UTF-8 bytes or a text fixture")


class OfflineSynthesizer:
    """A fixture adapter: represent speech output as tagged UTF-8 bytes."""

    def synthesize(self, text: str) -> bytes:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return ("OFFLINE_AUDIO:" + text.strip()).encode("utf-8")


@dataclass(frozen=True)
class VoiceResult:
    transcript: str
    hits: tuple[Hit, ...]
    response: str
    audio: bytes


class VoicePipeline:
    """Transcribe → retrieve → respond → synthesize with injectable local adapters."""

    def __init__(
        self,
        retriever: Retriever,
        transcriber: Transcriber | None = None,
        synthesizer: Synthesizer | None = None,
    ):
        self.retriever = retriever
        self.transcriber = transcriber or OfflineTranscriber()
        self.synthesizer = synthesizer or OfflineSynthesizer()

    def run(self, audio: bytes | str, k: int = 3) -> VoiceResult:
        transcript = self.transcriber.transcribe(audio)
        hits = tuple(self.retriever.search(transcript, k=k, method="hybrid"))
        if hits:
            citations = ", ".join(f"[{hit.doc_id}]" for hit in hits)
            response = f"Offline answer grounded in: {citations}."
        else:
            response = "I found no local evidence for that question."
        return VoiceResult(
            transcript=transcript,
            hits=hits,
            response=response,
            audio=self.synthesizer.synthesize(response),
        )
