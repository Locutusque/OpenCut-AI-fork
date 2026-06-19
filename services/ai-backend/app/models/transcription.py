"""Transcription result models.

Produced by the in-process Whisper service (faster-whisper) and returned to
callers with word-level timestamps and per-segment confidence metrics.
"""

from pydantic import BaseModel, Field


class TranscriptionWord(BaseModel):
    """A single word with its timing and recognition probability."""

    word: str = Field(..., description="The recognized word, whitespace-stripped")
    start: float = Field(..., description="Word start time in seconds")
    end: float = Field(..., description="Word end time in seconds")
    probability: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Recognition probability (0-1)"
    )


class TranscriptionSegment(BaseModel):
    """A contiguous transcription segment with optional word-level detail."""

    id: int = Field(..., description="Zero-based segment index")
    text: str = Field(..., description="Segment text")
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    words: list[TranscriptionWord] = Field(
        default_factory=list, description="Word-level timestamps for this segment"
    )
    avg_logprob: float = Field(
        default=0.0, description="Average token log-probability for the segment"
    )
    no_speech_prob: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Probability the segment is non-speech"
    )
    speaker: str | None = Field(
        default=None, description="Speaker label when diarization is applied"
    )


class TranscriptionResult(BaseModel):
    """Full transcription output for an audio file."""

    text: str = Field(default="", description="Concatenated full transcript text")
    segments: list[TranscriptionSegment] = Field(
        default_factory=list, description="Ordered transcription segments"
    )
    language: str = Field(default="", description="Detected/used ISO 639-1 language code")
    duration: float = Field(default=0.0, description="Audio duration in seconds")
