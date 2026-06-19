"""Audio / text-to-speech request models.

``TTSRequest`` is accepted by the backend and forwarded (via ``model_dump()``)
to the downstream tts-service, whose request schema uses snake_case field
names (``speaker_wav``). The frontend sends camelCase (``speakerWav``), so the
field carries an alias and ``populate_by_name`` is enabled to accept either.
"""

from pydantic import BaseModel, ConfigDict, Field


class TTSRequest(BaseModel):
    """Text-to-speech generation request."""

    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(..., min_length=1, max_length=5000, description="Text to speak")
    language: str = Field(default="en", description="Language code (ISO 639-1)")
    speaker_wav: str | None = Field(
        default=None,
        alias="speakerWav",
        description="Path to a reference speaker WAV for voice cloning",
    )
    speaker: str | None = Field(
        default=None,
        description="Built-in speaker name (e.g. 'male', 'female')",
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
