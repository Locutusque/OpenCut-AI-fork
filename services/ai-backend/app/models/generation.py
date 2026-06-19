"""Generation request models — image generation, prompt enhancement, infographics.

``ImageGenParams`` is forwarded (via ``model_dump()``) to the downstream
image-service, whose schema uses snake_case (``negative_prompt``,
``guidance_scale``). The frontend sends camelCase, so those fields carry
aliases and ``populate_by_name`` accepts either spelling.
"""

from pydantic import BaseModel, ConfigDict, Field


class ImageGenParams(BaseModel):
    """Text-to-image generation parameters."""

    model_config = ConfigDict(populate_by_name=True)

    prompt: str = Field(..., min_length=1, description="Text prompt for image generation")
    negative_prompt: str = Field(
        default="", alias="negativePrompt", description="Negative prompt"
    )
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(
        default=7.5, ge=1.0, le=30.0, alias="guidanceScale"
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    model: str | None = Field(
        default=None, description="Optional model name/id to use for generation"
    )


class EnhancePromptRequest(BaseModel):
    """Request to expand a short prompt into a detailed image-generation prompt."""

    prompt: str = Field(..., min_length=1, description="The original short prompt")
    style: str = Field(default="photorealistic", description="Desired visual style")


class InfographicRequest(BaseModel):
    """Request to render an infographic overlay PNG."""

    topic: str = Field(..., min_length=1, description="Infographic title/topic")
    data_points: list[dict] = Field(
        default_factory=list,
        description="List of {label, value} (or {key, value}) entries to render",
    )
    style: str = Field(default="modern", description="Visual style preset")
    width: int = Field(default=1080, ge=64, le=4096)
    height: int = Field(default=1080, ge=64, le=4096)
    background_color: tuple[int, int, int, int] = Field(
        default=(26, 26, 46, 255),
        description="RGBA background color for the infographic canvas",
    )
