"""Natural-language editor command models.

The frontend posts a command plus an optional snapshot of the timeline state
(camelCase ``timelineState``); the LLM interprets it into a list of structured
``EditorAction`` operations the editor can apply.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CommandRequest(BaseModel):
    """A natural-language editing command from the user."""

    model_config = ConfigDict(populate_by_name=True)

    command: str = Field(..., min_length=1, description="The natural-language command")
    timeline_state: dict[str, Any] | None = Field(
        default=None,
        alias="timelineState",
        description="Optional snapshot of the current timeline for context",
    )
    model: str | None = Field(
        default=None, description="Optional LLM model override"
    )


class EditorAction(BaseModel):
    """A single structured action to apply to the editor timeline."""

    type: str = Field(..., description="Action type, e.g. 'cut', 'add_text', 'trim'")
    target: str | None = Field(
        default=None, description="Target clip/element id, or null for global actions"
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Action-specific parameters"
    )


class CommandResponse(BaseModel):
    """The interpreted result of a natural-language command."""

    actions: list[EditorAction] = Field(
        default_factory=list, description="Ordered actions to apply"
    )
    explanation: str = Field(default="", description="Human-readable explanation")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Model confidence in the interpretation"
    )
    raw_response: str | None = Field(
        default=None, description="Raw LLM JSON response for debugging"
    )
