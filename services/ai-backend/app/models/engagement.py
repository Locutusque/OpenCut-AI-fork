"""Engagement scoring, clip detection, and YouTube ingestion models.

These models back the YouTube-to-Reels pipeline and the in-editor engagement
scorer. A composite engagement score is assembled from seven sub-signals
(hook, curiosity, energy, audio sync, face presence, emotional arc, virality);
``EngagementScore.to_response()`` produces the exact JSON shape the frontend's
``EngagementScoreResult`` consumes.
"""

from typing import Any

from pydantic import BaseModel, Field

# A neutral sub-score used when a signal can't be computed (service offline,
# missing media, analyzer error). Keeps composites sensible rather than zero.
NEUTRAL = 50.0


# ── Sub-signal scores ────────────────────────────────────────────────


class HookScore(BaseModel):
    """First-3-seconds hook strength breakdown."""

    visual_novelty: float = Field(default=0.0, description="Frame-to-frame motion score (0-100)")
    audio_energy_spike: float = Field(default=0.0, description="Opening vs clip-average energy (0-100)")
    early_face_present: bool = Field(default=False, description="Face detected in the first 3s")
    hook_type: str = Field(default="neutral", description="Classified hook formula")
    hook_type_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    speech_rate: float = Field(default=0.0, description="Words per second in the opening")
    composite: float = Field(default=NEUTRAL, ge=0.0, le=100.0)


class EnergyScore(BaseModel):
    """Audio energy dynamics across the clip."""

    mean_energy: float = Field(default=0.0)
    peak_energy: float = Field(default=0.0)
    energy_variance: float = Field(default=0.0)
    has_dynamic_range: bool = Field(default=False)
    composite: float = Field(default=NEUTRAL, ge=0.0, le=100.0)


class CuriosityScore(BaseModel):
    """Curiosity-gap signals detected in the transcript."""

    has_question: bool = Field(default=False)
    has_bold_claim: bool = Field(default=False)
    has_open_loop: bool = Field(default=False)
    gap_count: int = Field(default=0, ge=0)
    composite: float = Field(default=NEUTRAL, ge=0.0, le=100.0)


class FacePresenceScore(BaseModel):
    """Face presence ratio scored against the 30-40% optimal target."""

    face_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    is_optimal: bool = Field(default=False)
    early_face_present: bool = Field(default=False)
    composite: float = Field(default=NEUTRAL, ge=0.0, le=100.0)


class EmotionalArcScore(BaseModel):
    """Emotional-arc structure across the clip's phases."""

    has_strong_open: bool = Field(default=False)
    has_buildup: bool = Field(default=False)
    has_peak: bool = Field(default=False)
    peak_timestamp: float = Field(default=0.0)
    dominant_emotion: str = Field(default="calm")
    composite: float = Field(default=NEUTRAL, ge=0.0, le=100.0)


class ViralityScore(BaseModel):
    """LLM-predicted viral potential, broken into four 0-25 dimensions."""

    hook_strength: int = Field(default=0, ge=0, le=25)
    shareability: int = Field(default=0, ge=0, le=25)
    emotional_impact: int = Field(default=0, ge=0, le=25)
    standalone_value: int = Field(default=0, ge=0, le=25)
    reason: str = Field(default="")
    suggested_title: str = Field(default="")
    composite: float = Field(default=NEUTRAL, ge=0.0, le=100.0)


class AudioSyncScore(BaseModel):
    """Beat detection and caption-to-beat alignment."""

    bpm: float | None = Field(default=None)
    beat_count: int = Field(default=0, ge=0)
    caption_beat_alignment: float = Field(default=0.0)
    composite: float = Field(default=NEUTRAL, ge=0.0, le=100.0)


class EnhancementSuggestion(BaseModel):
    """An actionable suggestion to improve a weak engagement signal."""

    signal: str = Field(..., description="Which signal this targets (e.g. 'hook')")
    current_score: float = Field(default=0.0, description="Current composite for that signal")
    suggestion: str = Field(..., description="Human-readable recommendation")
    action_type: str = Field(
        default="manual",
        description="How it can be applied: auto_apply, adjust_clip, open_tool, manual",
    )
    expected_impact: str = Field(
        default="medium", description="Estimated impact: high, medium, low"
    )


# ── Composite engagement score ───────────────────────────────────────

# Relative contribution of each sub-signal to the composite (sums to 1.0).
_SIGNAL_WEIGHTS: dict[str, float] = {
    "hook": 0.25,
    "curiosity": 0.15,
    "energy": 0.10,
    "audio_sync": 0.10,
    "face_presence": 0.10,
    "emotional_arc": 0.10,
    "virality": 0.20,
}

# (min composite, letter grade, descriptive label), highest first.
_GRADE_BANDS: list[tuple[float, str, str]] = [
    (85.0, "A", "Excellent"),
    (70.0, "B", "Strong"),
    (55.0, "C", "Decent"),
    (40.0, "D", "Needs work"),
    (0.0, "F", "Poor"),
]


class EngagementScore(BaseModel):
    """Full engagement breakdown for a single clip/video.

    The ``composite``, ``grade`` and ``grade_label`` are derived from the
    sub-signals rather than stored, so they always reflect current values.
    """

    hook: HookScore = Field(default_factory=HookScore)
    energy: EnergyScore = Field(default_factory=EnergyScore)
    curiosity: CuriosityScore = Field(default_factory=CuriosityScore)
    audio_sync: AudioSyncScore = Field(default_factory=AudioSyncScore)
    face_presence: FacePresenceScore = Field(default_factory=FacePresenceScore)
    emotional_arc: EmotionalArcScore = Field(default_factory=EmotionalArcScore)
    virality: ViralityScore = Field(default_factory=ViralityScore)
    suggestions: list[EnhancementSuggestion] = Field(default_factory=list)

    @property
    def composite(self) -> float:
        """Weighted average of the seven sub-signals, clamped to 0-100."""
        total = (
            self.hook.composite * _SIGNAL_WEIGHTS["hook"]
            + self.curiosity.composite * _SIGNAL_WEIGHTS["curiosity"]
            + self.energy.composite * _SIGNAL_WEIGHTS["energy"]
            + self.audio_sync.composite * _SIGNAL_WEIGHTS["audio_sync"]
            + self.face_presence.composite * _SIGNAL_WEIGHTS["face_presence"]
            + self.emotional_arc.composite * _SIGNAL_WEIGHTS["emotional_arc"]
            + self.virality.composite * _SIGNAL_WEIGHTS["virality"]
        )
        return round(max(0.0, min(100.0, total)), 1)

    @property
    def grade(self) -> str:
        composite = self.composite
        for threshold, letter, _label in _GRADE_BANDS:
            if composite >= threshold:
                return letter
        return "F"

    @property
    def grade_label(self) -> str:
        composite = self.composite
        for threshold, _letter, label in _GRADE_BANDS:
            if composite >= threshold:
                return label
        return "Poor"

    def to_response(self) -> dict[str, Any]:
        """Serialize to the JSON shape consumed by the frontend."""
        return {
            "hook": self.hook.model_dump(),
            "curiosity": self.curiosity.model_dump(),
            "energy": self.energy.model_dump(),
            "audio_sync": self.audio_sync.model_dump(),
            "face_presence": self.face_presence.model_dump(),
            "emotional_arc": self.emotional_arc.model_dump(),
            "virality": self.virality.model_dump(),
            "suggestions": [s.model_dump() for s in self.suggestions],
            "composite": self.composite,
            "grade": self.grade,
            "grade_label": self.grade_label,
        }


# ── Scoring requests ─────────────────────────────────────────────────


class ScoreClipRequest(BaseModel):
    """Input for scoring a single clip's engagement."""

    audio_path: str | None = Field(default=None, description="Path to clip audio (WAV)")
    video_path: str | None = Field(default=None, description="Path to clip video")
    transcript_text: str = Field(default="", description="Clip transcript text")
    transcript_segments: list[dict] | None = Field(
        default=None, description="Optional per-segment transcript with word timings"
    )
    start: float = Field(default=0.0, description="Clip start time in seconds")
    end: float = Field(default=0.0, description="Clip end time in seconds")
    title: str = Field(default="", description="Optional clip title for context")


class ScoreBatchRequest(BaseModel):
    """Input for scoring multiple clips in parallel."""

    clips: list[ScoreClipRequest] = Field(default_factory=list)


# ── Clip detection ───────────────────────────────────────────────────


class ScoredClip(BaseModel):
    """A detected clip candidate with its engagement score."""

    index: int = Field(..., description="Zero-based clip index")
    title: str = Field(default="", description="Catchy clip title")
    start: float = Field(..., description="Clip start time in seconds")
    end: float = Field(..., description="Clip end time in seconds")
    transcript_preview: str = Field(default="", description="First ~200 chars of transcript")
    tags: list[str] = Field(default_factory=list)
    engagement: EngagementScore = Field(default_factory=EngagementScore)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


# ── YouTube ingestion ────────────────────────────────────────────────


class YouTubeVideoMeta(BaseModel):
    """Metadata for an ingested YouTube video."""

    video_id: str = Field(..., description="YouTube video id")
    title: str = Field(default="Untitled")
    channel_name: str = Field(default="Unknown")
    channel_id: str = Field(default="")
    duration_seconds: int = Field(default=0, ge=0)
    thumbnail_url: str = Field(default="")
    upload_date: str = Field(default="")
    view_count: int | None = Field(default=None)
    is_live: bool = Field(default=False)
    is_private: bool = Field(default=False)
    warning: str | None = Field(default=None, description="Non-fatal warning (e.g. long video)")


class JobStatus(BaseModel):
    """Status of a background ingestion/processing job."""

    job_id: str = Field(..., description="Job identifier")
    status: str = Field(default="pending", description="pending, running, completed, failed, cancelled")
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = Field(default="")
    result: dict[str, Any] | None = Field(default=None)
    error: str | None = Field(default=None)
