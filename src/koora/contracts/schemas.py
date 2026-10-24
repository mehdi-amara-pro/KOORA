"""
Shared data contracts for the Inside the Game hackathon project.

Three contracts connect the four core layers:

    1. Event schema       simulator / video adapter  ->  stats engine
    2. Story atom schema  stats engine               ->  agent team
    3. Insight schema     stats engine + agents      ->  web app, Teams, overlay API

Assumptions (keep in sync with the README):
    - All data is synthetic. "Live" means a simulated feed replayed at match speed.
    - Pitch is 105 x 68 m. Origin (0, 0) is the corner at the home team's
      defensive left in period 1. x runs along the length, y across the width.
      Coordinates never flip at half time; `attacking_direction` says which way
      each team attacks in each period.
    - Tracking frames arrive at 25 Hz. Events arrive as they happen.
    - Every record carries provenance: where it came from and how sure we are.
      Video-derived data has lower confidence and may have gaps.
    - Times are match-clock milliseconds since the period started, plus the
      period number. Wall-clock time is kept for latency measurement only.
    - Event Hubs partition key = match_id, so many matches can run in parallel.

Requires: pydantic >= 2.6
Run `python schemas.py` to validate the examples and export JSON Schema files.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1.0"

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class Contract(BaseModel):
    """Base for all contract models: strict about unknown fields."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Source(str, Enum):
    SIMULATOR = "simulator"  # ground truth from our match simulator
    TRACKING = "tracking"  # optical tracking feed (production assumption)
    VIDEO = "video"  # estimated by the auto-eventing / CV pipeline
    DERIVED = "derived"  # computed by the stats engine from other records


class Provenance(Contract):
    source: Source
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="0 = guess, 1 = certain")
    model_version: str | None = Field(
        None, description="Version of the simulator, CV model or stats rule that produced this"
    )


class MatchClock(Contract):
    period: Literal[1, 2, 3, 4, 5] = Field(..., description="1-2 halves, 3-4 extra time, 5 penalties")
    period_ms: int = Field(..., ge=0, description="Milliseconds since the period kicked off")
    wall_time: datetime | None = Field(None, description="UTC time the record was produced")

    @property
    def display(self) -> str:
        """Broadcast-style clock, e.g. 45+2' is shown as 47'."""
        base = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}[self.period]
        return f"{base + self.period_ms // 60000 + 1}'"


class Point(Contract):
    x: float = Field(..., ge=-5.0, le=PITCH_LENGTH_M + 5.0, description="metres along the pitch")
    y: float = Field(..., ge=-5.0, le=PITCH_WIDTH_M + 5.0, description="metres across the pitch")


class Point3D(Point):
    z: float = Field(0.0, ge=0.0, le=50.0, description="metres above the ground")


# ---------------------------------------------------------------------------
# 1. EVENT SCHEMA  (simulator / video adapter -> stats engine)
# ---------------------------------------------------------------------------


class Position(str, Enum):
    GK = "GK"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


class Player(Contract):
    player_id: str
    team_id: str
    name: str = Field(..., description="Fictional full name, used for on-screen tags")
    short_name: str = Field(..., description="Name for tight overlays, e.g. 'K. Mensah'")
    shirt_number: int = Field(..., ge=1, le=99)
    position: Position
    starter: bool = True


class Team(Contract):
    team_id: str
    name: str = Field(..., description="Fictional club name, never a real club")
    short_name: str = Field(..., max_length=4, description="Three-letter code for score bugs")
    primary_colour: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")
    secondary_colour: str = Field(..., pattern=r"^#[0-9A-Fa-f]{6}$")
    players: list[Player]


class MatchMetadata(Contract):
    match_id: str
    competition: str
    kickoff: datetime
    venue: str
    home: Team
    away: Team
    attacking_direction: dict[str, dict[int, Literal["+x", "-x"]]] = Field(
        ..., description="team_id -> period -> direction of attack"
    )
    provenance: Provenance


class PlayerFrame(Contract):
    player_id: str
    position: Point
    speed_kmh: float | None = Field(None, ge=0, le=45)
    visible: bool = Field(True, description="False when off-camera in video mode; position is extrapolated")


class BallFrame(Contract):
    position: Point3D
    speed_kmh: float | None = Field(None, ge=0, le=200)
    in_play: bool = True
    owner_player_id: str | None = None


class TrackingFrame(Contract):
    match_id: str
    frame_id: int = Field(..., ge=0, description="Monotonic, 25 frames per second")
    clock: MatchClock
    ball: BallFrame
    players: list[PlayerFrame]
    provenance: Provenance


# --- Events (discriminated on `type`) --------------------------------------


class EventBase(Contract):
    event_id: str
    match_id: str
    seq: int = Field(..., ge=0, description="Monotonic per match; consumers order by this")
    clock: MatchClock
    team_id: str | None = None
    player_id: str | None = None
    location: Point | None = None
    related_event_ids: list[str] = Field(default_factory=list)
    provenance: Provenance


class PassEvent(EventBase):
    type: Literal["pass"] = "pass"
    end_location: Point
    recipient_id: str | None = None
    outcome: Literal["complete", "incomplete", "out", "offside"]
    height: Literal["ground", "low", "high"] = "ground"
    ball_speed_kmh: float | None = Field(None, ge=0, le=150)
    under_pressure: bool = False
    # Pass difficulty is NOT here: the stats engine derives it, so raw events stay raw.


class ShotEvent(EventBase):
    type: Literal["shot"] = "shot"
    end_location: Point3D = Field(..., description="Where the ball ended: goal plane, keeper or out")
    outcome: Literal["goal", "saved", "off_target", "blocked", "woodwork"]
    body_part: Literal["left_foot", "right_foot", "head", "other"]
    shot_speed_kmh: float | None = Field(None, ge=0, le=200)
    assist_event_id: str | None = None


class TackleEvent(EventBase):
    type: Literal["tackle"] = "tackle"
    opponent_id: str
    outcome: Literal["won", "lost", "foul"]


class InterceptionEvent(EventBase):
    type: Literal["interception"] = "interception"
    intercepted_event_id: str | None = None


class PressureEvent(EventBase):
    type: Literal["pressure"] = "pressure"
    target_player_id: str
    duration_ms: int = Field(..., ge=0)
    forced_turnover: bool = False


class PossessionChangeEvent(EventBase):
    type: Literal["possession_change"] = "possession_change"
    from_team_id: str
    to_team_id: str
    cause: Literal["tackle", "interception", "loose_ball", "out_of_play", "foul", "shot", "other"]


class FoulEvent(EventBase):
    type: Literal["foul"] = "foul"
    victim_id: str | None = None
    card: Literal["yellow", "second_yellow", "red"] | None = None


class SubstitutionEvent(EventBase):
    type: Literal["substitution"] = "substitution"
    player_off_id: str
    player_on_id: str


class PeriodEvent(EventBase):
    type: Literal["period_start", "period_end", "match_end"]


MatchEvent = Annotated[
    PassEvent
    | ShotEvent
    | TackleEvent
    | InterceptionEvent
    | PressureEvent
    | PossessionChangeEvent
    | FoulEvent
    | SubstitutionEvent
    | PeriodEvent,
    Field(discriminator="type"),
]


class StreamMessage(Contract):
    """Envelope for everything published to Event Hubs."""

    schema_version: str = SCHEMA_VERSION
    kind: Literal["metadata", "frame", "event"]
    match_id: str = Field(..., description="Also the Event Hubs partition key")
    payload: MatchMetadata | TrackingFrame | MatchEvent


# ---------------------------------------------------------------------------
# 2. STORY ATOM SCHEMA  (stats engine -> agent team)
# ---------------------------------------------------------------------------


class AtomKind(str, Enum):
    # Moments
    KEY_PASS = "key_pass"
    DIFFICULT_PASS = "difficult_pass"
    SHOT_QUALITY = "shot_quality"
    GOAL_CONTEXT = "goal_context"
    SPEED_THRESHOLD = "speed_threshold"
    DISTANCE_MILESTONE = "distance_milestone"
    PLAYER_MILESTONE = "player_milestone"
    # Phases and patterns (the "why it matters" layer)
    MOMENTUM_SHIFT = "momentum_shift"
    PRESSURE_SPELL = "pressure_spell"
    CONTROL_PHASE = "control_phase"
    CHAOS_PHASE = "chaos_phase"
    RHYTHM_CHANGE = "rhythm_change"
    FLANK_OVERLOAD = "flank_overload"


class MetricValue(Contract):
    value: float
    unit: str = Field(..., description="e.g. 'km/h', 'm', '%', 'xG', 'rating_0_10'")
    baseline: float | None = Field(None, description="Comparison value, e.g. match or season average")
    baseline_label: str | None = Field(None, description="What the baseline is, e.g. 'match average'")
    percentile: float | None = Field(None, ge=0, le=100)


class ClockWindow(Contract):
    start: MatchClock
    end: MatchClock


class StoryAtom(Contract):
    """A verified, structured fact. Agents may only narrate what atoms contain."""

    schema_version: str = SCHEMA_VERSION
    atom_id: str
    match_id: str
    kind: AtomKind
    window: ClockWindow
    team_ids: list[str] = Field(default_factory=list)
    player_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, MetricValue] = Field(
        default_factory=dict, description="Named numbers, e.g. {'pass_difficulty': ..., 'pass_length': ...}"
    )
    facts: list[str] = Field(
        default_factory=list,
        description="Short deterministic statements the narrator may rephrase, never extend",
    )
    importance: float = Field(..., ge=0.0, le=1.0, description="Used by the Producer to rank")
    evidence_event_ids: list[str] = Field(..., min_length=1)
    supersedes_atom_id: str | None = Field(None, description="Set when an ongoing phase is updated")
    provenance: Provenance


# ---------------------------------------------------------------------------
# 3. INSIGHT SCHEMA  (stats engine + agents -> web app, Teams, overlay API)
# ---------------------------------------------------------------------------


class Persona(str, Enum):
    CASUAL = "casual"
    ANALYST = "analyst"
    PLAYER_FOCUS = "player_focus"
    CLUB_FOCUS = "club_focus"
    METRIC_FOCUS = "metric_focus"
    PRODUCER = "producer"  # studio gallery / Teams channel


class Audience(Contract):
    persona: Persona
    language: str = Field("en", description="BCP-47 tag, e.g. 'en', 'fr', 'ar'")
    club_id: str | None = None
    player_id: str | None = None
    metric: str | None = None

    @property
    def segment_key(self) -> str:
        """Insights are generated per segment, not per user, so cost stays flat as viewers grow."""
        parts = [self.persona.value, self.language, self.club_id or "-", self.player_id or "-", self.metric or "-"]
        return ":".join(parts)


class InsightKind(str, Enum):
    LIVE = "live"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"
    ANSWER = "answer"  # reply to a fan or pundit question


class OverlayType(str, Enum):
    PLAYER_TAG = "player_tag"
    SPEED_BADGE = "speed_badge"
    DISTANCE_BADGE = "distance_badge"
    PASS_CARD = "pass_card"
    SHOT_CARD = "shot_card"
    MOMENTUM_BAR = "momentum_bar"
    STAT_PANEL = "stat_panel"
    LOWER_THIRD = "lower_third"
    TICKER = "ticker"


class OverlayAnchor(Contract):
    player_id: str | None = Field(None, description="Follow this player on screen")
    pitch_point: Point | None = Field(None, description="Pin to a pitch location")
    screen_region: Literal["top_left", "top_right", "bottom_left", "bottom_right", "lower_third", "full"] | None = None


class OverlaySpec(Contract):
    """Renderer-agnostic: a broadcast graphics system or our web app can draw it."""

    type: OverlayType
    anchor: OverlayAnchor
    data: dict[str, str | float | int | bool] = Field(
        default_factory=dict, description="Values the template needs, e.g. {'speed_kmh': 34.2}"
    )
    show_at: MatchClock
    duration_ms: int = Field(4000, ge=500, le=60000)


class VerificationStatus(str, Enum):
    NOT_REQUIRED = "not_required"  # deterministic rule output
    VERIFIED = "verified"
    REJECTED = "rejected"  # kept for traces, never displayed
    FALLBACK = "fallback"  # template used after a timeout or repeated rejection


class Verification(Contract):
    status: VerificationStatus
    attempts: int = Field(1, ge=1)
    notes: str | None = None


class Insight(Contract):
    schema_version: str = SCHEMA_VERSION
    insight_id: str
    match_id: str
    kind: InsightKind
    origin: Literal["rule", "agent"]
    audience: Audience
    headline: str = Field(..., max_length=60)
    body: str | None = Field(None, max_length=600)
    spoken_text: str | None = Field(None, description="Version written for text-to-speech")
    overlay: OverlaySpec | None = None
    priority: int = Field(3, ge=1, le=5, description="1 = must show, 5 = filler")
    ttl_ms: int = Field(15000, ge=1000, description="Drop if not shown within this time")
    evidence_atom_ids: list[str] = Field(default_factory=list)
    evidence_event_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Lowest confidence among its evidence")
    verification: Verification
    supersedes_insight_id: str | None = None
    generated_at: datetime
    latency_ms: int | None = Field(None, ge=0, description="From evidence event to insight ready")


# ---------------------------------------------------------------------------
# Examples and JSON Schema export
# ---------------------------------------------------------------------------


def _examples() -> dict[str, BaseModel]:
    now = datetime(2026, 10, 21, 15, 0, tzinfo=UTC)
    sim = Provenance(source=Source.SIMULATOR, confidence=1.0, model_version="sim-0.1")
    derived = Provenance(source=Source.DERIVED, confidence=0.95, model_version="stats-0.1")
    clock = MatchClock(period=2, period_ms=1_032_000, wall_time=now)

    pass_event = StreamMessage(
        kind="event",
        match_id="m-001",
        payload=PassEvent(
            event_id="e-1842",
            match_id="m-001",
            seq=1842,
            clock=clock,
            team_id="t-hbr",
            player_id="p-hbr-08",
            location=Point(x=48.2, y=20.5),
            end_location=Point(x=79.0, y=55.1),
            recipient_id="p-hbr-11",
            outcome="complete",
            height="high",
            ball_speed_kmh=71.4,
            under_pressure=True,
            provenance=sim,
        ),
    )

    atom = StoryAtom(
        atom_id="a-0311",
        match_id="m-001",
        kind=AtomKind.DIFFICULT_PASS,
        window=ClockWindow(start=clock, end=clock),
        team_ids=["t-hbr"],
        player_ids=["p-hbr-08", "p-hbr-11"],
        metrics={
            "pass_difficulty": MetricValue(value=8.7, unit="rating_0_10", baseline=4.1, baseline_label="match average"),
            "pass_length": MetricValue(value=46.3, unit="m"),
            "ball_speed": MetricValue(value=71.4, unit="km/h"),
        },
        facts=[
            "Diagonal switch from the left half-space to the right wing",
            "Passer was under pressure from two opponents",
            "Hardest completed pass of the match so far",
        ],
        importance=0.72,
        evidence_event_ids=["e-1842"],
        provenance=derived,
    )

    insight = Insight(
        insight_id="i-0927",
        match_id="m-001",
        kind=InsightKind.LIVE,
        origin="agent",
        audience=Audience(persona=Persona.CASUAL, language="fr", club_id="t-hbr"),
        headline="La passe du match",
        body="Sous la pression de deux adversaires, il trouve son ailier 46 mètres plus loin.",
        spoken_text="Quelle passe ! Quarante-six mètres, sous pression, en plein dans la course.",
        overlay=OverlaySpec(
            type=OverlayType.PASS_CARD,
            anchor=OverlayAnchor(player_id="p-hbr-11"),
            data={"difficulty": 8.7, "length_m": 46.3, "speed_kmh": 71.4},
            show_at=clock,
            duration_ms=5000,
        ),
        priority=2,
        evidence_atom_ids=["a-0311"],
        evidence_event_ids=["e-1842"],
        confidence=0.95,
        verification=Verification(status=VerificationStatus.VERIFIED, attempts=1),
        generated_at=now,
        latency_ms=2300,
    )

    return {"stream_message": pass_event, "story_atom": atom, "insight": insight}


def export_json_schemas(out_dir: str | Path = "schemas") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in {
        "stream_message": StreamMessage,
        "match_metadata": MatchMetadata,
        "tracking_frame": TrackingFrame,
        "story_atom": StoryAtom,
        "insight": Insight,
    }.items():
        path = out / f"{name}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2))
        written.append(path)
    for name, example in _examples().items():
        path = out / f"{name}.example.json"
        path.write_text(example.model_dump_json(indent=2, exclude_none=True))
        written.append(path)
    return written


if __name__ == "__main__":
    examples = _examples()
    # Round-trip every example to prove serialisation and validation agree.
    for model in examples.values():
        type(model).model_validate_json(model.model_dump_json())
    for p in export_json_schemas():
        print("wrote", p)
