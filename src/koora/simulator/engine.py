"""
Match engine: runs the simulation tick by tick and emits contract-valid StreamMessages.

Each tick (1/25 s):
    1. the player on the ball decides (pass, carry or shoot) when their decision timer is up
    2. every player gets a target point (shape, pressing, chasing the ball)
    3. players move towards their targets with realistic speed and acceleration
    4. the ball moves: attached to the dribbler, or under physics
    5. shots resolve (block, save), the ball is checked against the lines (goal, out of play)
    6. a free ball can be controlled by the nearest player within reach
    7. duels: pressure episodes and tackles on the ball carrier

Internally positions are numpy arrays in pitch coordinates. Contract objects are only built
at the output: Pydantic models for events, plain dicts for the 25 Hz frames (much faster;
the tests validate frames against the TrackingFrame model).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from koora.contracts.schemas import (
    SCHEMA_VERSION,
    FoulEvent,
    InterceptionEvent,
    MatchClock,
    MatchMetadata,
    PassEvent,
    PeriodEvent,
    PossessionChangeEvent,
    PressureEvent,
    Provenance,
    ShotEvent,
    Source,
    StreamMessage,
    TackleEvent,
    Team,
)
from koora.simulator import config as C
from koora.simulator import physics, tactics
from koora.simulator.decisions import DecisionContext, Option, choose_action
from koora.simulator.fake_feed import AWAY, HOME

MODEL_VERSION = "koora-sim-0.1"
PROVENANCE = Provenance(source=Source.SIMULATOR, confidence=1.0, model_version=MODEL_VERSION)
PROVENANCE_DICT = PROVENANCE.model_dump(mode="json", exclude_none=True)
ROLE_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


# --------------------------------------------------------------------------- in-flight state


@dataclass
class PassInFlight:
    passer: int
    team: int
    receiver: int
    target: np.ndarray  # pitch coordinates
    start_xy: np.ndarray
    start_ms: int
    lofted: bool
    speed_kmh: float
    under_pressure: bool


@dataclass
class ShotInFlight:
    shooter: int
    team: int
    start_xy: np.ndarray
    start_ms: int
    speed_kmh: float
    xg: float
    plan: str  # what the shooter is going for: goal, saved, off_target, block
    keeper: int
    blocker: int | None
    goal_x: float
    body_part: str
    launched_at: float


@dataclass
class Stoppage:
    kind: str  # kickoff, throw_in, corner, goal_kick, free_kick
    team: int
    spot: np.ndarray
    taker: int
    resume_at: float
    deadline: float
    ball_hold_until: float = 0.0


@dataclass
class Pressure:
    presser: int
    target: int
    start_time: float
    start_ms: int
    location: np.ndarray


# --------------------------------------------------------------------------- simulator


class MatchSimulator:
    def __init__(
        self,
        minutes: float = 10.0,
        seed: int = 7,
        match_id: str | None = None,
        home: Team = HOME,
        away: Team = AWAY,
        frame_every: int = 1,
    ) -> None:
        if minutes <= 0:
            raise ValueError("minutes must be positive")
        if frame_every < 1:
            raise ValueError("frame_every must be at least 1")
        self.frame_every = frame_every
        self.rng = np.random.default_rng(seed)
        self.match_id = match_id or f"m-sim-{seed}"
        self.teams = (home, away)
        self.half_seconds = minutes * 60.0 / 2.0

        # Players: 11 per team, ordered GK, DEF, MID, FWD so they map onto the 4-3-3 slots.
        lineup = []
        for team in (home, away):
            starters = sorted(team.players, key=lambda p: ROLE_ORDER[p.position.value])[:11]
            if len(starters) != 11:
                raise ValueError(f"{team.name} needs 11 players")
            lineup.extend(starters)
        self.player_ids = [p.player_id for p in lineup]
        self.team = np.array([0] * 11 + [1] * 11)
        self.team_ids = (home.team_id, away.team_id)
        self.roles = [p.position.value for p in lineup]
        self.slots = tactics.SLOT_NAMES * 2
        self._indices = {
            (t, outfield): np.array(
                [i for i in range(22) if self.team[i] == t and not (outfield and self.roles[i] == "GK")]
            )
            for t in (0, 1)
            for outfield in (False, True)
        }

        # Attributes (hidden from the output, they drive the simulation)
        base_speed = {"GK": 7.6, "DEF": 8.7, "MID": 8.8, "FWD": 9.1}
        self.max_speed = np.array([np.clip(self.rng.normal(base_speed[r], 0.3), 7.2, 9.7) for r in self.roles])
        self.passing = self._attribute({"GK": 0.65, "DEF": 0.72, "MID": 0.8, "FWD": 0.74})
        self.shooting = self._attribute({"GK": 0.3, "DEF": 0.5, "MID": 0.68, "FWD": 0.8})
        self.tackling = self._attribute({"GK": 0.3, "DEF": 0.75, "MID": 0.65, "FWD": 0.45})
        self.dribbling = self._attribute({"GK": 0.4, "DEF": 0.58, "MID": 0.7, "FWD": 0.78})
        self.right_footed = self.rng.random(22) < 0.75

        # Dynamic state
        self.pos = np.zeros((22, 2))
        self.vel = np.zeros((22, 2))
        self.stamina = np.ones(22)
        self.distance = np.zeros(22)
        self.top_speed = np.zeros(22)
        self.tackle_ready_at = np.zeros(22)
        self.ball_pos = np.array([C.CENTRE_X, C.CENTRE_Y, 0.0])
        self.ball_vel = np.zeros(3)
        self.prev_ball = self.ball_pos.copy()
        self.ball_in_play = False
        self.owner: int | None = None
        self.possession = 0
        self.last_touch_team = 0
        self.last_kicker: int | None = None
        self.last_kick_time = -10.0
        self.kick_point = self.ball_pos[:2].copy()
        self.screened: set[int] = set()
        self.loose_cause = "loose_ball"
        self.carry_target: np.ndarray | None = None
        self.next_decision = 0.0
        self.active_pass: PassInFlight | None = None
        self.active_shot: ShotInFlight | None = None
        self.stoppage: Stoppage | None = None
        self.pressure: Pressure | None = None
        self.last_options: list[Option] = []  # what the last decision considered (for explainability)

        self.score = [0, 0]
        self.period = 1
        self.period_time = 0.0
        self.time = 0.0
        self.frame_id = 0
        self.tick_count = 0
        self.seq = 0
        self.pending: list[str] = []
        self.stats = {
            "passes": [0, 0],
            "passes_complete": [0, 0],
            "shots": [0, 0],
            "on_target": [0, 0],
            "xg": [0.0, 0.0],
            "tackles": [0, 0],
            "fouls": [0, 0],
            "possession_s": [0.0, 0.0],
        }

    def _attribute(self, means: dict[str, float]) -> np.ndarray:
        return np.array([np.clip(self.rng.normal(means[r], 0.07), 0.2, 0.97) for r in self.roles])

    # ----------------------------------------------------------------------- helpers

    def _dir(self, team: int) -> int:
        """+1 if the team attacks towards x = 105 in the current period, else -1."""
        base = 1 if team == 0 else -1
        return base if self.period == 1 else -base

    def _tf(self, xy: np.ndarray, team: int) -> np.ndarray:
        """Pitch <-> team frame. Rotating 180 degrees is its own inverse, so this works both ways."""
        if self._dir(team) == 1:
            return np.array((xy[0], xy[1]), dtype=float)
        return np.array((C.PITCH_LENGTH - xy[0], C.PITCH_WIDTH - xy[1]))

    def _own_goal(self, team: int) -> np.ndarray:
        return np.array([0.0 if self._dir(team) == 1 else C.PITCH_LENGTH, C.CENTRE_Y])

    def _team_indices(self, team: int, outfield: bool = False) -> np.ndarray:
        return self._indices[(team, outfield)]

    def _keeper(self, team: int) -> int:
        return next(i for i in range(22) if self.team[i] == team and self.roles[i] == "GK")

    def _slot_player(self, team: int, slot: str) -> int:
        return next(i for i in range(22) if self.team[i] == team and self.slots[i] == slot)

    def _in_own_box(self, player: int, xy: np.ndarray) -> bool:
        x, y = self._tf(xy, self.team[player])
        return x < C.PENALTY_AREA_DEPTH and abs(y - C.CENTRE_Y) < C.PENALTY_AREA_HALF_WIDTH

    def _nearest_opponent_gap(self, player: int) -> float:
        opp = self._team_indices(1 - self.team[player])
        return float(np.min(np.linalg.norm(self.pos[opp] - self.pos[player], axis=1)))

    def _now_ms(self) -> int:
        return int(round(self.period_time * 1000))

    @staticmethod
    def _pt(xy) -> dict[str, float]:
        return {
            "x": round(C.clamp(float(xy[0]), -4.9, C.PITCH_LENGTH + 4.9), 2),
            "y": round(C.clamp(float(xy[1]), -4.9, C.PITCH_WIDTH + 4.9), 2),
        }

    def _pt3(self, xyz) -> dict[str, float]:
        return {**self._pt(xyz), "z": round(C.clamp(float(xyz[2]), 0.0, 49.9), 2)}

    # ----------------------------------------------------------------------- output

    def _emit(self, model, ms: int | None = None, **fields) -> str:
        self.seq += 1
        event_id = f"e-{self.seq:05d}"
        event = model(
            event_id=event_id,
            match_id=self.match_id,
            seq=self.seq,
            clock=MatchClock(period=self.period, period_ms=self._now_ms() if ms is None else ms),
            provenance=PROVENANCE,
            **fields,
        )
        message = StreamMessage(kind="event", match_id=self.match_id, payload=event)
        self.pending.append(message.model_dump_json(exclude_none=True))
        return event_id

    def _flush(self) -> Iterator[str]:
        yield from self.pending
        self.pending.clear()

    def _metadata_line(self) -> str:
        home, away = self.teams
        message = StreamMessage(
            kind="metadata",
            match_id=self.match_id,
            payload=MatchMetadata(
                match_id=self.match_id,
                competition="KOORA Simulated League",
                kickoff=datetime(2026, 10, 24, 15, 0, tzinfo=UTC),
                venue="Harbour Park",
                home=home,
                away=away,
                attacking_direction={home.team_id: {1: "+x", 2: "-x"}, away.team_id: {1: "-x", 2: "+x"}},
                provenance=PROVENANCE,
            ),
        )
        return message.model_dump_json(exclude_none=True)

    def _frame_line(self) -> str:
        kmh = np.minimum(np.hypot(self.vel[:, 0], self.vel[:, 1]) * 3.6, 44.9).round(1).tolist()
        xs = np.clip(self.pos[:, 0], -4.9, C.PITCH_LENGTH + 4.9).round(2).tolist()
        ys = np.clip(self.pos[:, 1], -4.9, C.PITCH_WIDTH + 4.9).round(2).tolist()
        players = [
            {"player_id": pid, "position": {"x": x, "y": y}, "speed_kmh": v, "visible": True}
            for pid, x, y, v in zip(self.player_ids, xs, ys, kmh, strict=True)
        ]
        ball = {
            "position": self._pt3(self.ball_pos),
            "speed_kmh": round(min(float(np.linalg.norm(self.ball_vel)) * 3.6, 199.9), 1),
            "in_play": self.ball_in_play,
        }
        if self.owner is not None:
            ball["owner_player_id"] = self.player_ids[self.owner]
        payload = {
            "match_id": self.match_id,
            "frame_id": self.frame_id,
            "clock": {"period": self.period, "period_ms": self._now_ms()},
            "ball": ball,
            "players": players,
            "provenance": PROVENANCE_DICT,
        }
        self.frame_id += 1
        message = {"schema_version": SCHEMA_VERSION, "kind": "frame", "match_id": self.match_id, "payload": payload}
        return json.dumps(message, separators=(",", ":"))

    # ----------------------------------------------------------------------- main loop

    def run(self) -> Iterator[str]:
        """Simulate the whole match, yielding one JSON line per StreamMessage."""
        yield self._metadata_line()
        ticks_per_half = int(round(self.half_seconds / C.DT))
        for period in (1, 2):
            self._start_period(period)
            yield from self._flush()
            for _ in range(ticks_per_half):
                self._tick()
                self.time += C.DT
                self.period_time += C.DT
                if self.tick_count % self.frame_every == 0:
                    yield self._frame_line()
                else:
                    self.frame_id += 1  # keep frame ids tied to simulation time
                self.tick_count += 1
                yield from self._flush()
            self._emit(PeriodEvent, type="period_end")
            if period == 2:
                self._emit(PeriodEvent, type="match_end")
            yield from self._flush()

    def _start_period(self, period: int) -> None:
        self.period = period
        self.period_time = 0.0
        self.active_pass = self.active_shot = self.pressure = None
        self.owner = None
        kicking = 0 if period == 1 else 1
        for i in range(22):  # teleport to kick-off positions
            t = self.team[i]
            self.pos[i] = self._tf(tactics.kickoff_target(self.slots[i], kicking=(t == kicking)), t)
        self.vel[:] = 0
        self.possession = kicking
        self._emit(PeriodEvent, type="period_start")
        self._stop("kickoff", kicking, np.array([C.CENTRE_X, C.CENTRE_Y]), C.KICKOFF_DELAY)

    def _tick(self) -> None:
        if self.stoppage is not None:
            self._tick_stoppage()
            return

        if self.owner is not None and self.time >= self.next_decision:
            self._decide()

        targets, caps, accel = self._targets()
        self._move(targets, caps, accel)

        if self.owner is not None:
            self._attach_ball()
        else:
            self.prev_ball[:] = self.ball_pos
            physics.step_ball(self.ball_pos, self.ball_vel)
            if self.active_shot is not None:
                self._resolve_shot_contacts()
            if self.owner is None and self.stoppage is None:
                self._check_lines()
            if self.owner is None and self.stoppage is None and self.active_shot is None:
                self._check_control()

        if self.owner is not None and self.stoppage is None:
            self._duels()
        elif self.pressure is not None:
            self._end_pressure(forced=False)

        if self.ball_in_play:
            self.stats["possession_s"][self.possession] += C.DT

    # ----------------------------------------------------------------------- movement

    def _shape_targets(self, kickoff_team: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        targets = np.zeros((22, 2))
        caps = np.zeros(22)
        ball_xy = self.ball_pos[:2]
        for t in (0, 1):
            ball_tf = self._tf(ball_xy, t)
            opp_x = self.pos[self.team == 1 - t, 0]
            opp_x = np.sort(opp_x if self._dir(t) == 1 else C.PITCH_LENGTH - opp_x)
            last_defender_x = float(opp_x[-2])  # deepest is their keeper; second deepest is the offside line
            in_possession = self.possession == t
            for i in self._team_indices(t):
                slot = self.slots[i]
                if kickoff_team is not None:
                    target = tactics.kickoff_target(slot, kicking=(t == kickoff_team))
                elif slot == "GK":
                    target = tactics.keeper_target(ball_tf)
                else:
                    target = tactics.shape_target(slot, ball_tf, in_possession)
                    if in_possession and self.roles[i] in ("MID", "FWD"):
                        target = tactics.onside(target, ball_tf, last_defender_x)
                targets[i] = self._tf(target, t)
        far = np.hypot(*(targets - self.pos).T) > 10
        caps[:] = np.where(far, C.RUN_SPEED, C.JOG_SPEED)
        return targets, caps

    def _targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        targets, caps = self._shape_targets()
        accel = np.full(22, C.ACCELERATION)
        ball_xy = self.ball_pos[:2]

        if self.owner is not None:
            c = self.owner
            if self.carry_target is not None:
                targets[c] = self.carry_target
                caps[c] = self.max_speed[c] * C.DRIBBLE_SPEED_FACTOR
            else:
                targets[c] = self.pos[c]  # shielding the ball, waiting for an option
                caps[c] = 1.0
            # Out of possession: nearest player presses, second nearest covers when the ball is near goal
            opp = 1 - self.team[c]
            outfield = self._team_indices(opp, outfield=True)
            order = np.argsort(np.linalg.norm(self.pos[outfield] - self.pos[c], axis=1))
            goal_side = self._own_goal(opp) - self.pos[c]
            goal_side /= np.linalg.norm(goal_side) + 1e-6
            presser = outfield[order[0]]
            targets[presser] = self.pos[c] + goal_side * C.PRESS_DISTANCE
            caps[presser] = self.max_speed[presser]
            if self._tf(ball_xy, opp)[0] < 55:
                cover = outfield[order[1]]
                targets[cover] = self.pos[c] + goal_side * 5.0
                caps[cover] = C.RUN_SPEED
        else:
            # Loose ball or pass in flight: the quickest player of each team goes for it
            predicted = self._predicted_ball()
            predicted = np.array([C.clamp(predicted[0], 0, C.PITCH_LENGTH), C.clamp(predicted[1], 0, C.PITCH_WIDTH)])
            for t in (0, 1):
                if self.active_pass is not None and self.active_pass.team == t:
                    r = self.active_pass.receiver
                    far = np.linalg.norm(ball_xy - self.pos[r]) > 6
                    targets[r] = self.active_pass.target if far else predicted
                    caps[r] = self.max_speed[r]
                    continue
                candidates = [
                    i for i in self._team_indices(t) if self.roles[i] != "GK" or self._in_own_box(i, predicted)
                ]
                chaser, point = self._interceptor(np.array(candidates), predicted)
                targets[chaser] = point
                caps[chaser] = self.max_speed[chaser]

        if self.active_shot is not None:
            self._keeper_dive(targets, caps, accel)
        return targets, caps, accel

    def _predicted_ball(self) -> np.ndarray:
        """Where to run for the ball: its landing point if it is in the air, else just ahead of it."""
        x, y, z = self.ball_pos
        vx, vy, vz = self.ball_vel
        if z > 0.3 or vz > 0.6:
            t_land = (vz + math.sqrt(vz * vz + 2 * C.GRAVITY * z)) / C.GRAVITY
            t = t_land * 0.85  # drag shortens the flight
        else:
            t = 0.4
        return np.array([x + vx * t, y + vy * t])

    def _interceptor(self, candidates: np.ndarray, predicted: np.ndarray) -> tuple[int, np.ndarray]:
        """Which player can get to the ball first, and where.

        For a rolling ball: check points along its path and pick the player with the best
        margin over the ball's arrival there. For a ball in the air: race to the landing point.
        """
        speed = float(np.hypot(self.ball_vel[0], self.ball_vel[1]))
        airborne = self.ball_pos[2] > 0.3 or self.ball_vel[2] > 0.6
        if airborne or speed < 1.0:
            times = np.linalg.norm(self.pos[candidates] - predicted, axis=1) / self.max_speed[candidates]
            return int(candidates[int(np.argmin(times))]), predicted

        heading = self.ball_vel[:2] / speed
        stop = min(physics.ground_stop_distance(speed), 40.0)
        s = np.linspace(min(0.5, stop), stop, 16)
        t_ball = physics.ground_time_to(speed, s)
        reachable = np.isfinite(t_ball)
        s, t_ball = s[reachable], t_ball[reachable]
        if len(s) == 0:
            return int(candidates[0]), predicted
        points = self.ball_pos[:2] + np.outer(s, heading)
        points[:, 0] = np.clip(points[:, 0], 0, C.PITCH_LENGTH)
        points[:, 1] = np.clip(points[:, 1], 0, C.PITCH_WIDTH)
        gaps = np.linalg.norm(points[:, None, :] - self.pos[candidates][None, :, :], axis=2)
        reach = np.maximum(gaps - C.CONTROL_RADIUS, 0.0)
        t_player = reach / self.max_speed[candidates][None, :]
        margin = t_player - t_ball[:, None]  # negative: the player gets there first
        # Earliest point each player can make, then the player who makes it soonest
        ok = margin <= 0
        best_point = np.where(ok.any(axis=0), np.argmax(ok, axis=0), np.argmin(margin, axis=0))
        arrival = t_ball[best_point] + np.maximum(margin[best_point, np.arange(len(candidates))], 0)
        k = int(np.argmin(arrival))
        return int(candidates[k]), points[best_point[k]]

    def _keeper_dive(self, targets: np.ndarray, caps: np.ndarray, accel: np.ndarray) -> None:
        shot = self.active_shot
        k = shot.keeper
        y = self.ball_pos[1]
        if abs(self.ball_vel[0]) > 0.5:
            t_cross = (self.pos[k][0] - self.ball_pos[0]) / self.ball_vel[0]
            if t_cross > 0:
                y = self.ball_pos[1] + self.ball_vel[1] * t_cross
        y = C.clamp(float(y), C.CENTRE_Y - 4, C.CENTRE_Y + 4)
        if shot.plan == "goal":  # beaten: the keeper reacts late
            y = self.pos[k][1] + 0.35 * (y - self.pos[k][1])
        targets[k] = np.array([self.pos[k][0], y])
        caps[k] = 7.5
        accel[k] = C.KEEPER_DIVE_ACCELERATION

    def _move(self, targets: np.ndarray, caps: np.ndarray, accel: np.ndarray | float = C.ACCELERATION) -> None:
        """Steering: head for the target, slow down on arrival, limited by acceleration and top speed."""
        to_target = targets - self.pos
        dist = np.linalg.norm(to_target, axis=1)
        effective_max = self.max_speed * (C.STAMINA_SPEED_FLOOR + (1 - C.STAMINA_SPEED_FLOOR) * self.stamina)
        desired_speed = np.minimum(np.minimum(caps, dist * C.ARRIVAL_GAIN), effective_max)
        desired_speed[dist < C.DEADBAND] = 0.0
        direction = np.divide(to_target, dist[:, None], out=np.zeros_like(to_target), where=dist[:, None] > 1e-6)
        desired = direction * desired_speed[:, None]

        dv = desired - self.vel
        dv_norm = np.linalg.norm(dv, axis=1)
        limit = np.asarray(accel) * C.DT
        scale = np.minimum(1.0, np.divide(limit, dv_norm, out=np.ones_like(dv_norm), where=dv_norm > 1e-9))
        self.vel += dv * scale[:, None]

        speed = np.linalg.norm(self.vel, axis=1)
        too_fast = speed > effective_max
        self.vel[too_fast] *= (effective_max[too_fast] / speed[too_fast])[:, None]
        self.pos += self.vel * C.DT

        # Separation: push apart players who overlap
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        gap = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(gap, np.inf)
        overlap = np.clip(C.SEPARATION - gap, 0, None)
        push = np.divide(diff, gap[..., None], out=np.zeros_like(diff), where=np.isfinite(gap)[..., None])
        self.pos += 0.5 * (push * overlap[..., None]).sum(axis=1)
        self.pos[:, 0] = np.clip(self.pos[:, 0], -1.5, C.PITCH_LENGTH + 1.5)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -1.5, C.PITCH_WIDTH + 1.5)

        # Stamina and running stats
        speed = np.linalg.norm(self.vel, axis=1)
        self.stamina -= C.STAMINA_DRAIN * np.clip(speed - C.SPRINT_DRAIN_FROM, 0, None) * C.DT
        self.stamina += np.where(speed < 3.0, C.STAMINA_RECOVERY * C.DT, 0.0)
        self.stamina = np.clip(self.stamina, 0.0, 1.0)
        self.distance += speed * C.DT
        self.top_speed = np.maximum(self.top_speed, speed)

    def _attach_ball(self) -> None:
        c = self.owner
        v = self.vel[c]
        speed = float(np.linalg.norm(v))
        opponents = self._team_indices(1 - self.team[c])
        gaps = np.linalg.norm(self.pos[opponents] - self.pos[c], axis=1)
        if float(np.min(gaps)) < C.PRESSURE_START:  # shield: keep the ball on the far side from the presser
            away = self.pos[c] - self.pos[opponents[int(np.argmin(gaps))]]
            direction = away / (np.linalg.norm(away) + 1e-6)
        elif speed > 0.5:
            direction = v / speed
        else:
            direction = np.array([float(self._dir(self.team[c])), 0.0])
        xy = self.pos[c] + direction * C.DRIBBLE_OFFSET
        xy = np.array([C.clamp(xy[0], 0.3, C.PITCH_LENGTH - 0.3), C.clamp(xy[1], 0.3, C.PITCH_WIDTH - 0.3)])
        self.ball_pos[:] = (xy[0], xy[1], 0.0)
        self.ball_vel[:] = (v[0], v[1], 0.0)

    # ----------------------------------------------------------------------- decisions and kicks

    def _decide(self) -> None:
        c = self.owner
        t = self.team[c]
        d = self._dir(t)
        pressured = self._nearest_opponent_gap(c) < C.UNDER_PRESSURE_DISTANCE
        ctx = DecisionContext(
            pos=np.array([self._tf(p, t) for p in self.pos]),
            vel=self.vel * d,
            team=self.team,
            roles=self.roles,
            max_speed=self.max_speed,
            carrier=c,
            ball=self._tf(self.ball_pos[:2], t),
            passing=float(self.passing[c]),
            pressured=pressured,
            rng=self.rng,
        )
        choice, self.last_options = choose_action(ctx)
        if choice.kind == "pass":
            self._execute_pass(choice.receiver, self._tf(choice.target, t), choice.lofted)
        elif choice.kind == "shoot":
            self._execute_shot()
        else:
            self.carry_target = self._tf(choice.target, t)
            self.next_decision = self.time + self.rng.uniform(*C.CARRY_TIME)

    def _kick(self, velocity: np.ndarray) -> None:
        kicker = self.owner
        # Opponents right next to the kicker get one chance to block; the rest let it past
        self.screened = set()
        for j in self._team_indices(1 - self.team[kicker]):
            if np.linalg.norm(self.pos[j] - self.ball_pos[:2]) < C.BLOCK_RANGE and self.rng.random() > C.BLOCK_CHANCE:
                self.screened.add(int(j))
        self.kick_point = self.ball_pos[:2].copy()
        self.last_kicker = kicker
        self.last_kick_time = self.time
        self.last_touch_team = self.team[kicker]
        self.ball_vel[:] = velocity
        self.owner = None
        self.carry_target = None

    def _execute_pass(self, receiver: int, target: np.ndarray, lofted: bool) -> None:
        passer = self.owner
        start = self.ball_pos[:2].copy()
        target = np.array([C.clamp(target[0], 1, C.PITCH_LENGTH - 1), C.clamp(target[1], 1, C.PITCH_WIDTH - 1)])
        offset = target - start
        distance = max(float(np.linalg.norm(offset)), 1.0)
        pressured = self._nearest_opponent_gap(passer) < C.UNDER_PRESSURE_DISTANCE

        # Execution error grows with distance, pressure and lower passing skill
        sigma = math.radians(0.8 + 2.5 * (1 - self.passing[passer]) + (1.0 if pressured else 0.0)) * (1 + distance / 80)
        angle = math.atan2(offset[1], offset[0]) + self.rng.normal(0, sigma)
        heading = np.array([math.cos(angle), math.sin(angle)])
        speed_error = self.rng.normal(1.0, 0.03)

        if lofted:
            elevation = math.radians(self.rng.uniform(28, 40))
            v0, _ = physics.lofted_kick_speed(distance, elevation)
            v0 = min(v0 * speed_error, 40.0)
            velocity = np.array([*(heading * v0 * math.cos(elevation)), v0 * math.sin(elevation)])
        else:
            v0 = min(physics.ground_kick_speed(distance, self.rng.uniform(4.5, 7.5)) * speed_error, 28.0)
            velocity = np.array([*(heading * v0), 0.0])

        self.active_pass = PassInFlight(
            passer=passer,
            team=self.team[passer],
            receiver=receiver,
            target=target,
            start_xy=start,
            start_ms=self._now_ms(),
            lofted=lofted,
            speed_kmh=round(v0 * 3.6, 1),
            under_pressure=pressured,
        )
        self.stats["passes"][self.team[passer]] += 1
        self._kick(velocity)

    def _execute_shot(self) -> None:
        shooter = self.owner
        t = self.team[shooter]
        start = self.ball_pos[:2].copy()
        start_tf = self._tf(start, t)
        pressured = self._nearest_opponent_gap(shooter) < C.UNDER_PRESSURE_DISTANCE
        xg = physics.expected_goals(*start_tf) * (0.75 if pressured else 1.0)
        keeper = self._keeper(1 - t)
        goal_x = C.PITCH_LENGTH if self._dir(t) == 1 else 0.0
        goal = np.array([goal_x, C.CENTRE_Y])

        # Defenders standing in the shooting lane can block
        blockers = []
        lane = goal - start
        lane_length = float(np.linalg.norm(lane))
        for j in self._team_indices(1 - t, outfield=True):
            rel = self.pos[j] - start
            along = float(np.dot(rel, lane) / lane_length)
            if 1.0 < along < 0.7 * lane_length:
                unit = lane / lane_length
                across = abs(unit[0] * rel[1] - unit[1] * rel[0])  # 2D cross product
                if across < 1.8:
                    blockers.append((along, j))
        blocker = min(blockers)[1] if blockers else None

        # The shot's intended outcome decides where it is aimed; physics then plays it out
        p_block = min(0.45, 0.12 + 0.15 * len(blockers)) if blockers else 0.0
        p_goal = xg
        rest = max(0.0, 1.0 - p_block - p_goal)
        plans = ["block", "goal", "saved", "off_target"]
        probs = np.array([p_block, p_goal, rest * 0.4, rest * 0.6])
        plan = plans[int(self.rng.choice(4, p=probs / probs.sum()))]
        if plan == "block" and blocker is None:
            plan = "off_target"

        keeper_y = self.pos[keeper][1]
        if plan in ("goal", "block"):
            side = np.sign(C.CENTRE_Y - keeper_y) or self.rng.choice([-1.0, 1.0])
            aim = (C.CENTRE_Y + side * self.rng.uniform(1.6, 3.3), self.rng.uniform(0.15, 2.1))
        elif plan == "saved":
            aim = (np.clip(keeper_y + self.rng.uniform(-1.6, 1.6), 30.6, 37.4), self.rng.uniform(0.2, 1.9))
        elif self.rng.random() < 0.75:  # wide
            aim = (C.CENTRE_Y + self.rng.choice([-1, 1]) * self.rng.uniform(3.9, 6.5), self.rng.uniform(0.2, 2.0))
        else:  # over the bar
            aim = (C.CENTRE_Y + self.rng.uniform(-3, 3), self.rng.uniform(3.2, 5.0))

        target = np.array([goal_x, aim[0]])
        flat = target - start
        distance = float(np.linalg.norm(flat))
        v_h = self.rng.uniform(22, 31) * (0.85 + 0.15 * self.shooting[shooter])
        v_z = physics.shot_lift(v_h, distance, aim[1])  # solved with drag, so it crosses at the aimed height
        heading = flat / distance
        velocity = np.array([*(heading * v_h), v_z])
        foot = "right_foot" if self.right_footed[shooter] else "left_foot"

        self.active_shot = ShotInFlight(
            shooter=shooter,
            team=t,
            start_xy=start,
            start_ms=self._now_ms(),
            speed_kmh=round(float(np.linalg.norm(velocity)) * 3.6, 1),
            xg=round(xg, 3),
            plan=plan,
            keeper=keeper,
            blocker=blocker,
            goal_x=goal_x,
            body_part=foot,
            launched_at=self.time,
        )
        self.stats["shots"][t] += 1
        self.stats["xg"][t] += xg
        self._kick(velocity)

    # ----------------------------------------------------------------------- shots

    def _end_shot(self, outcome: str, end_xyz: np.ndarray) -> None:
        shot = self.active_shot
        self.active_shot = None
        if outcome in ("goal", "saved"):
            self.stats["on_target"][shot.team] += 1
        self._emit(
            ShotEvent,
            ms=shot.start_ms,
            team_id=self.team_ids[shot.team],
            player_id=self.player_ids[shot.shooter],
            location=self._pt(shot.start_xy),
            end_location=self._pt3(end_xyz),
            outcome=outcome,
            body_part=shot.body_part,
            shot_speed_kmh=shot.speed_kmh,
        )

    def _resolve_shot_contacts(self) -> None:
        shot = self.active_shot
        ball_xy = self.ball_pos[:2]
        z = self.ball_pos[2]
        if shot.plan == "block" and shot.blocker is not None:
            lane = np.array([shot.goal_x, C.CENTRE_Y]) - shot.start_xy
            lane /= np.linalg.norm(lane)
            passed = np.dot(ball_xy - shot.start_xy, lane) >= np.dot(self.pos[shot.blocker] - shot.start_xy, lane)
            # The blocker throws themselves in the way: contact when the ball reaches them or goes past
            if (np.linalg.norm(ball_xy - self.pos[shot.blocker]) < 1.3 or passed) and z < 2.2:
                self._end_shot("blocked", self.ball_pos)
                v = self.ball_vel
                lateral = self.rng.normal(0, 4.0, size=2)
                self.ball_vel[:] = (-0.25 * v[0] + lateral[0], -0.25 * v[1] + lateral[1], self.rng.uniform(0.5, 3.0))
                self._touched_by(shot.blocker, cause="shot")
                return
        if shot.plan == "saved" and np.linalg.norm(ball_xy - self.pos[shot.keeper]) < 1.7 and z < 2.6:
            self._save()
            return
        if self.time - shot.launched_at > 3.0:  # a scuffed shot that never reached the goal
            self._end_shot("off_target", self.ball_pos)

    def _save(self, at_line: bool = False) -> None:
        shot = self.active_shot
        k = shot.keeper
        if at_line:  # the keeper got there at full stretch
            inside = 0.4 if shot.goal_x == 0.0 else -0.4
            self.ball_pos[0] = shot.goal_x + inside
            self.pos[k] = (shot.goal_x + inside, self.ball_pos[1])
        self._end_shot("saved", self.ball_pos)
        if self.rng.random() < 0.6:  # caught
            self.ball_pos[2] = 0.0
            self._gain_control(k, cause="shot")
        else:  # parried
            v = self.ball_vel
            self.ball_vel[:] = (-0.3 * v[0], 0.3 * v[1] + self.rng.normal(0, 5.0), self.rng.uniform(1.0, 4.0))
            self._touched_by(k, cause="shot")

    def _touched_by(self, player: int, cause: str) -> None:
        self.last_kicker = player
        self.last_kick_time = self.time
        self.last_touch_team = self.team[player]
        self.loose_cause = cause

    # ----------------------------------------------------------------------- lines: goals and restarts

    def _check_lines(self) -> None:
        x, y, z = self.ball_pos
        px, py, pz = self.prev_ball

        if x < 0 or x > C.PITCH_LENGTH:
            line_x = 0.0 if x < 0 else C.PITCH_LENGTH
            f = (line_x - px) / (x - px) if x != px else 1.0
            cross = np.array([line_x, py + f * (y - py), max(0.0, pz + f * (z - pz))])
            defending = 0 if self._own_goal(0)[0] == line_x else 1
            attacking = 1 - defending
            in_goal_mouth = abs(cross[1] - C.CENTRE_Y) < C.GOAL_HALF_WIDTH and cross[2] < C.GOAL_HEIGHT

            if in_goal_mouth and self.active_shot is not None:
                if self.active_shot.plan == "saved":
                    self.ball_pos[1] = cross[1]
                    self._save(at_line=True)
                    return
                self._goal(attacking, defending, cross)
                return
            if self.active_pass is not None:
                self._end_pass("out", cross)
            if self.active_shot is not None:
                self._end_shot("off_target", cross)
            corner_side = C.PITCH_WIDTH - 0.5 if cross[1] > C.CENTRE_Y else 0.5
            if self.last_touch_team == defending:  # corner
                spot = np.array([line_x + (0.5 if line_x == 0 else -0.5), corner_side])
                taker = self._nearest(attacking, spot, roles=("MID", "FWD"))
                self._stop("corner", attacking, spot, C.CORNER_DELAY, taker)
            else:  # goal kick
                inside = 5.5 if line_x == 0 else C.PITCH_LENGTH - 5.5
                spot = np.array([inside, C.CENTRE_Y + self.rng.choice([-9.0, 9.0])])
                self._stop("goal_kick", defending, spot, C.GOAL_KICK_DELAY, self._keeper(defending))
            return

        if y < 0 or y > C.PITCH_WIDTH:
            line_y = 0.0 if y < 0 else C.PITCH_WIDTH
            f = (line_y - py) / (y - py) if y != py else 1.0
            cross = np.array([px + f * (x - px), line_y, max(0.0, pz + f * (z - pz))])
            if self.active_pass is not None:
                self._end_pass("out", cross)
            if self.active_shot is not None:
                self._end_shot("off_target", cross)
            team = 1 - self.last_touch_team
            spot = np.array([np.clip(cross[0], 1.0, C.PITCH_LENGTH - 1.0), 0.3 if line_y == 0 else C.PITCH_WIDTH - 0.3])
            self._stop("throw_in", team, spot, C.THROW_IN_DELAY, self._nearest(team, spot))

    def _goal(self, attacking: int, defending: int, cross: np.ndarray) -> None:
        self.score[attacking] += 1
        self._end_shot("goal", cross)
        self.ball_pos[:] = (cross[0] + (-1.0 if cross[0] == 0 else 1.0), cross[1], 0.0)
        self._stop("kickoff", defending, np.array([C.CENTRE_X, C.CENTRE_Y]), C.GOAL_CELEBRATION, hold_ball=2.0)

    def _nearest(self, team: int, spot: np.ndarray, roles: tuple[str, ...] = ("DEF", "MID", "FWD")) -> int:
        idx = [i for i in self._team_indices(team) if self.roles[i] in roles]
        return idx[int(np.argmin([np.linalg.norm(self.pos[i] - spot) for i in idx]))]

    def _stop(self, kind: str, team: int, spot: np.ndarray, delay: float, taker: int | None = None, hold_ball=0.0):
        """Stop play for a restart. The taker walks to the spot; everyone else takes up position."""
        if self.active_pass is not None:
            self._end_pass("incomplete", self.ball_pos[:2])
        self.active_shot = None
        if self.pressure is not None:
            self._end_pressure(forced=False)
        self.owner = None
        self.carry_target = None
        self.ball_in_play = False
        self.ball_vel[:] = 0
        if team != self.possession:
            cause = {"kickoff": "other", "free_kick": "foul"}.get(kind, "out_of_play")
            self._emit(
                PossessionChangeEvent,
                team_id=self.team_ids[team],
                location=self._pt(spot),
                from_team_id=self.team_ids[self.possession],
                to_team_id=self.team_ids[team],
                cause=cause,
            )
            self.possession = team
        if taker is None:
            taker = self._slot_player(team, "ST")
        self.stoppage = Stoppage(
            kind=kind,
            team=team,
            spot=spot.astype(float),
            taker=taker,
            resume_at=self.time + delay,
            deadline=self.time + delay + 4.0,
            ball_hold_until=self.time + hold_ball,
        )

    def _tick_stoppage(self) -> None:
        s = self.stoppage
        if self.time >= s.ball_hold_until:
            self.ball_pos[:] = (s.spot[0], s.spot[1], 0.0)
            self.ball_vel[:] = 0
        else:
            self.prev_ball[:] = self.ball_pos
            physics.step_ball(self.ball_pos, self.ball_vel)

        targets, caps = self._shape_targets(kickoff_team=s.team if s.kind == "kickoff" else None)
        if s.kind == "corner":
            for t in (0, 1):
                attacking = t == s.team
                for n, i in enumerate(self._team_indices(t, outfield=True)):
                    if i != s.taker and (self.roles[i] != "DEF" or not attacking):
                        targets[i] = self._tf(tactics.corner_box_target(n, attacking), t)
                        caps[i] = C.RUN_SPEED
        behind = np.array([-0.6 * self._dir(s.team), 0.0])
        targets[s.taker] = s.spot + behind
        caps[s.taker] = C.RUN_SPEED
        if s.kind in ("free_kick", "corner", "kickoff"):
            for j in self._team_indices(1 - s.team):  # opponents keep their distance
                rel = targets[j] - s.spot
                gap = float(np.linalg.norm(rel))
                if gap < 9.5:
                    rel = rel / gap if gap > 1e-6 else (self._own_goal(1 - s.team) - s.spot) / 50.0
                    targets[j] = s.spot + rel / (np.linalg.norm(rel) + 1e-9) * 9.5
        self._move(targets, caps)

        if self.time < s.resume_at:
            return
        if np.linalg.norm(self.pos[s.taker] - s.spot) > 1.2:
            if self.time < s.deadline:
                return
            self.pos[s.taker] = s.spot + behind  # took too long: place them
        self._resume()

    def _resume(self) -> None:
        s = self.stoppage
        self.stoppage = None
        self.ball_in_play = True
        self.possession = s.team
        self.owner = s.taker
        self.last_touch_team = s.team
        self.ball_pos[:] = (s.spot[0], s.spot[1], 0.0)
        self.ball_vel[:] = 0
        if s.kind == "kickoff":
            mids = [i for i in self._team_indices(s.team) if self.roles[i] == "MID"]
            receiver = mids[int(np.argmin([np.linalg.norm(self.pos[i] - s.spot) for i in mids]))]
            self._execute_pass(receiver, self.pos[receiver].copy(), lofted=False)
        elif s.kind == "corner":
            target = self._tf(np.array([self.rng.uniform(93, 99), self.rng.uniform(28, 40)]), s.team)
            mates = [i for i in self._team_indices(s.team, outfield=True) if i != s.taker]
            receiver = mates[int(np.argmin([np.linalg.norm(self.pos[i] - target) for i in mates]))]
            self._execute_pass(receiver, target, lofted=True)
        else:
            hold = 1.0 if s.kind == "goal_kick" else 0.5
            self.next_decision = self.time + self.rng.uniform(hold * 0.6, hold * 1.6)

    # ----------------------------------------------------------------------- control and passes

    def _check_control(self) -> None:
        ball_xy = self.ball_pos[:2]
        z = self.ball_pos[2]
        speed = float(np.linalg.norm(self.ball_vel))
        gaps = np.linalg.norm(self.pos - ball_xy, axis=1)
        for i in np.argsort(gaps)[:5]:
            keeper_zone = self.roles[i] == "GK" and self._in_own_box(i, ball_xy)
            if gaps[i] > (C.KEEPER_CONTROL_RADIUS if keeper_zone else C.CONTROL_RADIUS):
                continue
            if z > (C.KEEPER_MAX_HEIGHT if keeper_zone else C.CONTROL_MAX_HEIGHT):
                continue
            if speed > (40.0 if keeper_zone else C.CONTROL_MAX_SPEED):
                continue
            if i == self.last_kicker and self.time - self.last_kick_time < C.KICK_COOLDOWN:
                continue
            if i in self.screened and np.linalg.norm(ball_xy - self.kick_point) < C.BLOCK_RANGE + 1.0:
                continue  # the ball was played past this player
            cutting_out = self.active_pass is not None and self.team[i] != self.active_pass.team
            if cutting_out and not keeper_zone and speed > C.CLEAN_CONTROL_SPEED and self.rng.random() > 0.6:
                self._deflect(i)
                return
            self._gain_control(int(i), cause=self.loose_cause)
            return

    def _deflect(self, player: int) -> None:
        """A fast pass is cut out but not controlled: the ball spins off loose."""
        if self.active_pass is not None:
            self._end_pass("incomplete", self.ball_pos[:2], interceptor=player)
        angle = self.rng.uniform(-1.2, 1.2)
        vx, vy = self.ball_vel[0], self.ball_vel[1]
        c, s = math.cos(angle), math.sin(angle)
        self.ball_vel[:] = (0.35 * (c * vx - s * vy), 0.35 * (s * vx + c * vy), self.rng.uniform(0.5, 2.5))
        self._touched_by(player, cause="interception")

    def _end_pass(self, outcome: str, end_xy, recipient: int | None = None, interceptor: int | None = None) -> None:
        p = self.active_pass
        self.active_pass = None
        if outcome == "complete":
            self.stats["passes_complete"][p.team] += 1
        event_id = self._emit(
            PassEvent,
            ms=p.start_ms,
            team_id=self.team_ids[p.team],
            player_id=self.player_ids[p.passer],
            location=self._pt(p.start_xy),
            end_location=self._pt(end_xy),
            recipient_id=self.player_ids[recipient] if recipient is not None else None,
            outcome=outcome,
            height="high" if p.lofted else "ground",
            ball_speed_kmh=min(p.speed_kmh, 149.9),
            under_pressure=p.under_pressure,
        )
        if interceptor is not None:
            self._emit(
                InterceptionEvent,
                team_id=self.team_ids[self.team[interceptor]],
                player_id=self.player_ids[interceptor],
                location=self._pt(end_xy),
                intercepted_event_id=event_id,
                related_event_ids=[event_id],
            )

    def _gain_control(self, player: int, cause: str) -> None:
        t = self.team[player]
        ball_xy = self.ball_pos[:2].copy()
        if self.active_pass is not None:
            if self.active_pass.team == t:
                self._end_pass("complete", ball_xy, recipient=player)
            else:
                self._end_pass("incomplete", ball_xy, interceptor=player)
                cause = "interception"
        if t != self.possession:
            self._emit(
                PossessionChangeEvent,
                team_id=self.team_ids[t],
                player_id=self.player_ids[player],
                location=self._pt(ball_xy),
                from_team_id=self.team_ids[self.possession],
                to_team_id=self.team_ids[t],
                cause=cause,
            )
            self.possession = t
        self.owner = player
        self.carry_target = None
        self.last_touch_team = t
        self.loose_cause = "loose_ball"
        self.ball_vel[:] = 0
        first_touch = (1.5, 3.0) if self.roles[player] == "GK" else C.FIRST_TOUCH_TIME
        self.next_decision = self.time + self.rng.uniform(*first_touch)

    # ----------------------------------------------------------------------- duels

    def _duels(self) -> None:
        c = self.owner
        opponents = self._team_indices(1 - self.team[c], outfield=True)
        gaps = np.linalg.norm(self.pos[opponents] - self.pos[c], axis=1)
        nearest = opponents[int(np.argmin(gaps))]
        nearest_gap = float(np.min(gaps))

        # Pressure episodes
        if self.pressure is not None:
            p = self.pressure
            if p.target != c or np.linalg.norm(self.pos[p.presser] - self.pos[c]) > C.PRESSURE_END:
                self._end_pressure(forced=False)
        if self.pressure is None and nearest_gap < C.PRESSURE_START:
            self.pressure = Pressure(nearest, c, self.time, self._now_ms(), self.pos[c].copy())

        # Tackles
        for j, gap in zip(opponents, gaps, strict=True):
            if gap > C.TACKLE_RANGE or self.time < self.tackle_ready_at[j]:
                continue
            if self.rng.random() > C.TACKLE_RATE * C.DT:
                continue
            self.tackle_ready_at[j] = self.time + C.TACKLE_COOLDOWN
            self.stats["tackles"][self.team[j]] += 1
            p_win = C.clamp(0.42 + 0.3 * float(self.tackling[j] - self.dribbling[c]), 0.15, 0.75)
            common = {
                "team_id": self.team_ids[self.team[j]],
                "player_id": self.player_ids[j],
                "location": self._pt(self.pos[c]),
                "opponent_id": self.player_ids[c],
            }
            if self.rng.random() < p_win:
                self._emit(TackleEvent, outcome="won", **common)
                self._end_pressure(forced=True)
                if self.rng.random() < 0.5:  # clean: the tackler comes away with it
                    self.owner = None
                    self._gain_control(int(j), cause="tackle")
                else:  # poked loose
                    self.owner = None
                    self.carry_target = None
                    angle = self.rng.uniform(0, 2 * math.pi)
                    kick = self.rng.uniform(3, 7)
                    self.ball_vel[:] = (math.cos(angle) * kick, math.sin(angle) * kick, 0.0)
                    self._touched_by(int(j), cause="tackle")
            elif self.rng.random() < C.FOUL_SHARE:
                self._emit(TackleEvent, outcome="foul", **common)
                card = "yellow" if self.rng.random() < C.YELLOW_CARD_SHARE else None
                self._emit(
                    FoulEvent,
                    team_id=self.team_ids[self.team[j]],
                    player_id=self.player_ids[j],
                    location=self._pt(self.pos[c]),
                    victim_id=self.player_ids[c],
                    card=card,
                )
                self.stats["fouls"][self.team[j]] += 1
                spot = np.array(
                    [C.clamp(self.pos[c][0], 1, C.PITCH_LENGTH - 1), C.clamp(self.pos[c][1], 1, C.PITCH_WIDTH - 1)]
                )
                self._stop("free_kick", self.team[c], spot, C.FREE_KICK_DELAY, c)
            else:
                self._emit(TackleEvent, outcome="lost", **common)
            return

    def _end_pressure(self, forced: bool) -> None:
        p = self.pressure
        self.pressure = None
        if self.time - p.start_time < C.MIN_PRESSURE_DURATION:
            return
        self._emit(
            PressureEvent,
            ms=p.start_ms,
            team_id=self.team_ids[self.team[p.presser]],
            player_id=self.player_ids[p.presser],
            location=self._pt(p.location),
            target_player_id=self.player_ids[p.target],
            duration_ms=int((self.time - p.start_time) * 1000),
            forced_turnover=forced,
        )

    # ----------------------------------------------------------------------- summary

    def summary(self) -> dict:
        """Match stats, plus rates comparable with real football.

        Event counts are scaled per minute of ball in play, then to a real match's ~57 minutes
        of play, because restarts here are much quicker than in real matches.
        """
        minutes = 2 * self.half_seconds / 60
        s = self.stats
        outfield = [i for i in range(22) if self.roles[i] != "GK"]
        in_play_s = sum(s["possession_s"]) or 1.0
        per_match = C.REAL_BALL_IN_PLAY_MINUTES * 60 / in_play_s
        passes = sum(s["passes"])
        shots = sum(s["shots"])
        return {
            "minutes": round(minutes, 1),
            "score": f"{self.score[0]}-{self.score[1]}",
            "ball_in_play_pct": round(100 * in_play_s / (minutes * 60), 1),
            "possession_home_pct": round(100 * s["possession_s"][0] / in_play_s, 1),
            "pass_completion_pct": round(100 * sum(s["passes_complete"]) / max(passes, 1), 1),
            "on_target_pct": round(100 * sum(s["on_target"]) / max(shots, 1), 1),
            "xg_per_shot": round(sum(s["xg"]) / max(shots, 1), 3),
            "passes_per_match": round(passes * per_match),
            "shots_per_match": round(shots * per_match, 1),
            "goals_per_match": round(sum(self.score) * per_match, 2),
            "xg_per_match": round(sum(s["xg"]) * per_match, 2),
            "tackles_per_match": round(sum(s["tackles"]) * per_match),
            "fouls_per_match": round(sum(s["fouls"]) * per_match),
            "distance_km_per_90": round(float(np.mean(self.distance[outfield])) / 1000 * 90 / minutes, 2),
            "top_speed_kmh": round(float(np.max(self.top_speed)) * 3.6, 1),
        }
