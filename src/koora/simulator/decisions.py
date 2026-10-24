"""
On-ball decisions: score every option, then pick one with a softmax.

Each option gets a success probability and a value:

    value = p * threat(where the ball ends up) - (1 - p) * cost of losing it there

Pass success comes from a "time to intercept" race: for points along the ball's path,
compare when the ball arrives with when each opponent could get there.
All geometry is in the team frame (attacking towards x = 105).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from koora.simulator import config as C
from koora.simulator import physics


@dataclass
class Option:
    kind: str  # "pass", "carry" or "shoot"
    value: float
    success: float
    receiver: int | None = None
    target: np.ndarray | None = None  # team frame
    lofted: bool = False


@dataclass
class DecisionContext:
    pos: np.ndarray  # (22, 2) team-frame positions
    vel: np.ndarray  # (22, 2) team-frame velocities
    team: np.ndarray  # (22,) 0 or 1
    roles: list[str]
    max_speed: np.ndarray
    carrier: int
    ball: np.ndarray  # team frame
    passing: float
    pressured: bool
    rng: np.random.Generator


def threat(xy: np.ndarray) -> np.ndarray:
    """How dangerous it is to have the ball at xy (a simple expected-threat grid)."""
    xy = np.asarray(xy, dtype=float)
    fx = np.clip(xy[..., 0] / C.PITCH_LENGTH, 0, 1)
    off_centre = np.minimum(1.0, np.abs(xy[..., 1] - C.CENTRE_Y) / C.CENTRE_Y)
    central = 1.0 - 0.7 * off_centre
    value = 0.005 + 0.12 * fx**3 * central
    in_box = (xy[..., 0] > C.PITCH_LENGTH - C.PENALTY_AREA_DEPTH) & (
        np.abs(xy[..., 1] - C.CENTRE_Y) < C.PENALTY_AREA_HALF_WIDTH
    )
    return value + np.where(in_box, 0.08 * central, 0.0)


def loss_cost(xy: np.ndarray) -> np.ndarray:
    """Cost of losing the ball at xy: worse the closer it is to our own goal."""
    xy = np.asarray(xy, dtype=float)
    fx = np.clip(xy[..., 0] / C.PITCH_LENGTH, 0, 1)
    return 0.02 + 0.10 * (1 - fx) ** 3


def option_value(p: float, target: np.ndarray, lose_at: np.ndarray) -> float:
    """Expected value of an action that succeeds with probability p.

    Success: the ball is at `target` and we keep possession (worth POSSESSION_VALUE on top).
    Failure: we lose the ball at `lose_at` and the possession with it.
    """
    gain = float(threat(target)) + C.POSSESSION_VALUE
    loss = float(loss_cost(lose_at)) + C.POSSESSION_VALUE
    return p * gain - (1 - p) * loss


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp_to_pitch(xy: np.ndarray) -> np.ndarray:
    return np.array([C.clamp(xy[0], 1, C.PITCH_LENGTH - 1), C.clamp(xy[1], 1, C.PITCH_WIDTH - 1)])


def time_to_reach(gaps: np.ndarray, max_speed: np.ndarray) -> np.ndarray:
    """Time for a player to get the ball within reach. Already in reach: no time at all."""
    reach = np.maximum(gaps - C.CONTROL_RADIUS, 0.0)
    return np.where(reach > 0, C.REACTION_TIME + reach / max_speed, 0.0)


def ground_pass_success(ctx: DecisionContext, receiver: int, target: np.ndarray, opponents: np.ndarray) -> float:
    """Probability a ground pass reaches the receiver.

    1. Find where the receiver meets the ball: the first point on the path they reach before it.
    2. Before that point, any opponent who can get there before the ball cuts it out.
    3. At the meeting point, the receiver contests with the nearest opponent.
    """
    ball = ctx.ball
    distance = float(np.linalg.norm(target - ball))
    v0 = min(28.0, physics.ground_kick_speed(distance, 6.0))

    # Points along the path. Opponents right next to the passer are covered by block_factor().
    start = min(C.BLOCK_RANGE + 1.0, 0.5 * distance)
    s = np.linspace(start, distance, 12)
    t_ball = physics.ground_time_to(v0, s)
    points = ball + np.outer(s / distance, target - ball)

    gaps = np.linalg.norm(points[:, None, :] - ctx.pos[opponents][None, :, :], axis=2)
    t_opp = np.min(time_to_reach(gaps, ctx.max_speed[opponents][None, :]), axis=1)
    receiver_gap = np.linalg.norm(points - ctx.pos[receiver], axis=1)
    t_recv = 0.1 + np.maximum(receiver_gap - C.CONTROL_RADIUS, 0.0) / ctx.max_speed[receiver]

    meets = np.nonzero(t_recv <= t_ball)[0]
    meet = int(meets[0]) if len(meets) else len(s) - 1

    # +0.2 s: a moving ball is harder to cut out cleanly than it is to reach
    path_margin = float(np.min(t_opp[:meet] - t_ball[:meet])) + 0.2 if meet > 0 else np.inf
    # +0.3 s: the receiver moves onto the ball and shields it; a marker has to go round them
    contest_margin = float(t_opp[meet] - t_recv[meet]) + 0.3

    margin = min(path_margin, contest_margin)
    execution = 0.995 - 0.004 * distance * (1.0 - ctx.passing)
    return _sigmoid(margin / 0.3) * execution * block_factor(ctx, target, opponents)


def block_factor(ctx: DecisionContext, target: np.ndarray, opponents: np.ndarray) -> float:
    """Chance the kick gets past opponents standing next to the passer.

    Only opponents roughly in the direction of the pass can block it.
    """
    heading = target - ctx.ball
    heading /= np.linalg.norm(heading) + 1e-9
    factor = 1.0
    for j in opponents:
        rel = ctx.pos[j] - ctx.ball
        gap = float(np.linalg.norm(rel))
        if gap < C.BLOCK_RANGE and float(np.dot(rel, heading)) > -0.3 * gap:
            factor *= 1.0 - C.BLOCK_CHANCE
    return factor


def lofted_pass_success(ctx: DecisionContext, receiver: int, target: np.ndarray, opponents: np.ndarray) -> float:
    distance = float(np.linalg.norm(target - ctx.ball))
    t_opp = time_to_reach(np.linalg.norm(ctx.pos[opponents] - target, axis=1), ctx.max_speed[opponents])
    t_receiver = 0.1 + float(np.linalg.norm(ctx.pos[receiver] - target)) / ctx.max_speed[receiver]
    # Defenders track the receiver during the flight, so the race is closer than it looks at kick time
    margin = float(np.min(t_opp)) - t_receiver - 0.25
    execution = 0.45 - 0.004 * distance * (1.0 - ctx.passing)  # controlling a dropping ball is hard
    return _sigmoid(margin / 0.45) * execution * block_factor(ctx, target, opponents)


def evaluate(ctx: DecisionContext) -> list[Option]:
    c = ctx.carrier
    my_team = ctx.team[c]
    mates = [i for i in range(len(ctx.team)) if ctx.team[i] == my_team and i != c]
    opponents = np.array([i for i in range(len(ctx.team)) if ctx.team[i] != my_team])
    ball = ctx.ball
    is_keeper = ctx.roles[c] == "GK"
    options: list[Option] = []

    # --- passes
    for k in mates:
        if ctx.roles[k] == "GK" and not (ball[0] < 35 and ctx.pressured):
            continue  # back-passes to the keeper only from deep and under pressure
        target = _clamp_to_pitch(ctx.pos[k] + ctx.vel[k] * 0.5)  # lead the receiver
        distance = float(np.linalg.norm(target - ball))
        if distance < 4:
            continue
        if distance <= 40 and not (is_keeper and distance > 30):
            p = ground_pass_success(ctx, k, target, opponents)
            options.append(Option("pass", option_value(p, target, target), p, k, target, lofted=False))
        if 25 <= distance <= 55:
            p = lofted_pass_success(ctx, k, target, opponents)
            options.append(Option("pass", option_value(p, target, target), p, k, target, lofted=True))

    if is_keeper:
        return options

    # --- carry: dribble ~6 m forward, drifting away from the nearest opponent
    gaps = np.linalg.norm(ctx.pos[opponents] - ball, axis=1)
    nearest = opponents[int(np.argmin(gaps))]
    nearest_gap = float(np.min(gaps))
    direction = np.array([1.0, 0.0])
    if nearest_gap < 6:
        away = ball - ctx.pos[nearest]
        direction = direction + 0.6 * away / (np.linalg.norm(away) + 1e-6)
    direction /= np.linalg.norm(direction)
    carry_to = _clamp_to_pitch(ball + direction * 6.0)
    carry_to = np.array([min(carry_to[0], C.PITCH_LENGTH - 2), C.clamp(carry_to[1], 2, C.PITCH_WIDTH - 2)])
    p_keep = _sigmoid((nearest_gap - 2.5) / 1.0) * 0.97
    options.append(Option("carry", option_value(p_keep, carry_to, ball), p_keep, target=carry_to))

    # --- shot
    if ball[0] > C.MIN_SHOT_X and abs(ball[1] - C.CENTRE_Y) < 25:
        xg = physics.expected_goals(ball[0], ball[1]) * (0.75 if ctx.pressured else 1.0)
        # A shot usually ends the possession, but rebounds and corners keep some of its value
        options.append(Option("shoot", xg * C.SHOT_VALUE_FACTOR - 0.5 * C.POSSESSION_VALUE * (1 - xg), xg))

    return options


def choose_action(ctx: DecisionContext) -> tuple[Option, list[Option]]:
    """Pick an option with a softmax over values. Returns the choice and everything considered."""
    options = evaluate(ctx)
    if not options:  # boxed in with no sensible option: keep the ball
        return Option("carry", 0.0, 0.5, target=ctx.ball.copy()), []
    values = np.array([o.value for o in options])
    weights = np.exp((values - values.max()) / C.DECISION_TEMPERATURE)
    choice = options[int(ctx.rng.choice(len(options), p=weights / weights.sum()))]
    return choice, options
