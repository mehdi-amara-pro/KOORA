"""
Where players want to be when they are not chasing the ball.

Everything here works in the *team frame*: the team always attacks towards x = 105,
whatever the real direction. The engine converts to and from pitch coordinates.
"""

from __future__ import annotations

import numpy as np

from koora.simulator import config as C

# 4-3-3. Offsets (depth, width) in metres relative to the centre of the team's block.
SLOT_NAMES = ["GK", "LB", "LCB", "RCB", "RB", "LCM", "CM", "RCM", "LW", "ST", "RW"]
SLOT_OFFSETS: dict[str, tuple[float, float]] = {
    "LB": (-14, -24),
    "LCB": (-19, -9),
    "RCB": (-19, 9),
    "RB": (-14, 24),
    "LCM": (-4, -13),
    "CM": (-9, 0),
    "RCM": (-4, 13),
    "LW": (12, -24),
    "ST": (16, 0),
    "RW": (12, 24),
}


def shape_target(slot: str, ball: np.ndarray, in_possession: bool) -> np.ndarray:
    """Formation spot that shifts with the ball.

    In possession the team spreads wide and pushes up; out of possession it drops
    behind the ball and squeezes narrow and short around it.
    """
    bx, by = ball
    if in_possession:
        cx = C.clamp(bx + 2, 35, 72)
        cy = C.CENTRE_Y + 0.30 * (by - C.CENTRE_Y)
        depth_scale, width_scale = 1.0, 1.0
    else:
        cx = C.clamp(bx - 10, 28, 62)
        cy = C.CENTRE_Y + 0.50 * (by - C.CENTRE_Y)
        depth_scale, width_scale = 0.8, 0.65
    dx, dy = SLOT_OFFSETS[slot]
    x = C.clamp(cx + dx * depth_scale, 3, 102)
    y = C.clamp(cy + dy * width_scale, 2, C.PITCH_WIDTH - 2)
    return np.array([x, y])


def kickoff_target(slot: str, kicking: bool) -> np.ndarray:
    """Kick-off positions: everyone in their own half, opponents outside the centre circle."""
    if slot == "GK":
        return np.array([4.0, C.CENTRE_Y])
    dx, dy = SLOT_OFFSETS[slot]
    x = 33 + dx
    y = C.CENTRE_Y + dy * 0.9
    limit = 50.5 if kicking else C.CENTRE_X - C.CENTRE_CIRCLE_RADIUS - 0.5
    return np.array([min(x, limit), y])


def keeper_target(ball: np.ndarray) -> np.ndarray:
    """Goalkeeper: between the ball and the centre of the goal, stepping out a little."""
    bx, by = ball
    x = C.clamp(3 + 0.07 * bx, 2, 12)
    y = C.clamp(C.CENTRE_Y + 0.22 * (by - C.CENTRE_Y), C.CENTRE_Y - 3, C.CENTRE_Y + 3)
    return np.array([x, y])


def onside(target: np.ndarray, ball: np.ndarray, last_defender_x: float) -> np.ndarray:
    """Attackers do not run beyond the last defender unless the ball is already past them."""
    limit = max(ball[0], last_defender_x) - 0.5
    return np.array([min(target[0], limit), target[1]])


def corner_box_target(index: int, attacking: bool) -> np.ndarray:
    """Spread players around the penalty area for a corner (team frame of the attacking team)."""
    if attacking:
        return np.array([91.0 + (index % 3) * 3.0, 27.0 + (index % 5) * 3.5])
    return np.array([4.0 + (index % 3) * 3.0, 27.0 + (index % 5) * 3.5])  # defending own box
