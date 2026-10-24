"""
Ball physics and the formulas built on it.

The ball is a point mass with gravity, quadratic air drag, rolling resistance and bounces.
On the ground the 1D deceleration is  a = -(R + K v^2), which has closed-form solutions,
so pass planning is exact for ground passes. Lofted balls are solved numerically.
"""

from __future__ import annotations

import math

import numpy as np

from koora.simulator import config as C

R = C.ROLLING_DECEL
K = C.AIR_DRAG


# --------------------------------------------------------------------------- stepping


def step_ball(pos: np.ndarray, vel: np.ndarray, dt: float = C.DT, substeps: int = C.BALL_SUBSTEPS) -> None:
    """Advance the ball by dt, in place. pos and vel are float arrays of shape (3,): x, y, z."""
    h = dt / substeps
    for _ in range(substeps):
        on_ground = pos[2] <= 1e-6 and abs(vel[2]) < 0.6
        speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)

        if on_ground:
            pos[2] = 0.0
            vel[2] = 0.0
            hs = math.hypot(vel[0], vel[1])
            decel = (R + K * hs * hs) * h
            if hs <= decel:  # resistance would reverse the ball: it has stopped
                vel[0] = vel[1] = 0.0
            else:
                scale = (hs - decel) / hs
                vel[0] *= scale
                vel[1] *= scale
        else:
            # Semi-implicit Euler: update velocity first, then position with the new velocity.
            vel -= K * speed * vel * h
            vel[2] -= C.GRAVITY * h

        pos += vel * h

        if pos[2] < 0.0:  # hit the ground
            pos[2] = 0.0
            if vel[2] < -0.6:
                vel[2] = -vel[2] * C.BOUNCE_RESTITUTION
                vel[0] *= C.BOUNCE_FRICTION
                vel[1] *= C.BOUNCE_FRICTION
            else:
                vel[2] = 0.0


# --------------------------------------------------------------------------- ground passes


def ground_kick_speed(distance: float, arrive_speed: float) -> float:
    """Kick speed so a rolling ball covers `distance` and arrives at `arrive_speed`.

    From v dv/ds = -(R + K v^2):   R + K v0^2 = (R + K v1^2) * exp(2 K s)
    """
    v0_sq = ((R + K * arrive_speed**2) * math.exp(2 * K * distance) - R) / K
    return math.sqrt(max(v0_sq, 0.0))


def ground_stop_distance(v0: float) -> float:
    """How far a rolling ball travels before stopping: s = ln((R + K v0^2) / R) / (2 K)."""
    return math.log((R + K * v0**2) / R) / (2 * K)


def ground_speed_at(v0: float, s: np.ndarray | float) -> np.ndarray:
    """Speed of a rolling ball after travelling s metres. NaN where the ball stops before s."""
    v_sq = ((R + K * v0**2) * np.exp(-2 * K * np.asarray(s)) - R) / K
    with np.errstate(invalid="ignore"):
        return np.where(v_sq > 0, np.sqrt(np.maximum(v_sq, 0)), np.nan)


def ground_time_to(v0: float, s: np.ndarray | float) -> np.ndarray:
    """Time for a rolling ball kicked at v0 to travel s metres (inf if it stops first).

    From dv/dt = -(R + K v^2):   t = (atan(v0 c) - atan(v c)) / sqrt(R K),  c = sqrt(K / R)
    """
    c = math.sqrt(K / R)
    v = ground_speed_at(v0, s)
    t = (math.atan(v0 * c) - np.arctan(v * c)) / math.sqrt(R * K)
    return np.where(np.isnan(v), np.inf, t)


# --------------------------------------------------------------------------- lofted balls


def flight(v0: float, angle: float, dt: float = 0.01) -> tuple[float, float]:
    """Horizontal distance and time until a lofted ball first lands. angle in radians."""
    vh, vz = v0 * math.cos(angle), v0 * math.sin(angle)
    x = z = t = 0.0
    while True:
        speed = math.hypot(vh, vz)
        vh -= K * speed * vh * dt
        vz -= (K * speed * vz + C.GRAVITY) * dt
        x += vh * dt
        z += vz * dt
        t += dt
        if z <= 0.0 or t > 10.0:
            return x, t


def lofted_kick_speed(distance: float, angle: float) -> tuple[float, float]:
    """Kick speed (m/s) and flight time (s) for a lofted ball to land `distance` away.

    Binary search on the same drag model the simulation uses, so the ball really lands there.
    """
    lo, hi = 4.0, 40.0
    for _ in range(20):
        mid = (lo + hi) / 2
        reach, _ = flight(mid, angle)
        if reach < distance:
            lo = mid
        else:
            hi = mid
    v0 = (lo + hi) / 2
    return v0, flight(v0, angle)[1]


def height_at(vh: float, vz: float, distance: float, dt: float = 0.005) -> tuple[float, float]:
    """Height and time when a ball kicked from the ground (vh, vz) has travelled `distance` horizontally."""
    x = z = t = 0.0
    while x < distance and t < 5.0:
        speed = math.hypot(vh, vz)
        vh -= K * speed * vh * dt
        vz -= (K * speed * vz + C.GRAVITY) * dt
        x += vh * dt
        z += vz * dt
        t += dt
        if z < 0:  # bounced: keep it simple and treat it as rolling on
            z, vz = 0.0, abs(vz) * C.BOUNCE_RESTITUTION
    return z, t


def shot_lift(vh: float, distance: float, height: float) -> float:
    """Vertical kick speed so a shot crosses `distance` metres at `height` (binary search)."""
    lo, hi = 0.0, 20.0
    for _ in range(18):
        mid = (lo + hi) / 2
        z, _ = height_at(vh, mid, distance)
        if z < height:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def straight_shot_time(distance: float, speed: float) -> float:
    """Flight time of a fast, low ball under drag only: x(t) = ln(1 + K v0 t) / K."""
    return (math.exp(K * distance) - 1.0) / (K * speed)


# --------------------------------------------------------------------------- shot quality


def expected_goals(x: float, y: float) -> float:
    """Expected goals of a shot from (x, y) in the team frame (attacking the goal at x = 105).

    Logistic model of the visible goal angle and distance. Roughly:
    penalty spot ~0.32, 25 m central ~0.07, tight angles near zero.
    """
    dx = C.PITCH_LENGTH - x
    if dx <= 0:
        return 0.0
    dy = y - C.CENTRE_Y
    distance = math.hypot(dx, dy)
    angle = abs(math.atan2(C.GOAL_HALF_WIDTH - dy, dx) - math.atan2(-C.GOAL_HALF_WIDTH - dy, dx))
    return 1.0 / (1.0 + math.exp(-(-2.0 + 3.0 * angle - 0.06 * distance)))
