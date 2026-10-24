import math

import numpy as np
import pytest

from koora.simulator import config as C
from koora.simulator import physics


def roll(v0: float, until_speed: float = 0.0) -> float:
    """Distance a ball rolls in the simulation before dropping to `until_speed`."""
    pos = np.zeros(3)
    vel = np.array([v0, 0.0, 0.0])
    for _ in range(5000):
        if np.linalg.norm(vel) <= until_speed:
            break
        physics.step_ball(pos, vel)
    return float(pos[0])


def test_dropped_ball_falls_about_4_9_m_in_one_second():
    pos = np.array([0.0, 0.0, 10.0])
    vel = np.zeros(3)
    for _ in range(C.TICK_HZ):
        physics.step_ball(pos, vel)
    assert 10.0 - pos[2] == pytest.approx(4.9, abs=0.15)  # a little less than 4.905 because of drag


def test_rolling_ball_stops_where_the_closed_form_says():
    v0 = 15.0
    assert roll(v0) == pytest.approx(physics.ground_stop_distance(v0), rel=0.03)


def test_ground_pass_arrives_at_planned_speed():
    distance, arrive = 25.0, 6.0
    v0 = physics.ground_kick_speed(distance, arrive)
    assert roll(v0, until_speed=arrive) == pytest.approx(distance, rel=0.03)


def test_ground_speed_never_increases():
    s = np.linspace(0, 40, 50)
    v = physics.ground_speed_at(20.0, s)
    v = v[~np.isnan(v)]
    assert np.all(np.diff(v) <= 0)


def test_lofted_pass_lands_on_target():
    angle = math.radians(35)
    v0, t = physics.lofted_kick_speed(40.0, angle)
    reach, _ = physics.flight(v0, angle)
    assert reach == pytest.approx(40.0, abs=0.3)
    assert 1.0 < t < 4.0


def test_shot_crosses_goal_line_at_aimed_height():
    vz = physics.shot_lift(25.0, 18.0, 1.8)
    z, _ = physics.height_at(25.0, vz, 18.0)
    assert z == pytest.approx(1.8, abs=0.05)


def test_expected_goals_is_sensible():
    penalty_spot = physics.expected_goals(94.0, 34.0)
    long_range = physics.expected_goals(80.0, 34.0)
    tight_angle = physics.expected_goals(103.0, 10.0)
    assert 0.2 < penalty_spot < 0.45
    assert long_range < penalty_spot
    assert tight_angle < 0.05
    assert physics.expected_goals(106.0, 34.0) == 0.0
