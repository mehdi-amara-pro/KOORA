"""
Turn a match file (JSON lines of StreamMessages) into a compact replay for the 3D viewer.

The full file repeats every player ID in every frame, so a 10-minute match is ~35 MB.
The replay sends positions as plain number arrays at a lower frame rate (~3 MB),
and keeps every event. Frame rows are:

    [t, period, ball_x, ball_y, ball_z, owner_index, x0, y0, x1, y1, ..., speed0, speed1, ...]

t is seconds since kick-off across both halves; owner_index is -1 when nobody has the ball.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DROP_EVENT_FIELDS = {"provenance", "match_id", "clock", "related_event_ids", "schema_version"}


def build_replay(path: Path, target_hz: float = 12.5) -> dict:
    return _build_cached(str(path), path.stat().st_mtime, target_hz)


@lru_cache(maxsize=8)
def _build_cached(path_str: str, _mtime: float, target_hz: float) -> dict:
    players: list[dict] = []
    index: dict[str, int] = {}
    teams: dict = {}
    frames: list[list[float]] = []
    events: list[dict] = []
    period_offset = {1: 0.0}
    last_period, last_period_seconds = 1, 0.0
    min_gap = 1.0 / target_hz - 1e-6
    last_frame_t = -1e9

    with open(path_str, encoding="utf-8") as f:
        for line in f:
            message = json.loads(line)
            kind = message["kind"]
            p = message["payload"]

            if kind == "metadata":
                for side in ("home", "away"):
                    team = p[side]
                    teams[side] = {
                        "team_id": team["team_id"],
                        "name": team["name"],
                        "code": team["short_name"],
                        "primary": team["primary_colour"],
                        "secondary": team["secondary_colour"],
                    }
                    for player in team["players"]:
                        index[player["player_id"]] = len(players)
                        players.append(
                            {
                                "id": player["player_id"],
                                "side": side,
                                "number": player["shirt_number"],
                                "name": player["name"],
                                "short": player["short_name"],
                                "position": player["position"],
                            }
                        )
                continue

            period = p["clock"]["period"]
            seconds = p["clock"]["period_ms"] / 1000.0
            if period != last_period:
                period_offset[period] = period_offset[last_period] + last_period_seconds
                last_period, last_period_seconds = period, 0.0
            last_period_seconds = max(last_period_seconds, seconds)
            t = period_offset.get(period, 0.0) + seconds

            if kind == "frame":
                if t - last_frame_t < min_gap:
                    continue
                last_frame_t = t
                ball = p["ball"]
                n = len(players)
                xy = [0.0] * (2 * n)
                speeds = [0.0] * n
                for pf in p["players"]:
                    k = index.get(pf["player_id"])
                    if k is None:
                        continue
                    xy[2 * k] = pf["position"]["x"]
                    xy[2 * k + 1] = pf["position"]["y"]
                    speeds[k] = pf.get("speed_kmh") or 0.0
                owner = index.get(ball.get("owner_player_id"), -1)
                pos = ball["position"]
                frames.append([round(t, 2), period, pos["x"], pos["y"], pos.get("z", 0.0), owner, *xy, *speeds])

            elif kind == "event":
                event = {key: value for key, value in p.items() if key not in DROP_EVENT_FIELDS}
                event["t"] = round(t, 2)
                event["period"] = period
                events.append(event)

    return {
        "match_id": Path(path_str).stem,
        "teams": teams,
        "players": players,
        "fps": target_hz,
        "frames": frames,
        "events": sorted(events, key=lambda e: (e["t"], e.get("seq", 0))),
    }
