import hashlib
import json

import pytest

from koora.contracts.schemas import StreamMessage
from koora.simulator.engine import MatchSimulator


@pytest.fixture(scope="module")
def match_lines() -> list[str]:
    return list(MatchSimulator(minutes=3, seed=5).run())


def test_every_message_matches_the_contracts(match_lines):
    for line in match_lines:
        StreamMessage.model_validate_json(line)


def test_structure_and_ordering(match_lines):
    messages = [json.loads(line) for line in match_lines]
    assert messages[0]["kind"] == "metadata"
    frames = [m["payload"] for m in messages if m["kind"] == "frame"]
    events = [m["payload"] for m in messages if m["kind"] == "event"]

    assert len(frames) == 3 * 60 * 25  # 3 minutes at 25 Hz
    assert [f["frame_id"] for f in frames] == sorted(f["frame_id"] for f in frames)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))

    types = {e["type"] for e in events}
    assert {"period_start", "period_end", "match_end", "pass", "possession_change"} <= types
    assert events[-1]["type"] == "match_end"


def test_players_stay_near_the_pitch_and_move_at_human_speeds(match_lines):
    for line in match_lines[1::50]:
        m = json.loads(line)
        if m["kind"] != "frame":
            continue
        for p in m["payload"]["players"]:
            assert -2 <= p["position"]["x"] <= 107
            assert -2 <= p["position"]["y"] <= 70
            assert p["speed_kmh"] <= 37  # nobody outruns a top sprinter


def test_same_seed_same_match():
    def digest(seed: int) -> str:
        h = hashlib.sha256()
        for line in MatchSimulator(minutes=0.5, seed=seed).run():
            h.update(line.encode())
        return h.hexdigest()

    assert digest(3) == digest(3)
    assert digest(3) != digest(4)


def test_frame_every_thins_frames_but_keeps_events():
    full = list(MatchSimulator(minutes=1, seed=2).run())
    thin = list(MatchSimulator(minutes=1, seed=2, frame_every=5).run())
    count = lambda lines, kind: sum(1 for x in lines if f'"kind":"{kind}"' in x)  # noqa: E731
    assert count(thin, "frame") == count(full, "frame") // 5
    assert count(thin, "event") == count(full, "event")


def test_summary_has_realistic_shape():
    sim = MatchSimulator(minutes=6, seed=7)
    for _ in sim.run():
        pass
    s = sim.summary()
    assert 65 <= s["pass_completion_pct"] <= 95
    assert 500 <= s["passes_per_match"] <= 1600
    assert 7 <= s["distance_km_per_90"] <= 15
