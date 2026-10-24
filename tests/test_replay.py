import pytest
from fastapi.testclient import TestClient

import koora.api.main as api
from koora.api.replay import build_replay
from koora.simulator.engine import MatchSimulator


@pytest.fixture(scope="module")
def match_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("matches")
    with (folder / "unit-test.jsonl").open("w", encoding="utf-8") as f:
        for line in MatchSimulator(minutes=1, seed=9, match_id="unit-test").run():
            f.write(line + "\n")
    return folder


def test_replay_is_compact_and_consistent(match_dir):
    replay = build_replay(match_dir / "unit-test.jsonl")
    n = len(replay["players"])
    assert n == 22
    assert replay["teams"]["home"]["code"] == "HBR"
    frames = replay["frames"]
    assert len(frames) == pytest.approx(60 * 12.5, abs=3)  # 1 minute at 12.5 fps
    assert all(len(f) == 6 + 3 * n for f in frames)
    times = [f[0] for f in frames]
    assert times == sorted(times)
    assert times[-1] == pytest.approx(60, abs=0.2)  # second half continues the clock
    assert any(e["type"] == "match_end" for e in replay["events"])


def test_api_lists_and_serves_matches(match_dir, monkeypatch):
    monkeypatch.setattr(api, "DATA_DIR", match_dir)
    client = TestClient(api.app)
    assert client.get("/api/matches").json()[0]["id"] == "unit-test"
    response = client.get("/api/matches/unit-test/replay")
    assert response.status_code == 200
    assert len(response.json()["players"]) == 22
    assert client.get("/api/matches/missing/replay").status_code == 404
    assert client.get("/api/matches/..%2Fsecret/replay").status_code in (400, 404)


def test_viewer_page_is_served():
    client = TestClient(api.app)
    response = client.get("/viewer.html")
    assert response.status_code == 200
    assert "three" in response.text
