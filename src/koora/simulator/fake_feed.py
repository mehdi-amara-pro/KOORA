"""
Fake feed: a few minutes of contract-valid StreamMessages so the stats engine and
frontend can be built on day 1, before the real simulator exists.

It is NOT a football model. Positions and outcomes are random but well-formed.

Usage:
    python -m koora.simulator.fake_feed --minutes 3 --seed 7 > feed.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Iterator
from datetime import UTC, datetime

from koora.contracts.schemas import (
    MatchClock,
    MatchMetadata,
    PassEvent,
    Player,
    Point,
    Position,
    PossessionChangeEvent,
    PressureEvent,
    Provenance,
    ShotEvent,
    Source,
    StreamMessage,
    Team,
)

MATCH_ID = "m-demo"
PROVENANCE = Provenance(source=Source.SIMULATOR, confidence=1.0, model_version="fake-feed-0.1")

FORMATION = [Position.GK] + [Position.DEF] * 4 + [Position.MID] * 3 + [Position.FWD] * 3


def _team(team_id: str, name: str, code: str, colours: tuple[str, str], surnames: list[str]) -> Team:
    players = [
        Player(
            player_id=f"p-{code.lower()}-{i + 1:02d}",
            team_id=team_id,
            name=f"Player {surname}",
            short_name=surname,
            shirt_number=i + 1,
            position=FORMATION[i],
        )
        for i, surname in enumerate(surnames)
    ]
    return Team(
        team_id=team_id,
        name=name,
        short_name=code,
        primary_colour=colours[0],
        secondary_colour=colours[1],
        players=players,
    )


HOME = _team(
    "t-hbr",
    "Harbour Rovers",
    "HBR",
    ("#1B4F8C", "#F2F2F2"),
    ["Okafor", "Lindqvist", "Ferreira", "Haddad", "Novak", "Mensah", "Duarte", "Kowal", "Benali", "Sato", "Reyes"],
)
AWAY = _team(
    "t-ash",
    "Ashford Athletic",
    "ASH",
    ("#A3262A", "#111111"),
    ["Brandt", "Achebe", "Moreau", "Silva", "Petrov", "Kaya", "Lund", "Amrani", "Costa", "Ito", "Walsh"],
)


def _point(rng: random.Random) -> Point:
    return Point(x=round(rng.uniform(5, 100), 1), y=round(rng.uniform(3, 65), 1))


def generate(minutes: int = 3, seed: int = 7) -> Iterator[StreamMessage]:
    rng = random.Random(seed)
    kickoff = datetime(2026, 10, 24, 15, 0, tzinfo=UTC)

    yield StreamMessage(
        kind="metadata",
        match_id=MATCH_ID,
        payload=MatchMetadata(
            match_id=MATCH_ID,
            competition="KOORA Demo League",
            kickoff=kickoff,
            venue="Harbour Park",
            home=HOME,
            away=AWAY,
            attacking_direction={HOME.team_id: {1: "+x", 2: "-x"}, AWAY.team_id: {1: "-x", 2: "+x"}},
            provenance=PROVENANCE,
        ),
    )

    seq = 0
    t_ms = 0
    team, opponent = HOME, AWAY
    end_ms = minutes * 60_000

    while t_ms < end_ms:
        t_ms += rng.randint(1_500, 6_000)
        clock = MatchClock(period=1, period_ms=t_ms)
        seq += 1
        passer = rng.choice(team.players[1:])
        roll = rng.random()

        if roll < 0.70:
            receiver = rng.choice([p for p in team.players if p is not passer])
            complete = rng.random() < 0.82
            event = PassEvent(
                event_id=f"e-{seq:05d}",
                match_id=MATCH_ID,
                seq=seq,
                clock=clock,
                team_id=team.team_id,
                player_id=passer.player_id,
                location=_point(rng),
                end_location=_point(rng),
                recipient_id=receiver.player_id if complete else None,
                outcome="complete" if complete else "incomplete",
                height=rng.choice(["ground", "ground", "low", "high"]),
                ball_speed_kmh=round(rng.uniform(25, 85), 1),
                under_pressure=rng.random() < 0.3,
                provenance=PROVENANCE,
            )
            yield StreamMessage(kind="event", match_id=MATCH_ID, payload=event)
            if not complete:
                seq += 1
                yield StreamMessage(
                    kind="event",
                    match_id=MATCH_ID,
                    payload=PossessionChangeEvent(
                        event_id=f"e-{seq:05d}",
                        match_id=MATCH_ID,
                        seq=seq,
                        clock=clock,
                        team_id=opponent.team_id,
                        from_team_id=team.team_id,
                        to_team_id=opponent.team_id,
                        cause="interception",
                        related_event_ids=[event.event_id],
                        provenance=PROVENANCE,
                    ),
                )
                team, opponent = opponent, team

        elif roll < 0.90:
            presser = rng.choice(opponent.players[1:])
            yield StreamMessage(
                kind="event",
                match_id=MATCH_ID,
                payload=PressureEvent(
                    event_id=f"e-{seq:05d}",
                    match_id=MATCH_ID,
                    seq=seq,
                    clock=clock,
                    team_id=opponent.team_id,
                    player_id=presser.player_id,
                    location=_point(rng),
                    target_player_id=passer.player_id,
                    duration_ms=rng.randint(400, 2_500),
                    forced_turnover=False,
                    provenance=PROVENANCE,
                ),
            )

        else:
            shooter = rng.choice(team.players[6:])
            outcome = rng.choices(["goal", "saved", "off_target", "blocked"], weights=[1, 3, 4, 2])[0]
            yield StreamMessage(
                kind="event",
                match_id=MATCH_ID,
                payload=ShotEvent(
                    event_id=f"e-{seq:05d}",
                    match_id=MATCH_ID,
                    seq=seq,
                    clock=clock,
                    team_id=team.team_id,
                    player_id=shooter.player_id,
                    location=Point(x=round(rng.uniform(80, 100), 1), y=round(rng.uniform(20, 48), 1)),
                    end_location={"x": 105.0, "y": round(rng.uniform(28, 40), 1), "z": round(rng.uniform(0, 2.4), 2)},
                    outcome=outcome,
                    body_part=rng.choice(["left_foot", "right_foot", "head"]),
                    shot_speed_kmh=round(rng.uniform(60, 125), 1),
                    provenance=PROVENANCE,
                ),
            )
            team, opponent = opponent, team


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    for message in generate(args.minutes, args.seed):
        sys.stdout.write(message.model_dump_json(exclude_none=True) + "\n")


if __name__ == "__main__":
    main()
