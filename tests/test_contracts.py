from koora.contracts.schemas import Insight, StoryAtom, StreamMessage, _examples
from koora.simulator.fake_feed import generate


def test_examples_round_trip():
    for example in _examples().values():
        assert type(example).model_validate_json(example.model_dump_json()) == example


def test_fake_feed_is_contract_valid_and_ordered():
    messages = list(generate(minutes=3, seed=1))
    assert messages[0].kind == "metadata"
    events = [m for m in messages if m.kind == "event"]
    assert events, "feed should contain events"
    seqs = [m.payload.seq for m in events]
    assert seqs == sorted(seqs)
    for m in messages:
        StreamMessage.model_validate_json(m.model_dump_json())


def test_atoms_and_insights_carry_evidence():
    examples = _examples()
    atom: StoryAtom = examples["story_atom"]
    insight: Insight = examples["insight"]
    assert atom.evidence_event_ids
    assert set(insight.evidence_atom_ids) == {atom.atom_id}
