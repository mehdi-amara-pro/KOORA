# Assumptions

These shape the whole design. Change them early, as a team, and update `src/koora/contracts/schemas.py` to match.

## Data

- All data is synthetic and football-realistic. No real Premier League match data, clubs, players or trademarks.
- "Live" means a simulated feed replayed at match speed.
- Every record carries provenance: a `source` (simulator, tracking, video, derived) and a `confidence` from 0 to 1.

## Input modes

- **Tracking mode:** optical tracking at 25 Hz with known player identities and high confidence.
- **Video mode:** a single broadcast camera with partial pitch visibility. Players may be off-camera (`visible = false`), identities can swap, speeds are noisier, and confidence is lower. Downstream code softens or withholds claims accordingly.

## Geometry and time

- Pitch is 105 × 68 m. Origin (0, 0) is the corner at the home team's defensive left in period 1; x runs along the length, y across the width.
- Coordinates never flip at half time. `MatchMetadata.attacking_direction` gives each team's direction per period.
- Match time is a period number plus milliseconds since that period kicked off. Wall-clock time is kept only to measure latency.

## Performance

- Rule-based overlays (tags, speed and distance badges): under one second from event.
- Agent narratives: a few seconds. Insights carry a TTL and are dropped if stale.
- Alert thresholds (for example, a sprint above 30 km/h) are configurable, not learned.

## Scale

- Event Hubs partition key is `match_id`, so matches run in parallel.
- Insights are generated per audience segment (persona × language × club × player × metric), not per viewer.
