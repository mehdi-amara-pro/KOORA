"""Agent team (Microsoft Agent Framework + Foundry Agent Service).

Narrator  writes from StoryAtoms per audience segment and language.
Verifier  rejects any claim not supported by the atom's metrics, facts and evidence.
Producer  decides what reaches the screen: priority, de-duplication, pacing, TTL.

Inputs:  koora.contracts.StoryAtom
Outputs: koora.contracts.Insight (origin="agent")
"""
