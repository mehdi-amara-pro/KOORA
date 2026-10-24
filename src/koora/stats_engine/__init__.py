"""Stats engine: consumes StreamMessages, computes metrics, emits StoryAtoms and rule-based Insights.

Deterministic and sub-second. No LLM calls on this path.
Inputs:  koora.contracts.StreamMessage
Outputs: koora.contracts.StoryAtom, koora.contracts.Insight (origin="rule")
"""
