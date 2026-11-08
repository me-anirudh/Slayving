"""
Cross-Episode Synthesis Package
Synthesizes high-converting multi-episode promotional trailers spanning 2 or more distinct episodes:
- 3-Check Narrative Compatibility Architecture:
  1. Semantic Similarity Retrieval via Vector Embeddings & Cosine Similarity
  2. Character & Entity Compatibility Check (Canonical Aliases, Key Props, Relationships)
  3. LLM / Deterministic Narrative Progression Check (Escalation, Revelation/Twist, Power Inversion, Payoff)
- Diversity-Aware Final Ranking & Greedy Selection
- Pydantic Structured Output Validation & Controlled Retries
- Precise Trim Recommendations & Transitional Editing Instructions
"""

from .synthesizer import (
    TrimRecommendation,
    RubricScores,
    StitchedSegment,
    MultiEpisodeCandidate,
    SynthesisLLMResponse,
    EntityCompatibilityResult,
    ProgressionVerdict,
    EligibleCombination,
    SemanticMomentRetriever,
    CrossEpisodeSynthesizerEngine,
    diversity_aware_greedy_selection,
)

__all__ = [
    "TrimRecommendation",
    "RubricScores",
    "StitchedSegment",
    "MultiEpisodeCandidate",
    "SynthesisLLMResponse",
    "EntityCompatibilityResult",
    "ProgressionVerdict",
    "EligibleCombination",
    "SemanticMomentRetriever",
    "CrossEpisodeSynthesizerEngine",
    "diversity_aware_greedy_selection",
]
