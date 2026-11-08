"""
Phase 4: Cross-Episode Synthesis Package
Enhanced with:
- Improvement 1.1: Semantic Retrieval Using Embeddings & Cosine Similarity
- Improvement 1.2: Diversity-Aware Final Ranking & Greedy Selection
- Improvement 1.3: Pydantic Structured Output Validation & Controlled Retries
- Improvement 1.4: Precise Trim Recommendations & Transitional Editing Instructions
"""

from .synthesizer import (
    TrimRecommendation,
    RubricScores,
    StitchedSegment,
    MultiEpisodeCandidate,
    SynthesisLLMResponse,
    SemanticMomentRetriever,
    CrossEpisodeSynthesizerEngine,
    diversity_aware_greedy_selection,
    main,
)

__all__ = [
    "TrimRecommendation",
    "RubricScores",
    "StitchedSegment",
    "MultiEpisodeCandidate",
    "SynthesisLLMResponse",
    "SemanticMomentRetriever",
    "CrossEpisodeSynthesizerEngine",
    "diversity_aware_greedy_selection",
    "main",
]
