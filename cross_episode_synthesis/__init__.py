"""
Cross-Episode Synthesis Package
Synthesizes high-converting multi-episode promotional trailers spanning 2 or more distinct episodes:
- Semantic Retrieval using Embeddings & Cosine Similarity
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
    SemanticMomentRetriever,
    CrossEpisodeSynthesizerEngine,
)

__all__ = [
    "TrimRecommendation",
    "RubricScores",
    "StitchedSegment",
    "MultiEpisodeCandidate",
    "SynthesisLLMResponse",
    "SemanticMomentRetriever",
    "CrossEpisodeSynthesizerEngine",
]
