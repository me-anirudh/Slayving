"""
Quality Validation, Ranking & Editorial Delivery Package
Executes 5-step sequential filtration and editorial ranking:
- Step 1: Mechanical Invariant Gate
- Step 2: High-Overlap Deduplication
- Step 3: Deterministic Rubric Sort
- Step 4: Cold-First Quality Critic with 20-Candidate Buffer
- Step 5: Demotion & Explicit Backfill
"""

from .validator_ranker import (
    TrimRecommendation,
    PromoSegment,
    RubricScores,
    EditorialRationale,
    CriticReview,
    PromoCandidate,
    ValidatorRankerEngine,
)

__all__ = [
    "TrimRecommendation",
    "PromoSegment",
    "RubricScores",
    "EditorialRationale",
    "CriticReview",
    "PromoCandidate",
    "ValidatorRankerEngine",
]
