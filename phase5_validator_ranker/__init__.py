"""
Phase 5: Ranking & Quality Validation Package (validator_ranker)
Strict 5-Step Pipeline:
- Step 1: Mechanical Gate (Fix 2 & Fix 4)
- Step 2: High-Overlap Deduplication (Fix 5)
- Step 3: Deterministic Rubric Sort (Fix 3)
- Step 4: Cold-First Quality Critic with 20-Candidate Buffer (Fix 7 & Revision 5)
- Step 5: Demotion & Explicit Backfill (Top 15 Final Promos)
"""

from .validator_ranker import (
    PromoSegment,
    EditorialRationale,
    CriticReview,
    PromoCandidate,
    CriticSingleCandidateReview,
    CriticBatchReview,
    ValidatorRankerEngine,
    main,
)

__all__ = [
    "PromoSegment",
    "EditorialRationale",
    "CriticReview",
    "PromoCandidate",
    "CriticSingleCandidateReview",
    "CriticBatchReview",
    "ValidatorRankerEngine",
    "main",
]
