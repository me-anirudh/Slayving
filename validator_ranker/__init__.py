"""
Quality Validation, Ranking & Editorial Delivery Package
Executes 4-step sequential filtration and editorial ranking:
- Step 1: Mechanical Invariant Gate
- Step 2: High-Overlap Deduplication
- Step 3: Deterministic Rubric Sort
- Step 4: Cold-First Quality Critic with 20-Candidate Buffer
"""

from .validator_ranker import (
    PromoCandidate,
    CandidateProvenance,
    MechanicalAuditRecord,
    CriticEvaluation,
    ValidatorRankerEngine,
)

__all__ = [
    "PromoCandidate",
    "CandidateProvenance",
    "MechanicalAuditRecord",
    "CriticEvaluation",
    "ValidatorRankerEngine",
]
