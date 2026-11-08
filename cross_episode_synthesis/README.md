# 🎬 Cross-Episode Synthesis Engine

This directory contains the production implementation of the **Cross-Episode Synthesis Engine** for the 50-episode microdrama series *"I Married a Broken Soldier. He was a Duke"*.

---

## 🚀 Architectural Upgrade: 3-Check Narrative Compatibility Pipeline

### The Fundamental Problem
A naive approach relies exclusively on dense vector embeddings (`text-embedding-004`) and cosine similarity. While effective at finding thematic resonance, **semantic similarity does not necessarily mean narrative continuity**:

* *Episode 1:* A woman is publicly humiliated at the altar.
* *Episode 20:* A woman is threatened in a dark corridor.
* *Episode 42:* The protagonist reveals her royal identity in ducal armor.

Moments 1 and 2 often score high cosine similarity ($\operatorname{sim} \ge 0.85$) because both depict high-intensity female subjugation and betrayal vocabulary. However, **they involve disjoint characters and have zero narrative continuity**, making them ineffective when stitched into a promotional trailer.

### The 3-Check Solution
To solve this, we introduce a **three-tiered lightweight narrative compatibility pipeline**:

```text
Cross-Episode Candidate Retrieval
│
├── [Check 1: Semantic Similarity Retrieval]
│   └── Are the moments contextually related? (Cosine similarity >= 0.75 via text-embedding-004)
│
├── [Check 2: Character & Entity Compatibility Gate]
│   └── Do they involve the same relevant characters, objects, or relationships?
│       - Resolves character aliases (e.g., "The Broken Soldier" -> "Alexander", "Lady Juliana" -> "Juliana")
│       - Checks pivotal props & clues (Signet Ring, Bridal Gown, Herbal Grimoire)
│       - Audits active relationship graph (Spouses, Betrayal, Adversaries)
│       - Drops disjoint storyline beats
│
└── [Check 3: LLM Narrative Progression Check]
    └── Does the second moment meaningfully escalate, resolve, or invert the first?
        - Evaluates progression dynamics:
          * power_inversion: e.g., altar humiliation -> ducal vindication
          * revelation_or_twist: e.g., secret clue recognized -> identity unmasked
          * setup_to_payoff: e.g., marriage vow -> battle alliance
          * escalation_of_stakes: e.g., whipping subjugation -> overt defiance
          * none: parallel or redundant threats without story movement (REJECTED)
        - Structured Pydantic validation via Gemini 2.5 Flash (with deterministic fallback)
│
└── 🎯 Eligible Cross-Episode Combinations (High-converting promo trailer blueprints)
```

| Check | Target Question | Technical Mechanism | Rejection Criteria |
|---|---|---|---|
| **1. Semantic Similarity** | Are the moments contextually related? | Dense vector cosine similarity ($\ge 0.75$) with minimum 2-episode gap | Cosine score $< 0.75$ or within same episode window |
| **2. Entity Compatibility** | Do they share characters, props, or relationships? | Canonical alias normalization, key prop tracking, relationship graph audit | Disjoint characters and no shared narrative clues |
| **3. Narrative Progression** | Does Moment 2 escalate, resolve, or invert Moment 1? | Structured LLM critic + deterministic archetype transition engine | `progression_type == "none"` (redundant or non-progressing) |

---

## 📁 Directory Structure

```text
cross_episode_synthesis/
├── __init__.py                                 # Package export symbols
├── synthesizer.py                              # Standalone Python module & CLI
├── cross_episode_synthesis.ipynb               # Interactive Jupyter notebook
├── embeddings_cache.json                       # Cached vector embeddings (text-embedding-004)
├── 03_synthesized_candidates.json              # Canonical output with 3-check audit trace
└── README.md                                   # Comprehensive documentation & architecture guide
```

---

## ⚙️ Strict Mechanical Invariants Verified

Every synthesized candidate is audited against 5 non-negotiable Python invariants:
1. **AST Existence:** All segment `start_element_id` and `end_element_id` exist in `01_structured_episodes.json`.
2. **Monotonicity:** $0 \le \text{start\_cs} < \text{end\_cs}$ within each segment.
3. **Episode Breadth:** Segments must span $\ge 2$ distinct episodes.
4. **Hook Timing:** $0.0\text{s} \le \text{hook\_timestamp\_offset\_seconds} \le 3.0\text{s}$ (grounded in Segment 1).
5. **Combined Duration:** Strictly $20.00\text{s} \le \text{total\_duration\_seconds} \le 60.00\text{s}$ ($2,000 \le \text{total\_duration\_cs} \le 6,000$).

---

## 🛠️ Usage Instructions

### Run Synthesis with Diversity Ranking
```powershell
# Execute 3-check cross-episode synthesis with diversity-aware selection
.venv\Scripts\python.exe cross_episode_synthesis/synthesizer.py --diversity-rank

# Offline / Deterministic mode
.venv\Scripts\python.exe cross_episode_synthesis/synthesizer.py --no-llm --diversity-rank
```
