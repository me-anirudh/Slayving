# 🎬 Quality Validation & Ranking Engine (`validator_ranker`)

This directory contains the production implementation of the **Quality Validation & Editorial Ranking Engine** for the 50-episode microdrama series *"I Married a Broken Soldier. He was a Duke"*.

It enforces the strict **5-Step Sequential Pipeline** designed to eliminate ranking ambiguity, curse of knowledge bias, and duplicate clutter while guaranteeing high-converting, editor-ready promotional candidates.

---

## 📁 Directory Structure

```text
validator_ranker/
├── __init__.py                                 # Package export symbols
├── validator_ranker.py                         # Standalone Python module & CLI
├── validator_ranker.ipynb               # Interactive Jupyter notebook with pre-executed outputs
├── final_promos.json                           # Canonical final Top 15 output artifact (Page 14 schema)
├── editorial_report.md                         # Executive & editorial analysis report
├── execution_trace.json                        # End-to-end mechanical audit trace across all 5 steps
└── README.md                                   # Comprehensive documentation & pipeline architecture guide
```

---

## 🚀 Strict 5-Step Pipeline Implementation

### Step 1 — Mechanical Gate (Duration & Hook Bounds)
* **Inputs:** Pools all single-episode candidates (`02_batch_extractions.json`) and multi-episode candidates (`03_synthesized_candidates.json`).
* **Validation Invariants:**
  1. **AST Existence & Monotonicity:** Every segment `start_element_id` and `end_element_id` is verified against the 3,727 canonical AST elements in `01_structured_episodes.json` with $0 \le \text{start\_cs} < \text{end\_cs}$.
  2. **Total Duration Gate:** Strictly $2,000\text{ cs} \le \text{duration\_cs} \le 6,000\text{ cs}$ ($20.00\text{s} - 60.00\text{s}$). Invalid durations ($< 20.00\text{s}$ or $> 60.00\text{s}$) are rejected outright and recorded in `execution_trace.json` rather than auto-extending to prevent cliffhanger resolution risks.
  3. **Hook Timing Gate:** $0.0\text{s} \le \text{hook\_timestamp\_offset\_seconds} \le 3.0\text{s}$, verifying that the opening hook element initiates within $300\text{ cs}$ of clip start.

### Step 2 — High-Overlap Deduplication (Containment & Climax)
* **Pairwise Comparison:** Evaluates same-episode candidates and drops the lower-scoring clip if:
  $$\text{containment} = \frac{\text{intersection\_cs}}{\min(\text{dur\_a\_cs}, \text{dur\_b\_cs})} \ge 0.85$$
  **OR** if overlap $\ge 0.60$ and both candidates share the exact same `end_element_id` (shared climax).
* Preserves distinct dramatic moments while removing repetitive overlapping variations.

### Step 3 — Deterministic Rubric Sort (Lexicographic Key)
* **Lexicographic Sort Key:** Sorts all surviving candidates using the exact tuple:
  ```python
  passes_min_floor = (
      c.standalone_coherence >= 4
      and c.hook_strength >= 3
      and c.cliffhanger_strength >= 3
  )
  rubric_sort_key = (
      passes_min_floor,        # Boolean: True sorts higher than False
      c.standalone_coherence,  # Primary priority (1..5)
      c.hook_strength,         # Secondary priority (1..5)
      c.cliffhanger_strength,  # Tertiary priority (1..5)
      c.emotional_impact,      # Quaternary priority (1..5)
      c.total_rubric_score     # Sum of the 4 scores (4..20)
  )
  ```
* Ensures high cold coherence ($\ge 4$) and visceral hooks take absolute priority over raw sums.

### Step 4 — Cold-First Quality Critic (Isolated Clip Buffer)
* **20-Candidate Buffer Optimization:** Evaluates a 20-candidate buffer instead of just 15 so that if up to 5 clips are demoted, replacements are already Critic-approved and avoid `UNREVIEWED_BACKFILL` tags.
* **Cold-First Prompt Structure:**
  - **(a) Isolated Clip Transcript first:** Verbatim dialogue and action lines without backstory.
  - **(b) $\pm 3$ Surrounding Elements & Episode Logline second:** Screens for spoilers and continuity issues.
  - *Eliminates "Curse of Knowledge" bias:* forces the Critic to experience the clip as a cold mobile viewer first.
* **5 Editorial Questions Audited:**
  1. `Cold Comprehension`: Understandable to a cold viewer?
  2. `3-Second Opening`: Explosive visceral hook in $\le 3.0$s?
  3. `Unresolved Ending`: Cuts at peak jeopardy?
  4. `Spoiler / Over-Reveal Guard`: Avoids premature plot resolution?
  5. `Multi-Episode Continuity`: Preserves authentic dialogue meaning across episodes?
* Verdict: `"APPROVE"`, `"FLAG_WITH_OBJECTION"`, or `"DEMOTE"`. Raw response saved to `raw_llm_outputs/quality_critic_review.json`.

### Step 5 — Demotion & Explicit Backfill
* Candidates receiving `"DEMOTE"` are removed from the top tier.
* The Top 15 are filled from the highest-ranked surviving candidates.
* Any replacement drawn from within the 20-candidate buffer inherits its verified Critic status.
* Any unreviewed candidate drawn from rank 21+ is explicitly labeled:
  `"critic_status": "UNREVIEWED_BACKFILL"`.
* Emits final deliverables conforming to the Page 14 concrete schema.

---

## 🛠️ Usage Instructions

### Option A: Interactive Jupyter Notebook
Open and run `validator_ranker/validator_ranker.ipynb` in VS Code or JupyterLab.
All 19 cells are pre-executed with complete outputs, markdown diagrams, and inspection tables.

### Option B: Standalone Python CLI
Run the ranking engine directly:

```powershell
# Standard run
python validator_ranker/validator_ranker.py

# Custom paths and buffer size
python validator_ranker/validator_ranker.py --ast 01_structured_episodes.json --extractions 02_batch_extractions.json --synthesized 03_synthesized_candidates.json --buffer 20
```

---

## 📊 Summary of Final Deliverables

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| **`final_promos.json`** | [`validator_ranker/final_promos.json`](file:///c:/Users/Anirudh%20Singh/OneDrive/Documents/Kuku%20FM/validator_ranker/final_promos.json) | Top 15 ranked promotional blueprints conforming to Page 14 schema with timestamps, segment trims, rubric scores, editorial rationales, and critic review. |
| **`editorial_report.md`** | [`validator_ranker/editorial_report.md`](file:///c:/Users/Anirudh%20Singh/OneDrive/Documents/Kuku%20FM/validator_ranker/editorial_report.md) | Executive and marketing analysis report with archetype breakdowns and segment-level trim blueprints for video editors. |
| **`execution_trace.json`** | [`validator_ranker/execution_trace.json`](file:///c:/Users/Anirudh%20Singh/OneDrive/Documents/Kuku%20FM/validator_ranker/execution_trace.json) | Complete mechanical audit logging raw candidate pool (27), Step 1 rejections (2), Step 2 deduplications (2), Step 3 rubric rankings, Step 4 critic reviews, and Step 5 final 15 selection. |
