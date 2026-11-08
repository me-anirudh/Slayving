# 🎬 Phase 4: Cross-Episode Synthesis Engine

This directory contains the production implementation of **Phase 4: Cross-Episode Synthesis** for the 50-episode microdrama series *"I Married a Broken Soldier. He was a Duke"*, enhanced with all four high-priority architectural improvements:

1. **Semantic Retrieval using Embeddings & Cosine Similarity (Improvement 1.1)**
2. **Diversity-Aware Final Ranking & Greedy Selection (Improvement 1.2)**
3. **Structured Output Validation & Controlled Retries (Improvement 1.3)**
4. **Precise Trim Recommendations & Transitional Editing Instructions (Improvement 1.4)**

---

## 📁 Directory Structure

```text
phase4_synthesis/
├── __init__.py                                 # Package export symbols
├── synthesizer.py                              # Standalone Python module & CLI
├── phase4_cross_episode_synthesis.ipynb        # Interactive Jupyter notebook with pre-executed outputs
├── embeddings_cache.json                       # Cached vector embeddings of narrative moments
├── 03_synthesized_candidates.json              # Canonical Phase 4 output artifact
└── README.md                                   # Comprehensive documentation & architecture guide
```

---

## 🚀 Key Improvements Implemented

### 1.1 Semantic Retrieval Using Embeddings (High Priority)
* **Problem:** Simple case-insensitive keyword/alias matching fails when screenplay events describe the same story beat using different vocabulary (e.g., *"soldier examines royal signet"* in Ep 7 vs. *"ancestral seal is presented before court"* in Ep 42).
* **Implementation:**
  - Extracts rich semantic text representation for each narrative moment (summary + moment type + keywords + AST verbatim lines).
  - Generates dense vector embeddings using Google GenAI `text-embedding-004` (with deterministic unit-norm vector fallback for offline execution).
  - Caches embeddings in `embeddings_cache.json` on disk to avoid redundant API consumption.
  - Computes pairwise cosine similarity:
    $$\operatorname{sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$
  - Retrieves the top 3–5 semantically related cross-episode moments and attaches verbatim source screenplay elements directly from `01_structured_episodes.json` before passing them to the LLM.

### 1.2 Diversity-Aware Final Ranking (High Priority)
* **Problem:** Ranking solely by raw sum of rubric scores leads to heavy storyline clustering (e.g., 10 recommendations all coming from Episode 42).
* **Implementation:**
  - Implements a **Greedy Diversity Selection Algorithm**:
    - **Episode Concentration:** Maximum 3 recommendations per episode (for multi-episode promos, each constituent episode count increments).
    - **Archetype Diversity:** Guaranteed representation across all 3 standardized promo archetypes (`immediate_shock_opener`, `visual_anomaly_mystery_escalation`, `emotional_reversal_turning_point`).
    - **Composition Diversity:** Ensures a balanced mix of single-episode and cross-episode promos.
    - **Quality First:** Never selects a low-quality clip (overall score $< 14$) merely to fill a diversity quota.
  - Logs every selection decision and skip reason in a transparent audit log.

### 1.3 LLM Output Reliability & Controlled Retries (High Priority)
* **Problem:** LLM responses can contain invalid JSON, missing mandatory fields, incorrect element IDs, or scores out of range.
* **Implementation:**
  - Enforces strict Pydantic v2 schemas with constrained types (`Literal`, `Field(ge=1, le=5)`).
  - Uses Gemini provider-supported structured output (`response_schema=SynthesisLLMResponse`, `response_mime_type="application/json"`).
  - Implements a **Controlled Retry Harness**:
    1. Captures raw response text.
    2. Intercepts schema `ValidationError`.
    3. Builds a concise, targeted correction prompt pointing out the exact field errors.
    4. Retries once with the correction prompt.
    5. Gracefully logs and falls back to deterministic AST-grounded synthesis if retries fail.

### 1.4 Precise Trim Recommendations & Editing Instructions (High Priority)
* **Problem:** Dialogue elements often contain trailing lines or dead air, but character-count interpolation without audio waveforms is ungrounded.
* **Implementation:**
  - Every candidate and constituent segment includes structured `TrimRecommendation`:
    - `exact_opening_dialogue`: Verbatim spoken line (or initial visual description) that begins the clip.
    - `exact_ending_dialogue`: Verbatim spoken line that ends the clip right before the cliffhanger cut.
    - `verbatim_trim_quote`: Exact internal phrase/sentence for sentence-level internal cuts.
    - `boundary_type`: `"element_boundary"` vs `"internal_cut"`.
    - `requires_manual_editor_verification`: Flag indicating whether the cut requires audio waveform check.
    - `transition_instructions`: Step-by-step editing directives between constituent segments (e.g., sound effects, match cuts, pacing).

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

### Option A: Interactive Jupyter Notebook
Open and run `phase4_synthesis/phase4_cross_episode_synthesis.ipynb` in VS Code or JupyterLab.
All cells are pre-executed with complete outputs, markdown diagrams, and inspection tables.

### Option B: Standalone Python CLI
Run the synthesizer engine directly:

```powershell
# Standard run with diversity-aware final ranking
python phase4_synthesis/synthesizer.py --diversity-rank

# Custom paths or model selection
python phase4_synthesis/synthesizer.py --ast 01_structured_episodes.json --extractions 02_batch_extractions.json --output phase4_synthesis/03_synthesized_candidates.json --diversity-rank
```

### Environment Variables
- `GEMINI_API_KEY`: *(Optional)* If set, calls Gemini 2.5 Flash for live cross-episode synthesis and text embeddings. If unset, automatically executes the deterministic AST-grounded semantic synthesis engine with 100% mechanical validity.
