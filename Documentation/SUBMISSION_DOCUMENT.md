# 🎬 Kuku AI Generalist Assignment: Solution Submission
**Track:** Problem 1 — Find the Promos (Ads) Across Microdrama Series  
**Candidate Name:** Anirudh Singh  
**Series Evaluated:** *"I Married a Broken Soldier. He was a Duke"* (50 Episodes, 3,727 Screenplay Elements)  
**Deliverables Produced:** [`final_promos.json`](file:///c:/Users/Anirudh%20Singh/OneDrive/Documents/Kuku%20FM/final_promos.json) (Top 15 Promo Blueprints) | [`editorial_report.md`](file:///c:/Users/Anirudh%20Singh/OneDrive/Documents/Kuku%20FM/editorial_report.md) | [`execution_trace.json`](file:///c:/Users/Anirudh%20Singh/OneDrive/Documents/Kuku%20FM/execution_trace.json)  

---

### 1. Why I Selected "Find the Promos"
In the direct-to-consumer microdrama economy, user acquisition is won or lost on paid short-form video ads (Instagram Reels, TikTok, YouTube Shorts). A compelling promotional ad creative fatigues within 3 to 7 days, yet human video editors spend 15 to 25 hours per series scrubbing through scripts to find viable cuts. 

Finding promos is not a generic summarization task; it requires balancing three conflicting narrative mechanics:
1. **The 3-Second Hook:** Stopping the thumb scroll with an immediate, visceral conflict ($\le 3.0\text{s}$).
2. **Cold Viewer Clarity:** Making complete sense to a cold user who has never seen the show (no inside jokes, ungrounded pronouns, or confusing backstory).
3. **The Unresolved Cliffhanger:** Cutting hard at the absolute climax of tension before resolution, compelling the viewer to install the Kuku FM app.

This problem represents the quintessential AI engineering challenge: **large language models excel at emotional nuance and narrative contrast, but fail at sub-second arithmetic, duration constraints, and objective deduplication.** I selected this track to build a reliable hybrid system that eliminates LLM hallucinations and provides editor-ready cut blueprints.

---

### 2. What I Built
I designed and implemented the **Kuku Promo Finder**, a production-grade 5-phase engine that ingests 50 raw screenplay text files, parses them into a mathematically verifiable Abstract Syntax Tree (AST), extracts single-episode and cross-episode candidate clips, and subjects them to an uncompromising mechanical and editorial evaluation pipeline.

#### Key System Deliverables:
- **`01_structured_episodes.json`:** 3,727 atomic screenplay elements across 50 episodes with integer centisecond timestamps ($1\text{s} = 100\text{cs}$) and tag decomposition (Camera, Action, Dialogue, Sound FX).
- **`02_batch_extractions.json`:** 24 single-episode candidate clips, 11 AST-enriched cross-episode narrative moments, and a flat narrative context store (character aliases, key clues, relationships).
- **`03_synthesized_candidates.json`:** Multi-episode promo arcs stitched across disparate episodes using dense embeddings (`text-embedding-004`) and cosine similarity.
- **`final_promos.json`:** Top 15 ranked promotional blueprints strictly conforming to Kuku's concrete output schema, detailing exact start/end elements, verbatim opening and cliffhanger spoken lines, rubric scores, editorial rationales, and critic verdicts.
- **`editorial_report.md` & `execution_trace.json`:** Executive marketing analysis and an exhaustive mechanical audit trace tracking every dropped duplicate and rejection.

---

### 3. How the Solution Works: The 5-Step Pipeline
The system enforces a strict sequential filtration and ranking architecture:

```
[27 Raw Candidates Pooled: 24 Single-Episode + 3 Multi-Episode]
                            │
                            ▼
[Step 1: Mechanical Gate] ────────────────► Verifies AST IDs, monotonicity, duration (20-60s), hook (<=3s).
                            │               Rejects 2 duration violations (<20s) -> 25 pass.
                            ▼
[Step 2: High-Overlap Deduplication] ─────► Pairwise containment test (>=85% or >=60% with shared climax).
                            │               Drops 2 duplicate variations -> 23 pass.
                            ▼
[Step 3: Deterministic Rubric Sort] ──────► Lexicographic tuple key: passes_floor, coherence, hook, cliffhanger.
                            │               Sorts surviving 23 candidates by cold clarity and visceral hook.
                            ▼
[Step 4: Cold-First Quality Critic] ──────► Gemini 2.5 Flash reviews a 20-candidate buffer.
                            │               Isolated clip transcript evaluated FIRST to kill "curse of knowledge".
                            ▼
[Step 5: Demotion & Explicit Backfill] ───► Demotes rejections; emits final Top 15 (0 backfilled unreviewed).
```

1. **Step 1 — Mechanical Gate:** Validates duration ($2,000\text{ cs} \le \text{dur} \le 6,000\text{ cs}$) and hook timing ($\le 300\text{ cs}$). Rejects out-of-bounds candidates outright rather than auto-extending boundaries, which risks resolving cliffhangers.
2. **Step 2 — Deduplication:** Computes exact mathematical containment: $\text{containment}(A, B) = \frac{\text{intersection\_cs}}{\min(\text{dur}_A, \text{dur}_B)}$. Drops clips with $\ge 85\%$ overlap or $\ge 60\%$ overlap sharing the exact same climax element.
3. **Step 3 — Deterministic Rubric Sort:** Uses a strict Python lexicographic tuple key ensuring standalone coherence ($\ge 4$) and visceral hooks ($\ge 3$) supersede raw summed scores.
4. **Step 4 — Cold-First Quality Critic:** Eliminates cognitive bias by presenting the isolated transcript *first* (cold mobile perspective) and $\pm 3$ surrounding elements and logline *second* (spoiler/continuity guard).
5. **Step 5 — Demotion & Backfill:** Any candidate admitted from rank 21+ to replace a demoted clip is explicitly tagged `"critic_status": "UNREVIEWED_BACKFILL"`.

---

### 4. Division of Labor: What AI Does vs. What Conventional Code Does

| System Component | Conventional Deterministic Code | Generative AI / Large Language Models |
| :--- | :--- | :--- |
| **Screenplay Parsing** | Regex tokenization, tag decomposition, and integer centisecond conversion ($1\text{s} = 100\text{cs}$) to prevent float drift. | *None.* (LLMs are barred from parsing to guarantee 100% mechanical consistency). |
| **Candidate Discovery** | Pre-computes and prints element durations directly in prompt transcripts. | `gemini-2.5-flash` identifies dramatic turning points aligned with the 3 standardized archetypes. |
| **Cross-Episode Linking**| Computes pairwise vector cosine similarity: $\operatorname{sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$ over cached embeddings. | Stitches disparate moments (e.g., Ep 1 Humiliation $\to$ Ep 42 Ducal Triumph) with audio/video transition directives. |
| **Duration & Hook Bounds**| Strictly enforces $20.00\text{s} \le \text{dur} \le 60.00\text{s}$ and $\text{hook\_offset} \le 3.00\text{s}$. Rejects violations. | Returns `hook_timestamp_offset_seconds` and identifies verbatim opening/cliffhanger lines. |
| **Deduplication & Sorting**| Calculates mathematical containment ($\ge 85\%$) and executes multi-key lexicographic sorting. | *None.* (LLM sorting is subjective; Python code ensures deterministic, reproducible rankings). |
| **Editorial Critic** | Constructs inverted context prompts; intercepts schema errors for controlled retries. | Audits isolated clips against the 5 Editorial Questions (Comprehension, Hook, Cliffhanger, Spoilers, Continuity). |

---

### 5. Core Assumptions Made
1. **Screenplay Element Grounding:** Screenplay cues (`**CAMERA:**`, `**ACTION:**`, `**DIALOGUE:**`) accurately reflect intended scene timing, but sub-element sentence cuts cannot be calculated via character interpolation without audio waveforms. The system provides verbatim cut quotes and flags internal trims for editor waveform verification.
2. **Cold Viewer Comprehension:** A promo must stand entirely on its own. If an ad requires the viewer to know who "Rupert" is without establishing his antagonistic status in the clip, the candidate is penalized.
3. **Curse of Knowledge Bias:** Evaluators who know the whole story find subtle references compelling that leave cold viewers confused. Inverting the critic prompt structure is essential to simulate true user acquisition.

---

### 6. Output Evaluation & Mechanical Audit
Across all 50 episodes of *"I Married a Broken Soldier. He was a Duke"*:
- **Parsing Coverage:** 50/50 episodes (3,727 elements) parsed with zero timing errors.
- **Step 1 Mechanical Gate:** 27 candidates evaluated $\to$ 2 rejected ($17.50\text{s}$ and $2.80\text{s}$) $\to$ 25 passed.
- **Step 2 Deduplication:** 2 overlapping variations dropped ($\text{containment} = 1.000$) $\to$ 23 passed.
- **Final Top 15 Selection:** 13 single-episode promos and 2 multi-episode promos (`PROMO_011` Ep 1 & 42; `PROMO_012` Ep 4 & 42).
- **Archetype Representation:** 6 `immediate_shock_opener`, 3 `visual_anomaly_mystery_escalation`, 6 `emotional_reversal_turning_point`.
- **Quality & Invariant Compliance:** 100% of final promos satisfy the $20\text{s}–60\text{s}$ gate (mean: $26.89\text{s}$) and initiate hooks within $\le 1.50\text{s}$. Critic Approved: 15/15; Unreviewed Backfill: 0.

---

### 7. Failure Modes & Limitations (Where the Tool Fails)
1. **Text-Only Modality:** The tool evaluates written screenplays. Directorial pacing, awkward actor delivery, or poor physical blocking in the actual filmed video cannot be detected from text alone. Human editor review remains necessary.
2. **Premature Cliffhanger Cuts:** In dense dialogues, an element boundary may cut 1–2 seconds *after* an ultimatum is answered. The system mitigates this by generating `TrimRecommendation.verbatim_trim_quote` to instruct editors where to make internal cuts.
3. **Inter-Batch Plot Separation:** Fixed 10-episode batching could miss an arc connecting Episode 10 to Episode 11 if not captured in the extracted narrative moments.

---

### 8. Production Scaling to Kuku's Workflow
- **Economics:** At ~140,000 input tokens and ~13,000 output tokens per 50 episodes using Gemini 2.5 Flash, the complete pipeline costs **~\$0.015 (1.5 cents USD) per season**.
- **Velocity:** Batch extractions can be parallelized asynchronously across Cloud Run workers, reducing end-to-end execution from 70 seconds to under 15 seconds per series.
- **Automated Ad Ops Integration:** Final deliverables export directly to Edit Decision Lists (EDL/XML) for Premiere and DaVinci Resolve. Performance data (CTR, 3s hook view rate, CPI) from Meta and TikTok Ads APIs can be piped back to dynamically optimize rubric weights.
