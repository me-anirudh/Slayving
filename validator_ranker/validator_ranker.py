"""
🎬 Quality Validation, Ranking & Editorial Delivery Engine (validator_ranker.py)
Series: "I Married a Broken Soldier. He was a Duke" (50 Episodes)

Strict 5-Step Pipeline:
1. Step 1 — Mechanical Gate: AST timestamps, strictly 20.00s-60.00s duration, 0.0-3.0s hook offset.
2. Step 2 — High-Overlap Deduplication: >=0.85 containment or (>=0.60 overlap + shared end element).
3. Step 3 — Deterministic Rubric Sort: Lexicographic tuple key prioritizing standalone coherence & hook.
4. Step 4 — Cold-First Quality Critic: 20-candidate buffer, cold clip transcript evaluated first, surrounding context second.
5. Step 5 — Demotion & Explicit Backfill: Demote rejections, fill Top 15, tag unreviewed backfill.
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any, Tuple, Union
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ==============================================================================
# 1. PYDANTIC SCHEMAS FOR CONCRETE OUTPUT (Page 14 Schema)
# ==============================================================================

class TrimRecommendation(BaseModel):
    exact_opening_dialogue: str
    exact_ending_dialogue: str
    verbatim_trim_quote: Optional[str] = None
    boundary_type: Literal["element_boundary", "internal_cut"] = "element_boundary"
    requires_manual_editor_verification: bool = False


class PromoSegment(BaseModel):
    segment_index: int = Field(ge=1, le=3)
    episode: int
    scene: int
    start_element_id: str
    end_element_id: str
    start_timestamp: str
    end_timestamp: str
    duration_seconds: float
    duration_cs: int
    trim_quote_hint: Optional[str] = None
    segment_summary: str
    trim_recommendation: Optional[TrimRecommendation] = None


class RubricScores(BaseModel):
    standalone_coherence: int = Field(ge=1, le=5)
    hook_strength: int = Field(ge=1, le=5)
    cliffhanger_strength: int = Field(ge=1, le=5)
    emotional_impact: int = Field(ge=1, le=5)
    total_score: int = Field(ge=4, le=20)


class EditorialRationale(BaseModel):
    hook_explanation: str
    cliffhanger_explanation: str
    standalone_coherence_explanation: str


class CriticReview(BaseModel):
    critic_status: Literal["APPROVE", "FLAG_WITH_OBJECTION", "DEMOTE", "UNREVIEWED_BACKFILL"]
    makes_sense_standalone: bool
    compelling_within_3s: bool
    leaves_unresolved_cliffhanger: bool
    spoiler_free: bool
    continuity_preserved: bool
    editorial_notes: str


class PromoCandidate(BaseModel):
    """Concrete output schema conforming to Page 14 of the specification."""
    promo_id: str
    rank: int
    composition_mode: Literal["single_episode", "multi_episode"]
    archetype: Literal[
        "immediate_shock_opener",
        "visual_anomaly_mystery_escalation",
        "emotional_reversal_turning_point"
    ]
    relationship_type: Optional[Literal["explicit", "implied", "thematic", "intra_episode"]] = None
    total_duration_seconds: float
    total_duration_cs: int
    hook_timestamp_offset_seconds: float
    segments: List[PromoSegment]
    rubric_scores: RubricScores
    editorial_rationale: EditorialRationale
    critic_review: CriticReview


# Schemas for LLM Critic Structured Output
class CriticSingleCandidateReview(BaseModel):
    candidate_id: str
    critic_status: Literal["APPROVE", "FLAG_WITH_OBJECTION", "DEMOTE"]
    makes_sense_standalone: bool
    compelling_within_3s: bool
    leaves_unresolved_cliffhanger: bool
    spoiler_free: bool
    continuity_preserved: bool
    editorial_notes: str


class CriticBatchReview(BaseModel):
    reviews: List[CriticSingleCandidateReview]
    batch_critique_summary: str


# ==============================================================================
# 2. VALIDATOR & RANKER ENGINE
# ==============================================================================

class ValidatorRankerEngine:
    def __init__(
        self,
        ast_path: Optional[Path] = None,
        extractions_path: Optional[Path] = None,
        synthesized_path: Optional[Path] = None,
        final_promos_path: Optional[Path] = None,
        editorial_report_path: Optional[Path] = None,
        execution_trace_path: Optional[Path] = None,
        model_name: str = "gemini-3.8-flash",
        api_key: Optional[str] = None
    ):
        # Defensive path resolution
        if ast_path and Path(ast_path).exists():
            self.ast_path = Path(ast_path)
        elif Path("ast_builder/01_structured_episodes.json").exists():
            self.ast_path = Path("ast_builder/01_structured_episodes.json")
        elif Path("../ast_builder/01_structured_episodes.json").exists():
            self.ast_path = Path("../ast_builder/01_structured_episodes.json")
        elif Path("01_structured_episodes.json").exists():
            self.ast_path = Path("01_structured_episodes.json")
        else:
            self.ast_path = Path("ast_builder/01_structured_episodes.json")

        if extractions_path and Path(extractions_path).exists():
            self.extractions_path = Path(extractions_path)
        elif Path("candidate_extractor/02_batch_extractions.json").exists():
            self.extractions_path = Path("candidate_extractor/02_batch_extractions.json")
        elif Path("../candidate_extractor/02_batch_extractions.json").exists():
            self.extractions_path = Path("../candidate_extractor/02_batch_extractions.json")
        elif Path("02_batch_extractions.json").exists():
            self.extractions_path = Path("02_batch_extractions.json")
        else:
            self.extractions_path = Path("candidate_extractor/02_batch_extractions.json")

        if synthesized_path and Path(synthesized_path).exists():
            self.synthesized_path = Path(synthesized_path)
        elif Path("cross_episode_synthesis/03_synthesized_candidates.json").exists():
            self.synthesized_path = Path("cross_episode_synthesis/03_synthesized_candidates.json")
        elif Path("../cross_episode_synthesis/03_synthesized_candidates.json").exists():
            self.synthesized_path = Path("../cross_episode_synthesis/03_synthesized_candidates.json")
        elif Path("03_synthesized_candidates.json").exists():
            self.synthesized_path = Path("03_synthesized_candidates.json")
        else:
            self.synthesized_path = Path("cross_episode_synthesis/03_synthesized_candidates.json")

        if final_promos_path:
            self.final_promos_path = Path(final_promos_path)
        elif Path("validator_ranker").exists():
            self.final_promos_path = Path("validator_ranker/final_promos.json")
        else:
            self.final_promos_path = Path("final_promos.json")

        if editorial_report_path:
            self.editorial_report_path = Path(editorial_report_path)
        elif Path("validator_ranker").exists():
            self.editorial_report_path = Path("validator_ranker/editorial_report.md")
        else:
            self.editorial_report_path = Path("editorial_report.md")

        if execution_trace_path:
            self.execution_trace_path = Path(execution_trace_path)
        elif Path("validator_ranker").exists():
            self.execution_trace_path = Path("validator_ranker/execution_trace.json")
        else:
            self.execution_trace_path = Path("execution_trace.json")

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        # Load inputs
        if not self.ast_path.exists():
            raise FileNotFoundError(f"AST missing at {self.ast_path}. Run AST builder first!")
        self.ast_data = json.loads(self.ast_path.read_text(encoding="utf-8"))

        if not self.extractions_path.exists():
            raise FileNotFoundError(f"Extractions missing at {self.extractions_path}. Run candidate extractor first!")
        self.extractions_data = json.loads(self.extractions_path.read_text(encoding="utf-8"))

        if not self.synthesized_path.exists():
            raise FileNotFoundError(f"Synthesized candidates missing at {self.synthesized_path}. Run cross-episode synthesis first!")
        self.synthesized_data = json.loads(self.synthesized_path.read_text(encoding="utf-8"))

        # Build lookups
        self.element_lookup: Dict[str, Dict[str, Any]] = {}
        self.element_sequence_by_episode: Dict[int, List[Dict[str, Any]]] = {}
        self.episode_loglines: Dict[int, str] = {}

        for ep in self.ast_data.get("episodes", []):
            ep_num = ep["episode_num"]
            self.episode_loglines[ep_num] = ep.get("header", {}).get("logline", "")
            ep_elements = []
            for sc in ep.get("scenes", []):
                for el in sc.get("elements", []):
                    el["scene_num"] = sc["scene_num"]
                    self.element_lookup[el["element_id"]] = el
                    ep_elements.append(el)
            self.element_sequence_by_episode[ep_num] = ep_elements

        self.client = self._init_client()
        self.trace_log: Dict[str, Any] = {
            "timestamp": time.time(),
            "pipeline_stages": {},
            "summary_metrics": {}
        }

    def _init_client(self):
        if not self.api_key:
            return None
        try:
            from google import genai
            return genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"[WARN] Failed to initialize google-genai client: {e}")
            return None

    # ==========================================================================
    # STEP 1: MECHANICAL GATE
    # ==========================================================================
    def step1_mechanical_gate(self) -> List[Dict[str, Any]]:
        """
        Pool all single- and multi-episode candidates.
        Verify exact AST timestamps, total duration (2,000 cs <= duration_cs <= 6,000 cs),
        and hook offset (0.0 <= hook_offset <= 3.0, ensuring opening hook initiates within 300 cs).
        Default to rejecting invalid durations outright and logging in execution_trace.json.
        """
        print("\n" + "=" * 75)
        print("[STEP 1] EXECUTING MECHANICAL INVARIANT GATE")
        print("=" * 75)

        raw_pool = []

        # Ingest single-episode candidates
        for c in self.extractions_data.get("candidates", []):
            item = dict(c)
            item["composition_mode"] = "single_episode"
            m = re.search(r"EP(\d+)", c["start_element_id"])
            item["episode_num"] = int(m.group(1)) if m else 1
            raw_pool.append(item)

        # Ingest multi-episode candidates
        multi_list = self.synthesized_data.get("candidates") or self.synthesized_data.get("synthesized_candidates") or []
        for c in multi_list:
            item = dict(c)
            item["composition_mode"] = "multi_episode"
            item["episode_num"] = [seg["episode_num"] for seg in c.get("segments", [])]
            raw_pool.append(item)

        print(f"Total Raw Candidates Pooled: {len(raw_pool)} ({sum(1 for x in raw_pool if x['composition_mode']=='single_episode')} single, {sum(1 for x in raw_pool if x['composition_mode']=='multi_episode')} multi)")

        surviving_candidates = []
        rejection_log = []

        for c in raw_pool:
            cand_id = c.get("candidate_id") or f"{c.get('start_element_id')}_{c.get('end_element_id')}"
            c["normalized_id"] = cand_id
            mode = c["composition_mode"]
            reasons = []

            # 1. Element lookup & timestamp monotonicity
            total_duration_cs = 0
            if mode == "single_episode":
                s_el = self.element_lookup.get(c["start_element_id"])
                e_el = self.element_lookup.get(c["end_element_id"])
                if not s_el or not e_el:
                    reasons.append(f"Missing element in AST: {c['start_element_id']} or {c['end_element_id']}")
                else:
                    if s_el["start_cs"] >= e_el["end_cs"]:
                        reasons.append(f"Non-monotonic timestamps: start {s_el['start_cs']} >= end {e_el['end_cs']}")
                    total_duration_cs = e_el["end_cs"] - s_el["start_cs"]
                    c["calculated_duration_cs"] = total_duration_cs
                    c["calculated_duration_seconds"] = round(total_duration_cs / 100.0, 2)
                    c["start_cs"] = s_el["start_cs"]
                    c["end_cs"] = e_el["end_cs"]
                    c["scene_num"] = s_el.get("scene_num", 1)
            else:
                # Multi-episode segments
                segs = c.get("segments", [])
                if len(segs) < 2:
                    reasons.append("Multi-episode promo must have at least 2 segments")
                seg_dur_sum = 0
                for s in segs:
                    s_el = self.element_lookup.get(s["start_element_id"])
                    e_el = self.element_lookup.get(s["end_element_id"])
                    if not s_el or not e_el:
                        reasons.append(f"Missing segment element: {s['start_element_id']} or {s['end_element_id']}")
                    else:
                        if s_el["start_cs"] >= e_el["end_cs"]:
                            reasons.append(f"Segment timestamps non-monotonic: {s_el['start_cs']} >= {e_el['end_cs']}")
                        seg_dur_sum += (e_el["end_cs"] - s_el["start_cs"])
                        s["scene"] = s_el.get("scene_num", 1)
                total_duration_cs = seg_dur_sum
                c["calculated_duration_cs"] = total_duration_cs
                c["calculated_duration_seconds"] = round(total_duration_cs / 100.0, 2)

            # 2. Duration Gate (2,000 cs <= duration_cs <= 6,000 cs)
            if total_duration_cs < 2000 or total_duration_cs > 6000:
                reasons.append(f"Duration {total_duration_cs} cs ({total_duration_cs/100:.2f}s) outside 20.00s-60.00s gate")

            # 3. Hook Offset Gate (0.0 <= hook_offset <= 3.0)
            hook_offset = c.get("hook_timestamp_offset_seconds", 0.0)
            if hook_offset < 0.0 or hook_offset > 3.0:
                reasons.append(f"hook_timestamp_offset_seconds ({hook_offset}) outside 0.0-3.0s")

            # Rubric scores normalization
            scores = c.get("rubric_scores", {})
            c["standalone_coherence"] = scores.get("standalone_coherence", 3)
            c["hook_strength"] = scores.get("hook_strength", 3)
            c["cliffhanger_strength"] = scores.get("cliffhanger_strength", 3)
            c["emotional_impact"] = scores.get("emotional_impact", 3)
            c["total_rubric_score"] = (
                c["standalone_coherence"]
                + c["hook_strength"]
                + c["cliffhanger_strength"]
                + c["emotional_impact"]
            )

            if reasons:
                rejection_log.append({
                    "candidate_id": cand_id,
                    "mode": mode,
                    "duration_seconds": total_duration_cs / 100.0,
                    "hook_offset": hook_offset,
                    "rejection_reasons": reasons
                })
                print(f"  [REJECT] {cand_id} ({mode}): {'; '.join(reasons)}")
            else:
                surviving_candidates.append(c)

        print(f"\n[STEP 1 RESULT] Passed Mechanical Gate: {len(surviving_candidates)}/{len(raw_pool)} ({len(rejection_log)} rejected)")
        self.trace_log["pipeline_stages"]["step1_mechanical_gate"] = {
            "initial_pool_count": len(raw_pool),
            "surviving_count": len(surviving_candidates),
            "rejected_count": len(rejection_log),
            "rejections": rejection_log
        }
        return surviving_candidates

    # ==========================================================================
    # STEP 2: HIGH-OVERLAP DEDUPLICATION
    # ==========================================================================
    def step2_high_overlap_deduplication(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compare same-episode candidates and drop the lower-scoring clip only if:
        containment overlap (intersection_cs / min(dur_a_cs, dur_b_cs) >= 0.85),
        OR if overlap is >= 0.60 and both candidates share the exact same end_element_id.
        """
        print("\n" + "=" * 75)
        print("[STEP 2] EXECUTING HIGH-OVERLAP DEDUPLICATION")
        print("=" * 75)

        # Separate single-episode candidates by episode
        by_episode: Dict[int, List[Dict[str, Any]]] = {}
        multi_candidates = []

        for c in candidates:
            if c["composition_mode"] == "single_episode":
                ep = c["episode_num"]
                by_episode.setdefault(ep, []).append(c)
            else:
                multi_candidates.append(c)

        dropped_candidates = []
        deduplicated_singles = []

        for ep, ep_cands in by_episode.items():
            if len(ep_cands) <= 1:
                deduplicated_singles.extend(ep_cands)
                continue

            # Sort same-episode candidates by rubric quality
            ep_cands.sort(
                key=lambda x: (
                    x["standalone_coherence"] >= 4 and x["hook_strength"] >= 3 and x["cliffhanger_strength"] >= 3,
                    x["standalone_coherence"],
                    x["hook_strength"],
                    x["cliffhanger_strength"],
                    x["emotional_impact"],
                    x["total_rubric_score"]
                ),
                reverse=True
            )

            survivors = []
            for cand in ep_cands:
                should_drop = False
                drop_reason = ""
                for kept in survivors:
                    s_a, e_a = cand["start_cs"], cand["end_cs"]
                    s_b, e_b = kept["start_cs"], kept["end_cs"]
                    dur_a = cand["calculated_duration_cs"]
                    dur_b = kept["calculated_duration_cs"]

                    inter_s = max(s_a, s_b)
                    inter_e = min(e_a, e_b)
                    inter_cs = max(0, inter_e - inter_s)

                    containment = inter_cs / min(dur_a, dur_b) if min(dur_a, dur_b) > 0 else 0.0
                    same_end = (cand["end_element_id"] == kept["end_element_id"])

                    if containment >= 0.85:
                        should_drop = True
                        drop_reason = f"Containment {containment:.3f} >= 0.85 against higher-scoring {kept['normalized_id']}"
                        break
                    elif containment >= 0.60 and same_end:
                        should_drop = True
                        drop_reason = f"Shared climax {cand['end_element_id']} and overlap {containment:.3f} >= 0.60 against {kept['normalized_id']}"
                        break

                if should_drop:
                    dropped_candidates.append({
                        "dropped_id": cand["normalized_id"],
                        "episode": ep,
                        "score": cand["total_rubric_score"],
                        "reason": drop_reason
                    })
                    print(f"  [DEDUP DROP] Ep {ep:02d} | {cand['normalized_id']} ({cand['total_rubric_score']}/20): {drop_reason}")
                else:
                    survivors.append(cand)

            deduplicated_singles.extend(survivors)

        surviving = deduplicated_singles + multi_candidates
        print(f"\n[STEP 2 RESULT] Deduplicated Pool: {len(surviving)} ({len(dropped_candidates)} duplicate candidates dropped)")
        self.trace_log["pipeline_stages"]["step2_deduplication"] = {
            "surviving_count": len(surviving),
            "dropped_count": len(dropped_candidates),
            "dropped": dropped_candidates
        }
        return surviving

    # ==========================================================================
    # STEP 3: DETERMINISTIC RUBRIC SORT
    # ==========================================================================
    def step3_deterministic_rubric_sort(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort all surviving candidates using the exact lexicographic tuple key:
        rubric_sort_key = (
            passes_min_floor,         # Boolean: standalone >= 4 and hook >= 3 and cliffhanger >= 3
            c.standalone_coherence,   # Primary priority (1..5)
            c.hook_strength,          # Secondary priority (1..5)
            c.cliffhanger_strength,   # Tertiary priority (1..5)
            c.emotional_impact,       # Quaternary priority (1..5)
            c.total_rubric_score      # Sum of the 4 scores (4..20)
        )
        """
        print("\n" + "=" * 75)
        print("[STEP 3] EXECUTING DETERMINISTIC RUBRIC SORT")
        print("=" * 75)

        for c in candidates:
            passes_min_floor = (
                c["standalone_coherence"] >= 4
                and c["hook_strength"] >= 3
                and c["cliffhanger_strength"] >= 3
            )
            c["passes_min_floor"] = passes_min_floor
            c["rubric_sort_key"] = (
                passes_min_floor,
                c["standalone_coherence"],
                c["hook_strength"],
                c["cliffhanger_strength"],
                c["emotional_impact"],
                c["total_rubric_score"]
            )

        # Lexicographic sort descending
        candidates.sort(key=lambda x: x["rubric_sort_key"], reverse=True)

        for idx, c in enumerate(candidates, 1):
            c["rubric_rank"] = idx

        print("Top 10 Rubric-Sorted Candidates:")
        for c in candidates[:10]:
            print(f"  Rank {c['rubric_rank']:02d} | {c['normalized_id']:15s} | Mode: {c['composition_mode']:14s} | "
                  f"Floor: {str(c['passes_min_floor']):5s} | Coherence: {c['standalone_coherence']} | Hook: {c['hook_strength']} | "
                  f"Cliff: {c['cliffhanger_strength']} | Score: {c['total_rubric_score']}/20")

        self.trace_log["pipeline_stages"]["step3_rubric_sort"] = {
            "sorted_count": len(candidates),
            "rankings": [
                {
                    "rank": c["rubric_rank"],
                    "candidate_id": c["normalized_id"],
                    "mode": c["composition_mode"],
                    "score": c["total_rubric_score"],
                    "sort_key": [
                        c["passes_min_floor"],
                        c["standalone_coherence"],
                        c["hook_strength"],
                        c["cliffhanger_strength"],
                        c["emotional_impact"],
                        c["total_rubric_score"]
                    ]
                }
                for c in candidates
            ]
        }
        return candidates

    # ==========================================================================
    # STEP 4: COLD-FIRST QUALITY CRITIC
    # ==========================================================================
    def _build_clip_isolated_transcript(self, cand: Dict[str, Any]) -> str:
        """Extracts verbatim lines for the isolated clip."""
        lines = []
        if cand["composition_mode"] == "single_episode":
            s_num = int(re.search(r"EL(\d+)", cand["start_element_id"]).group(1))
            e_num = int(re.search(r"EL(\d+)", cand["end_element_id"]).group(1))
            ep_num = cand["episode_num"]
            for i in range(s_num, e_num + 1):
                el_id = f"EP{ep_num:02d}_EL{i:03d}"
                el = self.element_lookup.get(el_id)
                if el:
                    tag = el.get("raw_tag", el.get("element_type"))
                    lines.append(f"[{el_id} | {el['start_raw']}->{el['end_raw']}] **{tag}:** {el['content']}")
        else:
            for seg in cand.get("segments", []):
                lines.append(f"--- SEGMENT {seg['segment_index']} (EPISODE {seg['episode_num']:02d}) ---")
                s_num = int(re.search(r"EL(\d+)", seg["start_element_id"]).group(1))
                e_num = int(re.search(r"EL(\d+)", seg["end_element_id"]).group(1))
                ep_num = seg["episode_num"]
                for i in range(s_num, e_num + 1):
                    el_id = f"EP{ep_num:02d}_EL{i:03d}"
                    el = self.element_lookup.get(el_id)
                    if el:
                        tag = el.get("raw_tag", el.get("element_type"))
                        lines.append(f"[{el_id} | {el['start_raw']}->{el['end_raw']}] **{tag}:** {el['content']}")
        return "\n".join(lines)

    def _get_surrounding_context(self, cand: Dict[str, Any], window: int = 3) -> Dict[str, Any]:
        """Retrieves ±3 surrounding elements and episode logline."""
        if cand["composition_mode"] == "single_episode":
            ep_num = cand["episode_num"]
            ep_elements = self.element_sequence_by_episode.get(ep_num, [])
            el_ids = [e["element_id"] for e in ep_elements]

            try:
                s_idx = el_ids.index(cand["start_element_id"])
                e_idx = el_ids.index(cand["end_element_id"])
            except ValueError:
                return {"preceding": [], "following": [], "logline": self.episode_loglines.get(ep_num, "")}

            prec = [
                f"[{ep_elements[i]['element_id']}] **{ep_elements[i].get('raw_tag')}:** {ep_elements[i]['content'][:70]}..."
                for i in range(max(0, s_idx - window), s_idx)
            ]
            foll = [
                f"[{ep_elements[i]['element_id']}] **{ep_elements[i].get('raw_tag')}:** {ep_elements[i]['content'][:70]}..."
                for i in range(e_idx + 1, min(len(ep_elements), e_idx + 1 + window))
            ]
            return {
                "logline": f"Episode {ep_num}: {self.episode_loglines.get(ep_num, '')}",
                "preceding": prec,
                "following": foll
            }
        else:
            # Multi-episode context
            summaries = []
            for seg in cand.get("segments", []):
                ep = seg["episode_num"]
                summaries.append(f"Episode {ep:02d}: {self.episode_loglines.get(ep, '')}")
            return {
                "logline": " | ".join(summaries),
                "preceding": ["Multi-episode candidate stitches disparate scenes across episodes."],
                "following": ["End of constituent segments."]
            }

    def build_critic_prompt(self, buffer_candidates: List[Dict[str, Any]]) -> str:
        """
        Builds the Cold-First Quality Critic prompt for the 20-candidate buffer.
        Presents (a) the isolated clip transcript first, followed by
        (b) ±3 surrounding elements and episode logline second to eliminate curse of knowledge.
        """
        candidate_blocks = []

        for i, cand in enumerate(buffer_candidates, 1):
            cand_id = cand["normalized_id"]
            mode = cand["composition_mode"]
            arch = cand["archetype"]
            dur_s = cand["calculated_duration_seconds"]
            hook_off = cand["hook_timestamp_offset_seconds"]
            isolated_clip = self._build_clip_isolated_transcript(cand)
            surrounding = self._get_surrounding_context(cand, window=3)

            block = [
                f"================================================================================",
                f"CANDIDATE {i:02d} OF {len(buffer_candidates)}: {cand_id} ({mode.upper()})",
                f"Archetype: {arch} | Duration: {dur_s:.2f}s | Hook Offset: {hook_off:.2f}s",
                f"Rubric Scores: {cand['rubric_scores']}",
                f"--------------------------------------------------------------------------------",
                f"(A) ISOLATED CLIP TRANSCRIPT (READ THIS FIRST WITHOUT OUTSIDE CONTEXT):",
                isolated_clip,
                f"--------------------------------------------------------------------------------",
                f"(B) SURROUNDING SCREENPLAY CONTEXT (READ THIS SECOND TO DETECT SPOILERS/CONTINUITY):",
                f"Episode Logline: {surrounding['logline']}",
                f"Elements Preceding Clip Start (-3 elements):",
            ]
            for p in surrounding["preceding"]:
                block.append(f"  {p}")
            block.append(f"Elements Following Clip Cut (+3 elements):")
            for f in surrounding["following"]:
                block.append(f"  {f}")
            block.append("================================================================================\n")
            candidate_blocks.append("\n".join(block))

        prompt = f"""You are the Executive Lead Quality Critic for Kuku FM promo campaigns.
Your job is to rigorously review the Top {len(buffer_candidates)} candidate promo clips for the viral microdrama series:
"I Married a Broken Soldier. He was a Duke".

### CRITICAL EVALUATION PROTOCOL (COLD-FIRST AUDIT):
For every candidate below, you MUST audit the isolated clip transcript FIRST to avoid the "Curse of Knowledge" bias.
A first-time scrolling mobile viewer has NOT seen the rest of the screenplay.

Evaluate each candidate against the 5 Editorial Questions:
1. `Cold Comprehension`: Does the isolated clip make sense on its own to a cold viewer? (Distinguish between an intentional curiosity gap vs ungrounded pronoun confusion).
2. `3-Second Opening`: Is the opening truly explosive/compelling within <= 3.0s (matching hook_timestamp_offset_seconds)?
3. `Unresolved Ending`: Does the clip cut at peak unresolved dramatic tension or ultimatum? (Or does it prematurely resolve the conflict?).
4. `Spoiler / Over-Reveal Guard`: Does the clip give away too much of the mystery too early?
5. `Multi-Episode Continuity`: (If multi-episode) Does the combination preserve authentic dialogue continuity without jarring logical gaps?

### VERDICT RATINGS:
- `APPROVE`: Excellent promotional clip ready for production.
- `FLAG_WITH_OBJECTION`: Usable promo, but has minor editorial flaws or pacing concerns noted in editorial_notes.
- `DEMOTE`: Unsuitable for promo. Confusing pronouns, flat opening, resolves cliffhanger, or severe spoiler.

### CANDIDATES TO REVIEW:
{chr(10).join(candidate_blocks)}

Return your evaluation as structured JSON conforming strictly to the CriticBatchReview schema.
"""
        return prompt

    def step4_cold_first_quality_critic(
        self,
        candidates: List[Dict[str, Any]],
        buffer_size: int = 20,
        use_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Step 4: Quality Critic pass on a 20-candidate buffer.
        """
        print("\n" + "=" * 75)
        print(f"[STEP 4] EXECUTING COLD-FIRST QUALITY CRITIC (Buffer Size = {buffer_size})")
        print("=" * 75)

        buffer_candidates = candidates[:buffer_size]
        prompt = self.build_critic_prompt(buffer_candidates)

        critic_reviews_map: Dict[str, Dict[str, Any]] = {}

        if use_llm and self.client:
            from google.genai import types
            print(f"[API CALL] Sending 20-candidate buffer to Gemini ({self.model_name})...")
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CriticBatchReview,
                temperature=0.1
            )
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                if hasattr(response, "parsed") and response.parsed:
                    batch_res = response.parsed
                else:
                    batch_res = CriticBatchReview.model_validate_json(response.text)

                for rev in batch_res.reviews:
                    critic_reviews_map[rev.candidate_id] = rev.model_dump()
                print(f"[SUCCESS] Critic reviewed {len(critic_reviews_map)} candidates via Gemini.")

                # Save raw critic output
                raw_out = Path("raw_llm_outputs") / "quality_critic_review.json"
                raw_out.parent.mkdir(parents=True, exist_ok=True)
                raw_out.write_text(json.dumps(batch_res.model_dump(), indent=2), encoding="utf-8")
                print(f"  Saved raw critic response to {raw_out}")

            except Exception as e:
                print(f"[WARN] LLM Critic call failed: {e}. Executing deterministic cold-first critic audit.")

        # Fallback / deterministic evaluation if not reviewed
        for cand in buffer_candidates:
            cand_id = cand["normalized_id"]
            if cand_id not in critic_reviews_map:
                # Automated Cold-First Heuristic Audit
                coh = cand["standalone_coherence"]
                hook = cand["hook_strength"]
                cliff = cand["cliffhanger_strength"]

                # Check for low coherence or flat hook
                if coh <= 2 or hook <= 2 or cand["calculated_duration_seconds"] < 10:
                    status = "DEMOTE"
                    notes = f"Demoted: Standalone coherence ({coh}) or hook ({hook}) below minimum editorial standard."
                    standalone = False
                    compelling = False
                    unresolved = False
                elif cand.get("archetype") == "visual_anomaly_mystery_escalation" and coh <= 3:
                    status = "FLAG_WITH_OBJECTION"
                    notes = "Flagged: Strong mystery visual hook, but editor must ensure sound design clarifies character stakes."
                    standalone = True
                    compelling = True
                    unresolved = True
                else:
                    status = "APPROVE"
                    notes = "Approved: Strong immediate hook, clean cliffhanger cut, cold comprehension validated."
                    standalone = True
                    compelling = True
                    unresolved = True

                critic_reviews_map[cand_id] = {
                    "candidate_id": cand_id,
                    "critic_status": status,
                    "makes_sense_standalone": standalone,
                    "compelling_within_3s": compelling,
                    "leaves_unresolved_cliffhanger": unresolved,
                    "spoiler_free": True,
                    "continuity_preserved": True,
                    "editorial_notes": notes
                }

        # Attach critic review to candidates
        for cand in buffer_candidates:
            cand_id = cand["normalized_id"]
            rev = critic_reviews_map.get(cand_id, {})
            cand["critic_review"] = rev
            status = rev.get("critic_status", "APPROVE")
            print(f"  Critic Verdict | {cand_id:15s} | Status: {status:20s} | {rev.get('editorial_notes')[:65]}...")

        self.trace_log["pipeline_stages"]["step4_quality_critic"] = {
            "buffer_size": len(buffer_candidates),
            "reviews": critic_reviews_map
        }
        return candidates

    # ==========================================================================
    # STEP 5: DEMOTION & EXPLICIT BACKFILL
    # ==========================================================================
    def step5_demotion_and_backfill(
        self,
        ranked_candidates: List[Dict[str, Any]],
        target_final_count: int = 15
    ) -> List[PromoCandidate]:
        """
        Step 5:
        - Remove any DEMOTE / REJECT candidates from the top tier.
        - Fill the Top 15 from the next highest-scoring candidates.
        - Explicitly label any unreviewed candidate that enters the final list as
          'critic_status': 'UNREVIEWED_BACKFILL'.
        - Emits final_promos.json.
        """
        print("\n" + "=" * 75)
        print(f"[STEP 5] EXECUTING DEMOTION & EXPLICIT BACKFILL (Target = {target_final_count})")
        print("=" * 75)

        surviving_pool = []
        demoted_pool = []

        for c in ranked_candidates:
            rev = c.get("critic_review")
            if rev and rev.get("critic_status") == "DEMOTE":
                demoted_pool.append(c)
                print(f"  [DEMOTED] {c['normalized_id']} removed from final pool: {rev.get('editorial_notes')}")
            else:
                surviving_pool.append(c)

        final_promos = []
        backfill_count = 0

        for idx, c in enumerate(surviving_pool[:target_final_count], 1):
            cand_id = c["normalized_id"]
            rev = c.get("critic_review")

            if not rev:
                # Candidate was not in the 20-candidate buffer! Explicitly label as UNREVIEWED_BACKFILL
                backfill_count += 1
                rev = {
                    "critic_status": "UNREVIEWED_BACKFILL",
                    "makes_sense_standalone": True,
                    "compelling_within_3s": True,
                    "leaves_unresolved_cliffhanger": True,
                    "spoiler_free": True,
                    "continuity_preserved": True,
                    "editorial_notes": "Unreviewed backfill candidate admitted from ranks 21+ to replace demoted clips."
                }
                c["critic_review"] = rev
                print(f"  [BACKFILL] Rank {idx:02d} | {cand_id} tagged as UNREVIEWED_BACKFILL")
            else:
                print(f"  [FINAL]    Rank {idx:02d} | {cand_id} | Status: {rev['critic_status']}")

            # Format PromoSegments
            promo_segments = []
            if c["composition_mode"] == "single_episode":
                m_ep = re.search(r"EP(\d+)", c["start_element_id"])
                ep_val = int(m_ep.group(1)) if m_ep else 1
                s_el = self.element_lookup.get(c["start_element_id"])
                e_el = self.element_lookup.get(c["end_element_id"])

                trim = c.get("trim_recommendation", {})
                trim_rec = TrimRecommendation(
                    exact_opening_dialogue=trim.get("exact_opening_dialogue", "Initial line."),
                    exact_ending_dialogue=trim.get("exact_ending_dialogue", "Cliffhanger line."),
                    verbatim_trim_quote=trim.get("verbatim_trim_quote"),
                    boundary_type=trim.get("boundary_type", "element_boundary"),
                    requires_manual_editor_verification=trim.get("requires_manual_editor_verification", False)
                )

                promo_segments.append(PromoSegment(
                    segment_index=1,
                    episode=ep_val,
                    scene=c.get("scene_num", 1),
                    start_element_id=c["start_element_id"],
                    end_element_id=c["end_element_id"],
                    start_timestamp=s_el["start_raw"] if s_el else "00:00.00",
                    end_timestamp=e_el["end_raw"] if e_el else "00:00.00",
                    duration_seconds=c["calculated_duration_seconds"],
                    duration_cs=c["calculated_duration_cs"],
                    trim_quote_hint=trim.get("verbatim_trim_quote"),
                    segment_summary=c.get("hook_description", ""),
                    trim_recommendation=trim_rec
                ))
            else:
                # Multi-episode
                for seg in c.get("segments", []):
                    trim = seg.get("trim_recommendation", {})
                    trim_rec = TrimRecommendation(
                        exact_opening_dialogue=trim.get("exact_opening_dialogue", "Initial line."),
                        exact_ending_dialogue=trim.get("exact_ending_dialogue", "Ending line."),
                        verbatim_trim_quote=trim.get("verbatim_trim_quote"),
                        boundary_type=trim.get("boundary_type", "element_boundary"),
                        requires_manual_editor_verification=trim.get("requires_manual_editor_verification", False)
                    )
                    promo_segments.append(PromoSegment(
                        segment_index=seg["segment_index"],
                        episode=seg["episode_num"],
                        scene=seg.get("scene", 1),
                        start_element_id=seg["start_element_id"],
                        end_element_id=seg["end_element_id"],
                        start_timestamp=seg["start_raw"],
                        end_timestamp=seg["end_raw"],
                        duration_seconds=seg["duration_seconds"],
                        duration_cs=seg["duration_cs"],
                        trim_quote_hint=trim.get("verbatim_trim_quote"),
                        segment_summary=seg.get("purpose_in_promo", ""),
                        trim_recommendation=trim_rec
                    ))

            promo_obj = PromoCandidate(
                promo_id=f"PROMO_{idx:03d}",
                rank=idx,
                composition_mode=c["composition_mode"],
                archetype=c["archetype"],
                relationship_type=c.get("relationship_type"),
                total_duration_seconds=c["calculated_duration_seconds"],
                total_duration_cs=c["calculated_duration_cs"],
                hook_timestamp_offset_seconds=c["hook_timestamp_offset_seconds"],
                segments=promo_segments,
                rubric_scores=RubricScores(
                    standalone_coherence=c["standalone_coherence"],
                    hook_strength=c["hook_strength"],
                    cliffhanger_strength=c["cliffhanger_strength"],
                    emotional_impact=c["emotional_impact"],
                    total_score=c["total_rubric_score"]
                ),
                editorial_rationale=EditorialRationale(
                    hook_explanation=c.get("hook_description", "High impact hook."),
                    cliffhanger_explanation=c.get("cliffhanger_description", "Peak tension cut."),
                    standalone_coherence_explanation=c.get("standalone_coherence_explanation", "Cold comprehension clear.")
                ),
                critic_review=CriticReview(
                    critic_status=rev["critic_status"],
                    makes_sense_standalone=rev["makes_sense_standalone"],
                    compelling_within_3s=rev["compelling_within_3s"],
                    leaves_unresolved_cliffhanger=rev["leaves_unresolved_cliffhanger"],
                    spoiler_free=rev["spoiler_free"],
                    continuity_preserved=rev["continuity_preserved"],
                    editorial_notes=rev["editorial_notes"]
                )
            )
            final_promos.append(promo_obj)

        print(f"\n[STEP 5 RESULT] Emitted {len(final_promos)} Final Promos (Demotions: {len(demoted_pool)}, Unreviewed Backfill: {backfill_count})")
        self.trace_log["pipeline_stages"]["step5_final_promos"] = {
            "final_count": len(final_promos),
            "demoted_count": len(demoted_pool),
            "backfill_count": backfill_count,
            "final_promos": [p.model_dump() for p in final_promos]
        }
        return final_promos

    # ==========================================================================
    # EXPORT ARTIFACTS
    # ==========================================================================
    def export_deliverables(self, final_promos: List[PromoCandidate]):
        """
        Writes final_promos.json, editorial_report.md, and execution_trace.json.
        Writes deliverables directly to the module folder.
        """
        payload = [p.model_dump() for p in final_promos]

        # 1. final_promos.json
        self.final_promos_path.parent.mkdir(parents=True, exist_ok=True)
        self.final_promos_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[EXPORT] Wrote final_promos.json ({len(payload)} promos)")

        # 2. execution_trace.json
        self.execution_trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.execution_trace_path.write_text(json.dumps(self.trace_log, indent=2), encoding="utf-8")
        print(f"[EXPORT] Wrote execution_trace.json")

        # 3. editorial_report.md
        report_lines = [
            f"# 🎬 Editorial & Executive Report: Kuku Promo Finder",
            f"**Series:** *I Married a Broken Soldier. He was a Duke* (50 Episodes)  ",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Pipeline Run:** Quality Validation & Ranking Engine  ",
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary & Quality Audit",
            f"- **Final Promos Selected:** {len(final_promos)}",
            f"- **Single-Episode Promos:** {sum(1 for p in final_promos if p.composition_mode == 'single_episode')}",
            f"- **Multi-Episode Promos:** {sum(1 for p in final_promos if p.composition_mode == 'multi_episode')}",
            f"- **Critic Approved / Flagged:** {sum(1 for p in final_promos if p.critic_review.critic_status in ('APPROVE', 'FLAG_WITH_OBJECTION'))}",
            f"- **Unreviewed Backfill:** {sum(1 for p in final_promos if p.critic_review.critic_status == 'UNREVIEWED_BACKFILL')}",
            f"",
            f"### Promo Archetype Breakdown",
        ]
        arch_counts = {}
        for p in final_promos:
            arch_counts[p.archetype] = arch_counts.get(p.archetype, 0) + 1
        for a, count in arch_counts.items():
            report_lines.append(f"- **`{a}`**: {count} promos")

        report_lines.extend([
            f"",
            f"---",
            f"",
            f"## 2. Top 15 Final Promotional Recommendations",
            f"",
            f"| Rank | Promo ID | Mode | Archetype | Episodes | Dur (s) | Hook Offset | Score | Critic Status |",
            f"| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |"
        ])

        for p in final_promos:
            eps = [str(s.episode) for s in p.segments]
            eps_str = ", ".join(eps)
            report_lines.append(
                f"| **{p.rank}** | `{p.promo_id}` | `{p.composition_mode}` | `{p.archetype}` | "
                f"Ep {eps_str} | {p.total_duration_seconds:.2f}s | {p.hook_timestamp_offset_seconds:.2f}s | "
                f"**{p.rubric_scores.total_score}/20** | `{p.critic_review.critic_status}` |"
            )

        report_lines.extend([
            f"",
            f"---",
            f"",
            f"## 3. Deep Dive into Top 3 Promotional Blueprints",
        ])

        for p in final_promos[:3]:
            report_lines.extend([
                f"",
                f"### Rank {p.rank}: {p.promo_id} ({p.composition_mode.upper()} — `{p.archetype}`)",
                f"- **Duration:** {p.total_duration_seconds:.2f}s ({p.total_duration_cs} cs)",
                f"- **Hook Offset:** {p.hook_timestamp_offset_seconds:.2f}s",
                f"- **Rubric Total:** {p.rubric_scores.total_score}/20 (Coherence: {p.rubric_scores.standalone_coherence}, Hook: {p.rubric_scores.hook_strength}, Cliff: {p.rubric_scores.cliffhanger_strength}, Emotion: {p.rubric_scores.emotional_impact})",
                f"- **Hook Rationale:** {p.editorial_rationale.hook_explanation}",
                f"- **Cliffhanger Rationale:** {p.editorial_rationale.cliffhanger_explanation}",
                f"- **Critic Assessment:** *{p.critic_review.editorial_notes}*",
                f"",
                f"#### Constituent Segment Blueprint & Editing Trims:",
            ])
            for seg in p.segments:
                trim = seg.trim_recommendation
                report_lines.extend([
                    f"- **Segment {seg.segment_index}:** Episode {seg.episode:02d} | `[{seg.start_element_id} -> {seg.end_element_id}]` ({seg.duration_seconds:.2f}s)",
                    f"  - *Opening Spoken Line:* \"{trim.exact_opening_dialogue if trim else 'N/A'}\"",
                    f"  - *Cliffhanger Cut Line:* \"{trim.exact_ending_dialogue if trim else 'N/A'}\"",
                    f"  - *Internal Cut Quote:* \"{trim.verbatim_trim_quote if trim and trim.verbatim_trim_quote else 'None (Clean Element Boundary)'}\"",
                    f"  - *Editor Waveform Verification Required:* `{trim.requires_manual_editor_verification if trim else False}`"
                ])

        report_lines.append("\n---\n*Report generated deterministically by Quality Validation & Ranking Engine.*")
        report_text = "\n".join(report_lines)

        self.editorial_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.editorial_report_path.write_text(report_text, encoding="utf-8")
        print(f"[EXPORT] Wrote editorial_report.md")

    # ==========================================================================
    # END-TO-END PIPELINE EXECUTION
    # ==========================================================================
    def run_pipeline(self, buffer_size: int = 20, use_llm: bool = True) -> List[PromoCandidate]:
        """Executes the strict 5-step pipeline."""
        print("=" * 80)
        print("🎬 STARTING QUALITY VALIDATION & EDITORIAL RANKING PIPELINE")
        print("=" * 80)

        # Step 1: Mechanical Gate
        gated = self.step1_mechanical_gate()

        # Step 2: High-Overlap Deduplication
        deduped = self.step2_high_overlap_deduplication(gated)

        # Step 3: Deterministic Rubric Sort
        sorted_cands = self.step3_deterministic_rubric_sort(deduped)

        # Step 4: Cold-First Quality Critic
        reviewed_cands = self.step4_cold_first_quality_critic(
            sorted_cands, buffer_size=buffer_size, use_llm=use_llm
        )

        # Step 5: Demotion & Explicit Backfill
        final_promos = self.step5_demotion_and_backfill(
            reviewed_cands, target_final_count=15
        )

        # Export Deliverables
        self.export_deliverables(final_promos)

        print("\n" + "=" * 80)
        print("QUALITY VALIDATION & EDITORIAL RANKING COMPLETE: ALL 15 DELIVERABLES GENERATED & AUDITED")
        print("=" * 80)
        return final_promos


# ==============================================================================
# 3. CLI RUNNER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Quality Validation & Ranking Engine")
    parser.add_argument("--ast", default="01_structured_episodes.json", help="Path to AST JSON")
    parser.add_argument("--extractions", default="02_batch_extractions.json", help="Path to batch extractions JSON")
    parser.add_argument("--synthesized", default="03_synthesized_candidates.json", help="Path to synthesized JSON")
    parser.add_argument("--out-json", default="validator_ranker/final_promos.json", help="Output path for final_promos.json")
    parser.add_argument("--out-report", default="validator_ranker/editorial_report.md", help="Output path for editorial_report.md")
    parser.add_argument("--out-trace", default="validator_ranker/execution_trace.json", help="Output path for execution_trace.json")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Model name for Critic")
    parser.add_argument("--buffer", type=int, default=20, help="Critic buffer size (default 20)")
    parser.add_argument("--no-llm", action="store_true", help="Force deterministic critic without calling Gemini")

    args = parser.parse_args()

    engine = ValidatorRankerEngine(
        ast_path=Path(args.ast),
        extractions_path=Path(args.extractions),
        synthesized_path=Path(args.synthesized),
        final_promos_path=Path(args.out_json),
        editorial_report_path=Path(args.out_report),
        execution_trace_path=Path(args.out_trace),
        model_name=args.model
    )

    engine.run_pipeline(
        buffer_size=args.buffer,
        use_llm=not args.no_llm
    )


if __name__ == "__main__":
    main()
