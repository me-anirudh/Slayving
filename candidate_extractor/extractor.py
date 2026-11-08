"""
Batch Candidate & Narrative Memory Extractor
Extracts high-converting single-episode promo candidates and cross-episode narrative
moments across screenplay episodes using Google Gemini with structured Pydantic schemas,
atomic disk checkpointing, exponential backoff retries, and AST duration enrichment.
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any, Tuple
from pydantic import BaseModel, Field


class TrimRecommendation(BaseModel):
    """
    Precise trim recommendations for video editors.
    Provides actionable cut instructions without unreliable character-based timestamp interpolation.
    """
    exact_opening_dialogue: str = Field(
        description="Verbatim dialogue line that begins the clip (or initial action description if visual opener)"
    )
    exact_ending_dialogue: str = Field(
        description="Verbatim dialogue line that ends the clip right before the final cliffhanger cut"
    )
    verbatim_trim_quote: Optional[str] = Field(
        default=None,
        description="Verbatim quote within dialogue where trim should occur if sentence-level cutting is required"
    )
    boundary_type: Literal["element_boundary", "internal_cut"] = Field(
        description="'element_boundary' if cut aligns with start/end element, 'internal_cut' if internal sentence trim"
    )
    requires_manual_editor_verification: bool = Field(
        description="True if audio/waveform check is advised for clean speech boundary, False if natural clean pause"
    )


class RubricScores(BaseModel):
    hook_strength: int = Field(ge=1, le=5, description="Immediate conflict or anomaly within <=3.0s (1-5)")
    cold_open_clarity: int = Field(ge=1, le=5, description="Self-contained comprehensibility for cold viewers (1-5)")
    emotional_amplitude: int = Field(ge=1, le=5, description="Intensity of emotional escalation or stakes (1-5)")
    cliffhanger_potency: int = Field(ge=1, le=5, description="Sudden high-stakes unresolved cutoff (1-5)")
    production_feasibility: int = Field(ge=1, le=5, description="Audio clarity, dialogue-dominance, cut cleanability (1-5)")


class PromoCandidate(BaseModel):
    start_element_id: str = Field(description="Starting element ID from AST, e.g., 'EP01_EL012'")
    end_element_id: str = Field(description="Ending element ID from AST, e.g., 'EP01_EL035'")
    archetype: Literal[
        "immediate_shock_opener",
        "visual_anomaly_mystery_escalation",
        "emotional_reversal_turning_point"
    ] = Field(description="Standardized promo archetype")
    hook_timestamp_offset_seconds: float = Field(
        ge=0.0, le=3.0,
        description="Exact timestamp offset from candidate start where the thumb-stopping hook occurs (must be <= 3.0s)"
    )
    hook_summary: str = Field(description="1-2 sentence description of the 3-second hook beat")
    cliffhanger_summary: str = Field(description="1-2 sentence description of the unresolved cliffhanger cut")
    rubric_scores: RubricScores
    trim_recommendation: TrimRecommendation
    editorial_rationale: str = Field(description="Editorial argument explaining why this candidate succeeds as a high-converting ad")


class NarrativeMoment(BaseModel):
    moment_id: str = Field(description="Unique narrative moment ID, e.g., 'MOMENT_EP01_01'")
    moment_type: Literal[
        "catalyst_or_shock",
        "core_conflict",
        "power_inversion",
        "revelation_or_twist",
        "cliffhanger_or_suspense"
    ] = Field(description="Narrative beat archetype")
    summary: str = Field(description="High-density objective summary of the narrative beat")
    characters_involved: List[str] = Field(description="Character names involved in this moment")
    start_element_id: str = Field(description="Starting element ID in AST")
    end_element_id: str = Field(description="Ending element ID in AST")
    continuity_clues: List[str] = Field(default_factory=list, description="Props, aliases, secrets, or vows")


class NarrativeContext(BaseModel):
    character_aliases: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Aliases/titles mapped to canonical names, e.g., {'alias': 'The Broken Soldier', 'canonical': 'Alexander'}"
    )
    objects_and_clues: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Key narrative props/symbols, e.g., {'item': 'Royal Signet Ring', 'significance': 'Proves Duke identity'}"
    )
    relationships: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Inter-character relationship dynamics tracked across episodes"
    )
    episode_summaries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Dense 2-3 sentence summaries per episode in the batch"
    )


class BatchExtractionResponse(BaseModel):
    batch_index: int = Field(description="Index of current batch (1 to 10)")
    episodes_included: List[int] = Field(description="List of episode numbers in this batch")
    candidates: List[PromoCandidate] = Field(description="Extracted single-episode promo candidates (25-55s target duration)")
    narrative_moments: List[NarrativeMoment] = Field(description="Key narrative beats for cross-episode synthesis")
    narrative_context: NarrativeContext = Field(description="Cumulative narrative context and character/prop mapping")


EXTRACTION_SYSTEM_PROMPT = """You are an expert microdrama ad director and promo producer for Kuku FM.
Your task is to analyze batches of microdrama episodes and extract:
1. High-Converting Single-Episode Promo Candidates (target 25.00s–55.00s, hard bounds 20.00s–60.00s)
2. Precise Trim Recommendations for Video Editors (verbatim opening/closing spoken lines)
3. Key Cross-Episode Narrative Moments (story beats for multi-episode trailer synthesis)
4. Flat Narrative Context (aliases, props, character dynamics, episode summaries)

PROMO ARCHETYPES:
- immediate_shock_opener: Visceral physical action, slap, confrontation, or betrayal accusation within <= 3.00s.
- visual_anomaly_mystery_escalation: Suspicious visual cue in CAMERA or ACTION block within <= 3.00s escalating to a dramatic twist.
- emotional_reversal_turning_point: Status humiliation, vow, or power inversion within <= 3.00s building to high-stakes emotional climax.

TIMING & TRIM RIGOR:
- Hard duration constraint: combined clip duration must be strictly 20.00s to 60.00s (targeting 25.00s–55.00s).
- Hook timing: hook_timestamp_offset_seconds must occur within 0.0s to 3.0s from start_element_id.
- Exact dialogue lines: provide verbatim opening dialogue and ending dialogue in TrimRecommendation.
"""


def format_batch_transcript(episodes: List[Dict[str, Any]]) -> str:
    """Formats screenplay elements with duration stamps and element IDs."""
    lines = []
    for ep in episodes:
        ep_num = ep["episode_num"]
        header = ep.get("header", {})
        logline = header.get("logline", "")
        lines.append(f"=== EPISODE {ep_num:02d} ===")
        if logline:
            lines.append(f"Logline: {logline}")

        for scene in ep.get("scenes", []):
            sc_num = scene["scene_num"]
            heading = scene.get("heading", "")
            lines.append(f"\n--- SCENE {sc_num:02d}: {heading} ---")

            for el in scene.get("elements", []):
                el_id = el["element_id"]
                tag = el.get("raw_tag", el.get("element_type", "ELEMENT"))
                s_raw = el.get("start_raw", "")
                e_raw = el.get("end_raw", "")
                dur_s = round(el.get("duration_cs", 0) / 100.0, 2)
                content = el.get("content", "").strip()
                lines.append(f"[{el_id} | {s_raw}->{e_raw} | dur={dur_s:.2f}s] **{tag}:** {content}")

        lines.append("")
    return "\n".join(lines)


class BatchExtractorEngine:
    def __init__(
        self,
        ast_path: Optional[Path] = None,
        raw_outputs_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None
    ):
        # Defensive path resolution
        if ast_path:
            self.ast_path = Path(ast_path)
        elif Path("ast_builder/01_structured_episodes.json").exists():
            self.ast_path = Path("ast_builder/01_structured_episodes.json")
        elif Path("../ast_builder/01_structured_episodes.json").exists():
            self.ast_path = Path("../ast_builder/01_structured_episodes.json")
        elif Path("01_structured_episodes.json").exists():
            self.ast_path = Path("01_structured_episodes.json")
        else:
            self.ast_path = Path("ast_builder/01_structured_episodes.json")

        if raw_outputs_dir:
            self.raw_outputs_dir = Path(raw_outputs_dir)
        elif Path("candidate_extractor/raw_llm_outputs").exists():
            self.raw_outputs_dir = Path("candidate_extractor/raw_llm_outputs")
        elif Path("candidate_extractor").exists():
            self.raw_outputs_dir = Path("candidate_extractor/raw_llm_outputs")
        elif Path("raw_llm_outputs").exists():
            self.raw_outputs_dir = Path("raw_llm_outputs")
        else:
            self.raw_outputs_dir = Path("candidate_extractor/raw_llm_outputs")

        if output_path:
            self.output_path = Path(output_path)
        elif Path("candidate_extractor").exists():
            self.output_path = Path("candidate_extractor/02_batch_extractions.json")
        else:
            self.output_path = Path("02_batch_extractions.json")

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.raw_outputs_dir.mkdir(parents=True, exist_ok=True)

        if not self.ast_path.exists():
            raise FileNotFoundError(f"AST not found at {self.ast_path}. Run AST builder first!")
        self.ast_data = json.loads(self.ast_path.read_text(encoding="utf-8"))

        # Build quick element lookup table
        self.element_lookup: Dict[str, Dict[str, Any]] = {}
        for ep in self.ast_data.get("episodes", []):
            for sc in ep.get("scenes", []):
                for el in sc.get("elements", []):
                    self.element_lookup[el["element_id"]] = el

        self.client = self._init_client()

    def _init_client(self):
        if not self.api_key:
            print("[WARN] GEMINI_API_KEY environment variable not detected.")
        try:
            from google import genai
            return genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
        except Exception as e:
            print(f"[WARN] Could not initialize google-genai Client: {e}")
            return None

    def call_llm_batch(self, batch_index: Any, episodes_slice: List[Dict[str, Any]]) -> BatchExtractionResponse:
        """Calls Gemini with structured Pydantic schema enforcing precise trim recommendations."""
        from google.genai import types

        transcript = format_batch_transcript(episodes_slice)
        batch_prompt = (
            f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"Here is the complete screenplay transcript for Batch {batch_index} "
            f"(Episodes {episodes_slice[0]['episode_num']} to {episodes_slice[-1]['episode_num']}):\n\n"
            f"{transcript}\n\n"
            f"Extract all single-episode promo candidates (targeting 25-55s duration), "
            f"precise trim recommendations, cross-episode narrative moments, and flat narrative context according to the JSON schema."
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchExtractionResponse,
            temperature=0.2,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=batch_prompt,
            config=config
        )

        if hasattr(response, "parsed") and response.parsed:
            return response.parsed
        return BatchExtractionResponse.model_validate_json(response.text)

    def process_batch(
        self,
        batch_index: Any,
        episodes_slice: List[Dict[str, Any]],
        force_refresh: bool = False,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ) -> Dict[str, Any]:
        """Processes a single batch with disk checkpointing and exponential backoff retries."""
        checkpoint_file = self.raw_outputs_dir / f"batch_{batch_index}.json"

        if checkpoint_file.exists() and not force_refresh:
            print(f"[CHECKPOINT] Batch {batch_index} found at {checkpoint_file.name}. Loading from disk.")
            return json.loads(checkpoint_file.read_text(encoding="utf-8"))

        start_ep = episodes_slice[0]["episode_num"]
        end_ep = episodes_slice[-1]["episode_num"]
        print(f"[API CALL] Processing Batch {batch_index} (Episodes {start_ep}..{end_ep}) using {self.model_name}...")

        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                result_obj = self.call_llm_batch(batch_index, episodes_slice)
                elapsed = time.time() - t0
                result_dict = result_obj.model_dump()
                print(f"           Batch {batch_index} completed in {elapsed:.2f}s "
                      f"({len(result_dict['candidates'])} candidates, {len(result_dict['narrative_moments'])} moments)")

                checkpoint_file.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
                print(f"           Saved checkpoint to {checkpoint_file.name}")
                return result_dict

            except Exception as e:
                print(f"[WARN] Batch {batch_index} attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    backoff = retry_delay * (2 ** (attempt - 1))
                    print(f"       Waiting {backoff:.1f}s before retrying Batch {batch_index}...")
                    time.sleep(backoff)
                else:
                    if len(episodes_slice) > 5:
                        print(f"[SPLIT] Splitting Batch {batch_index} into two 5-episode sub-batches...")
                        half = len(episodes_slice) // 2
                        res1 = self.process_batch(f"{batch_index}a", episodes_slice[:half], force_refresh=force_refresh, max_retries=max_retries, retry_delay=retry_delay)
                        res2 = self.process_batch(f"{batch_index}b", episodes_slice[half:], force_refresh=force_refresh, max_retries=max_retries, retry_delay=retry_delay)
                        merged = {
                            "candidates": res1["candidates"] + res2["candidates"],
                            "narrative_moments": res1["narrative_moments"] + res2["narrative_moments"],
                            "narrative_context": {
                                "character_aliases": res1["narrative_context"].get("character_aliases", []) + res2["narrative_context"].get("character_aliases", []),
                                "objects_and_clues": res1["narrative_context"].get("objects_and_clues", []) + res2["narrative_context"].get("objects_and_clues", []),
                                "relationships": res1["narrative_context"].get("relationships", []) + res2["narrative_context"].get("relationships", []),
                                "episode_summaries": res1["narrative_context"].get("episode_summaries", []) + res2["narrative_context"].get("episode_summaries", []),
                            }
                        }
                        checkpoint_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")
                        return merged
                    else:
                        print(f"[ERROR] Batch {batch_index} exhausted all {max_retries} attempts.")
                        raise

    def enrich_narrative_moment_with_ast(self, moment: Dict[str, Any]) -> Dict[str, Any]:
        """AST Grounding: Looks up start_element_id..end_element_id in AST and attaches verbatim elements."""
        s_id = moment.get("start_element_id")
        e_id = moment.get("end_element_id")
        s_el = self.element_lookup.get(s_id)
        e_el = self.element_lookup.get(e_id)

        if not s_el or not e_el:
            moment["duration_seconds"] = 0.0
            moment["source_elements_verbatim"] = []
            return moment

        ep_num = s_el["episode_num"]
        s_num_m = re.search(r"EL(\d+)", s_id)
        e_num_m = re.search(r"EL(\d+)", e_id)
        s_num = int(s_num_m.group(1)) if s_num_m else 1
        e_num = int(e_num_m.group(1)) if e_num_m else 1

        duration_cs = e_el["end_cs"] - s_el["start_cs"]
        moment["duration_seconds"] = round(duration_cs / 100.0, 2)
        moment["duration_cs"] = duration_cs

        verbatim_lines = []
        for el_num in range(s_num, e_num + 1):
            curr_id = f"EP{ep_num:02d}_EL{el_num:03d}"
            curr_el = self.element_lookup.get(curr_id)
            if curr_el:
                tag = curr_el.get("raw_tag", curr_el.get("element_type", "ELEMENT"))
                content = curr_el.get("content", "").strip()
                s_raw, e_raw = curr_el.get("start_raw", ""), curr_el.get("end_raw", "")
                verbatim_lines.append(f"[{curr_id} | {s_raw}-{e_raw}] **{tag}:** {content}")

        moment["source_elements_verbatim"] = verbatim_lines
        return moment

    def run_all(
        self,
        batch_size: int = 5,
        force_refresh: bool = False,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ) -> Dict[str, Any]:
        """Executes extraction across all episodes with AST enrichment and disk checkpointing."""
        episodes = self.ast_data.get("episodes", [])
        total_episodes = len(episodes)
        total_batches = (total_episodes + batch_size - 1) // batch_size

        print("=" * 70)
        print(f"STARTING BATCH EXTRACTION: {total_episodes} Episodes into {total_batches} Batches")
        print("=" * 70)

        all_candidates = []
        all_narrative_moments = []
        merged_aliases = []
        merged_clues = []
        merged_relationships = []
        merged_summaries = []

        for b_idx in range(1, total_batches + 1):
            start_i = (b_idx - 1) * batch_size
            end_i = min(start_i + batch_size, total_episodes)
            batch_slice = episodes[start_i:end_i]

            batch_result = self.process_batch(
                b_idx,
                batch_slice,
                force_refresh=force_refresh,
                max_retries=max_retries,
                retry_delay=retry_delay
            )

            for cand in batch_result.get("candidates", []):
                s_el = self.element_lookup.get(cand["start_element_id"])
                e_el = self.element_lookup.get(cand["end_element_id"])
                if s_el and e_el:
                    dur_cs = e_el["end_cs"] - s_el["start_cs"]
                    cand["calculated_duration_seconds"] = round(dur_cs / 100.0, 2)
                    cand["calculated_duration_cs"] = dur_cs
                all_candidates.append(cand)

            for mom in batch_result.get("narrative_moments", []):
                enriched_mom = self.enrich_narrative_moment_with_ast(mom)
                all_narrative_moments.append(enriched_mom)

            ctx = batch_result.get("narrative_context", {})
            merged_aliases.extend(ctx.get("character_aliases", []))
            merged_clues.extend(ctx.get("objects_and_clues", []))
            merged_relationships.extend(ctx.get("relationships", []))
            merged_summaries.extend(ctx.get("episode_summaries", []))

        final_payload = {
            "series_title": self.ast_data.get("corpus_title", "Microdrama Series"),
            "total_episodes": total_episodes,
            "total_single_episode_candidates": len(all_candidates),
            "total_narrative_moments": len(all_narrative_moments),
            "candidates": all_candidates,
            "narrative_moments": all_narrative_moments,
            "narrative_context": {
                "character_aliases": merged_aliases,
                "objects_and_clues": merged_clues,
                "relationships": merged_relationships,
                "episode_summaries": merged_summaries
            }
        }

        self.output_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
        file_size_kb = self.output_path.stat().st_size / 1024

        print("\n" + "=" * 70)
        print("CANDIDATE & NARRATIVE MOMENT EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"Single-Episode Candidates Extracted: {len(all_candidates):4d}")
        print(f"Narrative Moments (AST Attached)   : {len(all_narrative_moments):4d}")
        print(f"Character Aliases Mapped           : {len(merged_aliases):4d}")
        print(f"Key Props and Clues Cataloged      : {len(merged_clues):4d}")
        print(f"Saved merged extractions to        : {self.output_path.resolve()} ({file_size_kb:.1f} KB)")
        print("=" * 70)

        return final_payload


def main():
    parser = argparse.ArgumentParser(description="Batch Candidate & Narrative Memory Extractor")
    parser.add_argument("--ast", type=Path, default=None, help="Path to canonical 01_structured_episodes.json")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory for raw LLM batch checkpoints")
    parser.add_argument("--output", type=Path, default=None, help="Path to save 02_batch_extractions.json")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--batch-size", type=int, default=5, help="Episodes per extraction batch")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh and bypass existing checkpoints")
    parser.add_argument("--max-retries", type=int, default=3, help="Max API call retries with exponential backoff")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="Initial retry delay in seconds")
    args = parser.parse_args()

    extractor = BatchExtractorEngine(
        ast_path=args.ast,
        raw_outputs_dir=args.raw_dir,
        output_path=args.output,
        model_name=args.model
    )
    extractor.run_all(
        batch_size=args.batch_size,
        force_refresh=args.force_refresh,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay
    )


if __name__ == "__main__":
    main()
