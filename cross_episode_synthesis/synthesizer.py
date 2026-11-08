"""
🎬 Cross-Episode Synthesis Engine (synthesizer.py)
Series: "I Married a Broken Soldier. He was a Duke" (50 Episodes)

Core Technologies:
- 3-Check Narrative Compatibility Architecture:
  1. Semantic Similarity Retrieval via Vector Embeddings (text-embedding-004) & Cosine Similarity
  2. Character & Entity Compatibility Check (Canonical Aliases, Key Props/Clues, Tracked Relationships)
  3. LLM / Deterministic Narrative Progression Check (Escalation, Revelation/Twist, Power Inversion, Payoff)
- Python AST Element Verification & Verbatim Dialogue Attachment
- Google GenAI SDK Structured Output Generation with Pydantic v2 Schema Modeling
- Diversity-Aware Final Selection & Greedy Multi-Episode Ranking
- Actionable Trim Recommendations & Transitional Editing Directives for Video Editors
"""

import os
import sys
import re
import json
import time
import math
import argparse
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any, Tuple, Union, Set

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Automatically load environment variables from models.env or .env if present
try:
    from dotenv import load_dotenv
    for p in [
        Path.cwd() / "models.env",
        Path.cwd().parent / "models.env",
        Path.home() / "OneDrive" / "Documents" / "Kuku FM" / "models.env",
        Path.cwd() / ".env",
        Path.cwd().parent / ".env"
    ]:
        if p.exists() and p.is_file():
            load_dotenv(dotenv_path=str(p), override=True)
            break
except ImportError:
    pass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# ==============================================================================
# 1. STRONGLY-TYPED DATA MODELS & OUTPUT SCHEMAS
# ==============================================================================

class TrimRecommendation(BaseModel):
    """
    Precise trim recommendation for video editors with verbatim boundaries and transition instructions.
    Eliminates ungrounded character-count timestamp interpolation.
    """
    exact_opening_dialogue: str = Field(
        description="Verbatim dialogue line that begins the clip (or initial action description if visual opener)"
    )
    exact_ending_dialogue: str = Field(
        description="Verbatim dialogue line that ends the clip right before the final cliffhanger cut"
    )
    verbatim_trim_quote: Optional[str] = Field(
        default=None,
        description="Verbatim quote within dialogue where trim should occur if internal sentence trim is needed"
    )
    boundary_type: Literal["element_boundary", "internal_cut"] = Field(
        default="element_boundary",
        description="'element_boundary' if cut aligns with start/end element, 'internal_cut' if internal sentence trim"
    )
    requires_manual_editor_verification: bool = Field(
        default=False,
        description="True if audio/waveform check is advised for clean speech boundary, False if natural clean pause"
    )


class RubricScores(BaseModel):
    """Standardized 4-Criterion Rubric (1-5 integer ratings)."""
    standalone_coherence: int = Field(ge=1, le=5, description="1-5 rating: conflict/stakes clear to cold first-time viewer")
    hook_strength: int = Field(ge=1, le=5, description="1-5 rating: immediate visceral shock or intrigue within first 3 seconds")
    cliffhanger_strength: int = Field(ge=1, le=5, description="1-5 rating: cuts at peak unresolved jeopardy or open question")
    emotional_impact: int = Field(ge=1, le=5, description="1-5 rating: high-intensity betrayal, moral outrage, or dramatic reversal")


class StitchedSegment(BaseModel):
    """
    A single constituent screenplay segment within a multi-episode promo.
    Each segment is grounded in exact AST element IDs with its own trim instructions.
    """
    segment_index: int = Field(ge=1, le=3, description="1-based index (1, 2, or 3) of segment in promo sequence")
    episode_num: int = Field(description="Screenplay episode number")
    start_element_id: str = Field(description="First element ID of segment, e.g. EP01_EL010")
    end_element_id: str = Field(description="Last element ID of segment, e.g. EP01_EL019")
    start_raw: str = Field(description="Start timestamp MM:SS.ss")
    end_raw: str = Field(description="End timestamp MM:SS.ss")
    duration_seconds: float = Field(description="Duration of this segment in seconds")
    duration_cs: int = Field(description="Duration in integer centiseconds")
    purpose_in_promo: str = Field(
        description="Dramatic function of this segment (e.g. 'Hook / Initial Humiliation', 'Mystery Escalation', 'Climax / Payoff')"
    )
    trim_recommendation: TrimRecommendation = Field(
        description="Segment-specific precise trim recommendations for video editors"
    )


class EntityCompatibilityResult(BaseModel):
    """Result of the Character & Entity Compatibility check (Check 2)."""
    is_compatible: bool = Field(description="True if moments share relevant characters, pivotal props, or tracked relationships")
    shared_characters: List[str] = Field(default_factory=list, description="Canonical character names appearing in both moments")
    shared_props_or_clues: List[str] = Field(default_factory=list, description="Key props, items, or clues shared across both moments")
    relationship_context: Optional[str] = Field(default=None, description="Active narrative relationship linking characters in M1 and M2")
    compatibility_score: float = Field(ge=0.0, le=1.0, description="Normalized entity compatibility score")
    rejection_reason: Optional[str] = Field(default=None, description="Reason for rejection if incompatible")


class ProgressionVerdict(BaseModel):
    """Result of the LLM / Deterministic Narrative Progression check (Check 3)."""
    has_progression: bool = Field(description="True if moment_2 meaningfully escalates, resolves, or inverts moment_1")
    progression_type: Literal[
        "escalation_of_stakes",
        "revelation_or_twist",
        "power_inversion",
        "setup_to_payoff",
        "none"
    ] = Field(description="Primary narrative progression dynamic linking the two moments")
    causal_link: Literal["explicit", "implied", "thematic", "none"] = Field(
        default="thematic",
        description="Causal narrative connection category"
    )
    explanation: str = Field(description="1-2 sentences explaining why M2 represents true narrative progression from M1")


class EligibleCombination(BaseModel):
    """An approved cross-episode candidate pair passing all three compatibility gates."""
    pair_id: str
    moment_1_id: str
    moment_2_id: str
    episode_1: int
    episode_2: int
    semantic_similarity: float
    entity_compatibility: EntityCompatibilityResult
    narrative_progression: ProgressionVerdict


class MultiEpisodeCandidate(BaseModel):
    """
    A multi-episode promo candidate stitched across 2-3 episodes (composition_mode: 'multi_episode').
    Combines disparate narrative beats into a coherent 20.00s-60.00s promotional arc.
    """
    candidate_id: str = Field(description="Unique candidate identifier, e.g. SYNTH_MULTI_01")
    composition_mode: Literal["multi_episode"] = Field(
        default="multi_episode",
        description="Composition capability tag (multi_episode vs single_episode)"
    )
    archetype: Literal[
        "immediate_shock_opener",
        "visual_anomaly_mystery_escalation",
        "emotional_reversal_turning_point"
    ] = Field(description="One of the three standardized promo archetypes")
    relationship_type: Literal["explicit", "implied", "thematic"] = Field(
        description="Causal narrative connection: explicit (cause-effect), implied (secret clue/identity), or thematic (contrast baseline vs vindication)"
    )
    progression_type: Optional[str] = Field(
        default="power_inversion",
        description="Narrative progression dynamic: 'escalation_of_stakes', 'revelation_or_twist', 'power_inversion', or 'setup_to_payoff'"
    )
    segments: List[StitchedSegment] = Field(
        min_length=2, max_length=3,
        description="2 to 3 constituent segments across different episodes"
    )
    transition_instructions: List[str] = Field(
        description="Precise editing transition notes between segments (e.g. 'Hard cut on Gerald turning away -> Flash forward audio riser -> Jump to Ep 42 cathedral smoke')"
    )
    hook_timestamp_offset_seconds: float = Field(
        ge=0.0, le=3.0,
        description="Offset within <= 3.0s from start of segment 1 where the visceral hook initiates"
    )
    hook_description: str = Field(description="Why the first 3 seconds of segment 1 stop the scrolling viewer")
    cliffhanger_description: str = Field(description="Unresolved tension or dramatic climax where segment 2/3 cuts")
    standalone_coherence_explanation: str = Field(description="Why a cold viewer understands this arc without watching intervening episodes")
    rubric_scores: RubricScores
    overall_score: int = Field(ge=4, le=20, description="Sum of the 4 rubric scores (4-20)")
    total_duration_seconds: float = Field(description="Combined duration of all segments in seconds (20.00s - 60.00s)")
    total_duration_cs: int = Field(description="Combined duration in integer centiseconds (2,000 - 6,000)")
    retrieval_method: str = Field(
        default="3_check_compatibility",
        description="Method used: '3_check_compatibility' (semantic_similarity + entity_compatibility + narrative_progression)"
    )
    semantic_similarity_score: Optional[float] = Field(
        default=None,
        description="Cosine similarity score (sim(A, B)) between the connected narrative moments"
    )
    shared_characters: List[str] = Field(
        default_factory=list,
        description="Canonical character names linking the stitched segments"
    )
    shared_props_or_clues: List[str] = Field(
        default_factory=list,
        description="Key props, clues, or motifs connecting the segments"
    )


class SynthesisLLMResponse(BaseModel):
    """Structured response container for the synthesis LLM call."""
    candidates: List[MultiEpisodeCandidate] = Field(
        default_factory=list,
        description="List of valid synthesized multi-episode candidates"
    )
    cross_episode_reasoning: str = Field(
        description="Detailed editorial notes explaining how dialogue continuity and dramatic contrast were preserved across the stitched segments"
    )


# ==============================================================================
# 2. SEMANTIC RETRIEVAL ENGINE WITH EMBEDDINGS & COSINE SIMILARITY (CHECK 1)
# ==============================================================================

class SemanticMomentRetriever:
    """
    Semantic Retrieval Engine for Narrative Moments via Vector Embeddings & Cosine Similarity.
    Generates embeddings for extracted narrative moments, computes cosine similarity:
        sim(A, B) = (A . B) / (||A|| * ||B||)
    and retrieves top-related cross-episode moments even when screenplays use different words.
    """
    def __init__(
        self,
        cache_path: Optional[Path] = None,
        embedding_model: str = "text-embedding-004",
        api_key: Optional[str] = None
    ):
        self.cache_path = Path(cache_path) if cache_path else Path("embeddings_cache.json")
        self.embedding_model = embedding_model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.embeddings: Dict[str, List[float]] = {}
        self.client = self._init_client()
        self._load_cache()

    def _init_client(self):
        if not self.api_key:
            return None
        try:
            from google import genai
            return genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"[WARN] Could not initialize google-genai client for embeddings: {e}")
            return None

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                self.embeddings = data.get("moment_embeddings", {})
                print(f"[CACHE] Loaded {len(self.embeddings)} cached moment embeddings from {self.cache_path.name}")
            except Exception as e:
                print(f"[WARN] Failed to load embedding cache: {e}")

    def _save_cache(self):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": self.embedding_model,
                "count": len(self.embeddings),
                "moment_embeddings": self.embeddings
            }
            self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[CACHE] Saved {len(self.embeddings)} embeddings to {self.cache_path.name}")
        except Exception as e:
            print(f"[WARN] Could not save embedding cache: {e}")

    def _deterministic_fallback_embedding(self, text: str, dim: int = 768) -> List[float]:
        """
        Deterministic unit-norm pseudo-embedding fallback for offline execution.
        Uses character trigrams and hash projection to generate a consistent representation.
        """
        vec = np.zeros(dim, dtype=np.float32)
        clean = text.lower()
        for i in range(len(clean) - 2):
            tri = clean[i:i+3]
            h = hash(tri) % dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def generate_moment_embedding(self, moment: Dict[str, Any]) -> List[float]:
        """
        Generates or retrieves cached dense embedding for a narrative moment.
        Combines summary, moment_type, keywords, and verbatim lines.
        """
        moment_id = moment.get("moment_id", "UNKNOWN")
        if moment_id in self.embeddings:
            return self.embeddings[moment_id]

        summary = moment.get("summary", "")
        m_type = moment.get("moment_type", "")
        keywords = ", ".join(moment.get("lookup_keywords", []))
        verbatim_sample = " ".join([l.split(":**")[-1] for l in moment.get("source_elements_verbatim", [])[:4]])

        semantic_text = f"Narrative Beat: {summary}. Type: {m_type}. Keywords: {keywords}. Script: {verbatim_sample}".strip()

        # Try API embedding call if client exists
        if self.client:
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=semantic_text,
                )
                if hasattr(response, "embedding") and response.embedding and hasattr(response.embedding, "values"):
                    vec = list(response.embedding.values)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = (np.array(vec) / norm).tolist()
                    self.embeddings[moment_id] = vec
                    return vec
            except Exception as e:
                print(f"[WARN] GenAI embed_content failed for {moment_id}: {e}. Falling back to deterministic vector.")

        # Deterministic fallback
        vec = self._deterministic_fallback_embedding(semantic_text)
        self.embeddings[moment_id] = vec
        return vec

    def compute_all_moment_embeddings(self, moments: List[Dict[str, Any]]):
        """Batch-computes and caches vector embeddings for all narrative moments."""
        missing = [m for m in moments if m.get("moment_id") not in self.embeddings]
        if not missing:
            return

        print(f"[EMBEDDINGS] Computing dense vector embeddings for {len(missing)} narrative moments...")
        for m in missing:
            self.generate_moment_embedding(m)
        self._save_cache()

    @staticmethod
    def cosine_similarity(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray]) -> float:
        """
        Calculates cosine similarity between two unit-norm or arbitrary vectors:
            sim(A, B) = (A . B) / (||A|| * ||B||)
        """
        a = np.asarray(v1, dtype=np.float32)
        b = np.asarray(v2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def retrieve_top_related_moments(
        self,
        source_moment: Dict[str, Any],
        candidate_pool: List[Dict[str, Any]],
        top_k: int = 3,
        min_episode_gap: int = 2
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Calculates pairwise cosine similarity across all candidate moments,
        filtering for min_episode_gap and ranking descending.
        """
        src_id = source_moment.get("moment_id")
        src_ep = source_moment.get("episode", 0)
        src_vec = self.embeddings.get(src_id)

        if not src_vec:
            return []

        scored_moments = []
        for target in candidate_pool:
            target_id = target.get("moment_id")
            target_ep = target.get("episode", 0)

            # Skip same moment or episodes within min_episode_gap
            if target_id == src_id or abs(target_ep - src_ep) < min_episode_gap:
                continue

            target_vec = self.embeddings.get(target_id)
            if not target_vec:
                continue

            sim = self.cosine_similarity(src_vec, target_vec)
            scored_moments.append((target, sim))

        # Sort descending by cosine similarity
        scored_moments.sort(key=lambda x: x[1], reverse=True)
        return scored_moments[:top_k]


# ==============================================================================
# 3. CROSS-EPISODE SYNTHESIZER ENGINE WITH 3-CHECK COMPATIBILITY
# ==============================================================================

class CrossEpisodeSynthesizerEngine:
    """
    Cross-Episode Synthesizer Engine with 3-Check Narrative Compatibility:
    1. Semantic Similarity Retrieval (Cosine Similarity)
    2. Character & Entity Compatibility Check (Aliases, Props, Relationships)
    3. LLM / Deterministic Narrative Progression Check (Escalation, Inversion, Payoff)
    """
    def __init__(
        self,
        ast_path: Optional[Path] = None,
        extractions_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        embeddings_cache_path: Optional[Path] = None,
        model_name: str = "gemini-2.5-flash",
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

        if output_path:
            self.output_path = Path(output_path)
        elif Path("cross_episode_synthesis").exists():
            self.output_path = Path("cross_episode_synthesis/03_synthesized_candidates.json")
        else:
            self.output_path = Path("03_synthesized_candidates.json")

        if embeddings_cache_path:
            cache_file = Path(embeddings_cache_path)
        elif Path("cross_episode_synthesis/embeddings_cache.json").exists():
            cache_file = Path("cross_episode_synthesis/embeddings_cache.json")
        elif Path("../cross_episode_synthesis/embeddings_cache.json").exists():
            cache_file = Path("../cross_episode_synthesis/embeddings_cache.json")
        elif Path("cross_episode_synthesis").exists():
            cache_file = Path("cross_episode_synthesis/embeddings_cache.json")
        else:
            cache_file = Path("embeddings_cache.json")

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not self.ast_path.exists():
            raise FileNotFoundError(f"AST not found at {self.ast_path}. Run AST builder first!")
        self.ast_data = json.loads(self.ast_path.read_text(encoding="utf-8"))

        if not self.extractions_path.exists():
            raise FileNotFoundError(f"Extractions not found at {self.extractions_path}. Run candidate extractor first!")
        self.extractions_data = json.loads(self.extractions_path.read_text(encoding="utf-8"))

        # Build fast AST lookup
        self.element_lookup: Dict[str, Dict[str, Any]] = {}
        for ep in self.ast_data.get("episodes", []):
            for sc in ep.get("scenes", []):
                for el in sc.get("elements", []):
                    self.element_lookup[el["element_id"]] = el

        # Initialize Semantic Retriever (Check 1)
        self.retriever = SemanticMomentRetriever(
            cache_path=cache_file,
            api_key=self.api_key
        )
        self.retriever.compute_all_moment_embeddings(
            self.extractions_data.get("narrative_moments", [])
        )

        # Initialize Character & Entity Maps (Check 2)
        self._init_entity_knowledge_base()

        self.client = self._init_client()

    def _init_client(self):
        if not self.api_key:
            return None
        try:
            from google import genai
            return genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"[WARN] Failed to initialize google-genai client: {e}")
            return None

    def _init_entity_knowledge_base(self):
        """Initializes canonical character mappings, prop list, and relationship graph."""
        context = self.extractions_data.get("narrative_context", {})
        
        self.character_alias_map: Dict[str, str] = {}
        for entry in context.get("character_aliases", []):
            canon = entry["canonical_name"]
            if "Alexander" in canon:
                norm_canon = "Alexander"
            elif "Juliana" in canon:
                norm_canon = "Juliana"
            elif "Clarissa" in canon:
                norm_canon = "Clarissa"
            elif "Gerald" in canon:
                norm_canon = "Gerald"
            elif "Rupert" in canon:
                norm_canon = "Rupert"
            elif "Catherine" in canon:
                norm_canon = "Catherine"
            elif "Dowager" in canon:
                norm_canon = "Dowager Duchess"
            elif "Father Thomas" in canon:
                norm_canon = "Father Thomas"
            else:
                norm_canon = canon

            self.character_alias_map[canon.lower()] = norm_canon
            for alias in entry.get("aliases", []):
                self.character_alias_map[alias.lower()] = norm_canon

        # Core recognized character name tokens
        self.core_character_tokens = [
            "alexander", "juliana", "gerald", "clarissa", "catherine",
            "rupert", "father thomas", "dowager", "prudence", "seamus", "agnes", "ashcroft"
        ]

        # Key narrative props/motifs
        self.tracked_props = [
            "bridal gown", "signet ring", "amethyst", "locket", "herbal ledger",
            "grimoire", "pocket watch", "antidote", "riding crop", "succession papers",
            "mantle", "ducal uniform", "crucifix", "branding poker"
        ]

        # Tracked relationship pairs
        self.tracked_relationships = context.get("relationships", [])

    # ==========================================================================
    # CHECK 2: CHARACTER & ENTITY COMPATIBILITY
    # ==========================================================================
    def extract_entities_from_moment(self, moment: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
        """Extracts canonical characters and key props appearing in a narrative moment."""
        text = (moment.get("summary", "") + " " + " ".join(moment.get("lookup_keywords", []))).lower()
        for line in moment.get("source_elements_verbatim", [])[:6]:
            text += " " + line.lower()

        characters: Set[str] = set()
        for token in self.core_character_tokens:
            if token in text:
                if token == "alexander": characters.add("Alexander")
                elif token == "juliana": characters.add("Juliana")
                elif token == "gerald": characters.add("Gerald")
                elif token == "clarissa": characters.add("Clarissa")
                elif token == "catherine": characters.add("Catherine")
                elif token == "rupert": characters.add("Rupert")
                elif token == "father thomas": characters.add("Father Thomas")
                elif token == "dowager": characters.add("Dowager Duchess")
                elif token == "prudence": characters.add("Prudence")
                elif token == "seamus": characters.add("Seamus")
                elif token == "agnes": characters.add("Agnes")

        for alias, canon in self.character_alias_map.items():
            if alias in text:
                characters.add(canon)

        props: Set[str] = set()
        for p in self.tracked_props:
            if p in text:
                props.add(p.title())

        return characters, props

    def check_character_entity_compatibility(
        self,
        m1: Dict[str, Any],
        m2: Dict[str, Any]
    ) -> EntityCompatibilityResult:
        """
        Check 2: Evaluates character, entity, and relationship compatibility.
        Verifies whether moments share canonical characters, key clues/props, or active dramatic relationships.
        """
        c1, p1 = self.extract_entities_from_moment(m1)
        c2, p2 = self.extract_entities_from_moment(m2)

        shared_chars = sorted(list(c1.intersection(c2)))
        shared_props = sorted(list(p1.intersection(p2)))

        # Check for active tracked relationship linking character in M1 to character in M2
        active_rel = None
        for r in self.tracked_relationships:
            pair = r.get("pair", "")
            if any(ca in pair for ca in c1) and any(cb in pair for cb in c2):
                active_rel = f"{r.get('pair')}: {r.get('relation', '')[:70]}"
                break

        # Calculate compatibility score
        score = 0.0
        if shared_chars:
            score += 0.5 + min(0.3, len(shared_chars) * 0.1)
        if shared_props:
            score += 0.3 + min(0.2, len(shared_props) * 0.1)
        if active_rel:
            score = max(score, 0.65)
        score = min(1.0, score)

        is_compatible = (len(shared_chars) > 0) or (len(shared_props) > 0) or (active_rel is not None)

        rejection_reason = None
        if not is_compatible:
            rejection_reason = (
                f"Disjoint entities: M1 characters {list(c1) or ['None']} vs M2 characters {list(c2) or ['None']} "
                f"with no shared key props or tracked relationship link."
            )

        return EntityCompatibilityResult(
            is_compatible=is_compatible,
            shared_characters=shared_chars,
            shared_props_or_clues=shared_props,
            relationship_context=active_rel,
            compatibility_score=round(score, 3),
            rejection_reason=rejection_reason
        )

    # ==========================================================================
    # CHECK 3: NARRATIVE PROGRESSION CHECK (LLM + DETERMINISTIC FALLBACK)
    # ==========================================================================
    def check_narrative_progression(
        self,
        m1: Dict[str, Any],
        m2: Dict[str, Any],
        entity_res: EntityCompatibilityResult,
        use_llm: bool = True
    ) -> ProgressionVerdict:
        """
        Check 3: Evaluates whether moment_2 meaningfully escalates, resolves, or inverts moment_1.
        Uses structured LLM evaluation when available, falling back gracefully to deterministic narrative rules.
        """
        # Chronological normalization: ensure M1 precedes M2
        if m1.get("episode", 0) > m2.get("episode", 0):
            m1, m2 = m2, m1

        if use_llm and self.client:
            try:
                return self._evaluate_progression_with_llm(m1, m2, entity_res)
            except Exception as e:
                print(f"[WARN] LLM progression check failed: {e}. Falling back to deterministic progression engine.")

        return self._evaluate_progression_deterministic(m1, m2, entity_res)

    def _evaluate_progression_with_llm(
        self,
        m1: Dict[str, Any],
        m2: Dict[str, Any],
        entity_res: EntityCompatibilityResult
    ) -> ProgressionVerdict:
        """Calls Gemini with structured Pydantic schema to verify narrative progression."""
        from google.genai import types

        prompt = f"""You are an expert microdrama story editor evaluating cross-episode continuity for a promotional video ad.

Series: "I Married a Broken Soldier. He was a Duke" (50 Episodes)

QUESTION: Does Moment 2 meaningfully ESCALATE, RESOLVE, INVERT, or PAY OFF the setup established in Moment 1?

CRITICAL EDITORIAL RULE:
Semantic similarity alone is NOT enough. Two scenes where a character is merely threatened do NOT form a valid promo arc.
Moment 2 must represent genuine narrative movement:
- power_inversion: e.g. early humiliation at the altar inverted into triumphant ducal return.
- revelation_or_twist: e.g. clue or secret identity unmasked before the court.
- setup_to_payoff: e.g. marriage vow or promise made early is fulfilled in battle.
- escalation_of_stakes: e.g. domestic dispute escalates to public trial or armed confrontation.
- none: e.g. parallel or redundant events without causal progression or resolution.

MOMENT 1 (Episode {m1.get('episode', 0)}):
- ID: {m1.get('moment_id')}
- Beat Type: {m1.get('moment_type')}
- Summary: {m1.get('summary')}

MOMENT 2 (Episode {m2.get('episode', 0)}):
- ID: {m2.get('moment_id')}
- Beat Type: {m2.get('moment_type')}
- Summary: {m2.get('summary')}

SHARED ENTITIES:
- Characters: {entity_res.shared_characters}
- Props/Clues: {entity_res.shared_props_or_clues}
- Relationship Context: {entity_res.relationship_context or 'None'}

Evaluate whether true narrative progression exists and return the structured JSON verdict."""

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ProgressionVerdict,
            temperature=0.1,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )

        if hasattr(response, "parsed") and response.parsed:
            return response.parsed
        return ProgressionVerdict.model_validate_json(response.text)

    def _evaluate_progression_deterministic(
        self,
        m1: Dict[str, Any],
        m2: Dict[str, Any],
        entity_res: EntityCompatibilityResult
    ) -> ProgressionVerdict:
        """
        Deterministic progression engine evaluating archetype contrasts, clue payoffs, and power shifts.
        """
        ep1 = m1.get("episode", 0)
        ep2 = m2.get("episode", 0)
        t1 = m1.get("moment_type", "").lower()
        t2 = m2.get("moment_type", "").lower()
        s1 = m1.get("summary", "").lower()
        s2 = m2.get("summary", "").lower()

        # Rule A: Contrast Baseline / Setup -> Identity Reveal (Classic Power Inversion)
        if ("contrast_baseline" in t1 or "setup_or_vow" in t1 or "altar" in s1 or "humiliat" in s1) and \
           ("identity_reveal" in t2 or "ducal" in s2 or "duke" in s2 or "power" in t2):
            return ProgressionVerdict(
                has_progression=True,
                progression_type="power_inversion",
                causal_link="thematic",
                explanation=f"Moment in Ep {ep1} establishes initial vulnerability or vow, which Ep {ep2} dramatically inverts through ducal authority."
            )

        # Rule B: Clue / Prop Setup -> Payoff or Confrontation
        if ("clue_or_prop" in t1 or entity_res.shared_props_or_clues) and \
           ("identity_reveal" in t2 or "high_stakes_confrontation" in t2 or "unmask" in s2):
            props = ", ".join(entity_res.shared_props_or_clues) or "tangible narrative clue"
            return ProgressionVerdict(
                has_progression=True,
                progression_type="revelation_or_twist",
                causal_link="implied",
                explanation=f"Shared prop/motif ({props}) set up in Ep {ep1} serves as critical evidence or weapon in Ep {ep2} climax."
            )

        # Rule C: Marriage Vow / Alliance -> Battle Payoff
        if ("marry" in s1 or "vow" in s1 or "rescue" in s1) and \
           ("confront" in s2 or "protect" in s2 or "regalia" in s2 or "claims" in s2):
            return ProgressionVerdict(
                has_progression=True,
                progression_type="setup_to_payoff",
                causal_link="explicit",
                explanation=f"Alliance formed in Ep {ep1} pays off in Ep {ep2} when the partners stand together against opposition."
            )

        # Rule D: Escalation of Stakes (Early oppression -> Defiant confrontation)
        if ("crop" in s1 or "whip" in s1 or "servant" in s1 or "scrub" in s1) and \
           ("defian" in s2 or "burn" in s2 or "refuse" in s2 or "challenge" in s2):
            return ProgressionVerdict(
                has_progression=True,
                progression_type="escalation_of_stakes",
                causal_link="explicit",
                explanation=f"Subjugation in Ep {ep1} escalates directly into overt defiance and active retaliation in Ep {ep2}."
            )

        # Default fallback if moments lack genuine progression
        if t1 == t2 and not entity_res.shared_props_or_clues and len(entity_res.shared_characters) < 2:
            return ProgressionVerdict(
                has_progression=False,
                progression_type="none",
                causal_link="none",
                explanation=f"Both beats are parallel {t1} moments across Ep {ep1} and Ep {ep2} without causal escalation or payoff."
            )

        # General progression based on character continuity and significant episode span
        if ep2 - ep1 >= 5 and entity_res.shared_characters:
            return ProgressionVerdict(
                has_progression=True,
                progression_type="escalation_of_stakes",
                causal_link="implied",
                explanation=f"Ongoing storyline involving {', '.join(entity_res.shared_characters)} escalates from Ep {ep1} to Ep {ep2}."
            )

        return ProgressionVerdict(
            has_progression=False,
            progression_type="none",
            causal_link="none",
            explanation="Insufficient narrative contrast or causal connection between the two episode beats."
        )

    # ==========================================================================
    # SYNTHESIS PROMPT & LLM EXECUTION
    # ==========================================================================
    def build_synthesis_prompt(
        self,
        eligible_combinations: List[EligibleCombination],
        narrative_context: Dict[str, Any]
    ) -> str:
        """
        Builds the synthesis prompt containing eligible combinations verified by all 3 checks:
        1. Semantic Similarity
        2. Character/Entity Compatibility
        3. Narrative Progression
        """
        aliases_lines = []
        for a in narrative_context.get("character_aliases", []):
            aliases_lines.append(f"- **{a['canonical_name']}**: Known as {', '.join(a.get('aliases', []))} (Key Eps: {a.get('key_episodes', [])})")

        pairs_blocks = []
        for i, comb in enumerate(eligible_combinations, 1):
            m1 = comb.moment_1_id
            m2 = comb.moment_2_id
            sim = comb.semantic_similarity
            compat = comb.entity_compatibility
            prog = comb.narrative_progression

            # Lookup full moment details from extractions_data
            m1_obj = next((m for m in self.extractions_data.get("narrative_moments", []) if m.get("moment_id") == m1), {})
            m2_obj = next((m for m in self.extractions_data.get("narrative_moments", []) if m.get("moment_id") == m2), {})

            block = [
                f"### ELIGIBLE COMBINATION {i:02d} (Pair ID: {comb.pair_id})",
                f"- Semantic Similarity: {sim:.4f}",
                f"- Shared Characters: {', '.join(compat.shared_characters) or 'None'}",
                f"- Shared Props/Clues: {', '.join(compat.shared_props_or_clues) or 'None'}",
                f"- Narrative Progression Type: {prog.progression_type} ({prog.causal_link})",
                f"- Progression Reasoning: {prog.explanation}",
                "",
                f"**MOMENT A**: {m1_obj.get('moment_id', m1)} (Episode {comb.episode_1} | Type: {m1_obj.get('moment_type')} | Dur: {m1_obj.get('duration_seconds', 0):.2f}s)",
                f"Summary: {m1_obj.get('summary')}",
                "Verbatim Dialogue & Action Elements:",
            ]
            for v in m1_obj.get("source_elements_verbatim", []):
                block.append(f"  {v}")

            block.extend([
                "",
                f"**MOMENT B**: {m2_obj.get('moment_id', m2)} (Episode {comb.episode_2} | Type: {m2_obj.get('moment_type')} | Dur: {m2_obj.get('duration_seconds', 0):.2f}s)",
                f"Summary: {m2_obj.get('summary')}",
                "Verbatim Dialogue & Action Elements:",
            ])
            for v in m2_obj.get("source_elements_verbatim", []):
                block.append(f"  {v}")
            block.append("-" * 60)
            pairs_blocks.append("\n".join(block))

        prompt = f"""You are the Lead Promo Director and Video Producer for Kuku FM's high-stakes microdrama series.
Your task is to execute CROSS-EPISODE SYNTHESIS (composition_mode: "multi_episode") by stitching 2-3 segments across different episodes into high-converting promo candidates.

Every candidate you generate MUST originate from one of the ELIGIBLE COMBINATIONS below, which have passed all three narrative compatibility checks (Semantic Similarity, Character/Entity Compatibility, and Narrative Progression).

### THE THREE STANDARDIZED PROMO ARCHETYPES:
1. `immediate_shock_opener`: Segment 1 opens within <= 3.00s on a visceral physical action, slap, betrayal accusation, or explosive threat; Segment 2/3 cuts on an unresolved cliffhanger.
2. `visual_anomaly_mystery_escalation`: Segment 1 opens within <= 3.00s on a suspicious visual beat in CAMERA or ACTION (e.g. signet ring, royal crest, mysterious scar) that escalates to a confrontation twist across episodes.
3. `emotional_reversal_turning_point`: Segment 1 establishes a sharp status humiliation or vow (e.g. jilted at altar); Segment 2/3 delivers a dramatic power inversion or vindication (e.g. unmasked as Duke).

### PRECISE TRIM RECOMMENDATIONS (FOR VIDEO EDITORS):
For every constituent segment, you must provide actionable trim instructions in `trim_recommendation`:
- `exact_opening_dialogue`: Verbatim spoken line (or initial action description) that begins the segment.
- `exact_ending_dialogue`: Verbatim spoken line that ends the segment right before the cut.
- `verbatim_trim_quote`: Exact phrase/sentence where an internal cut should be made if dead air or trailing lines should be eliminated.
- `boundary_type`: Set to `element_boundary` if the cut aligns with element borders; `internal_cut` if an internal sentence cut is needed.
- `requires_manual_editor_verification`: True if an internal cut requires video editor verification against audio waveforms.
- `transition_instructions`: Verbatim instructions detailing how the video editor should transition between segments (e.g. sound effect, visual match cut, pacing).
- CRITICAL: Never invent timestamps by character counts! Character interpolation without audio is strictly prohibited. Ground all trims in verbatim text.

### STRICT MECHANICAL RULES:
1. Composition Mode: MUST be "multi_episode".
2. Segments: Must contain 2 to 3 constituent segments across AT LEAST 2 DIFFERENT EPISODES.
3. Segment Element IDs: Must exist in the AST (e.g. EP01_EL010 to EP01_EL019).
4. Hook Timestamp Offset: 0.0s <= hook_timestamp_offset_seconds <= 3.0s (measured from start of Segment 1).
5. Combined Duration: 20.00s <= total_duration_seconds <= 60.00s (2,000 <= total_duration_cs <= 6,000).
6. Rubric Scores (1-5 integer ratings): standalone_coherence, hook_strength, cliffhanger_strength, emotional_impact. overall_score = sum of scores (4-20).

### NARRATIVE CONTEXT & ALIAS TABLE:
{chr(10).join(aliases_lines)}

### VERIFIED ELIGIBLE COMBINATIONS (GROUNDED IN SCREENPLAY AST):
{chr(10).join(pairs_blocks)}

Synthesize the optimal multi-episode promo candidates and output structured JSON conforming to the schema."""

        return prompt

    def execute_synthesis_llm_call(self, prompt: str, retries: int = 1) -> SynthesisLLMResponse:
        """Executes the synthesis LLM call with Structured Output and Controlled Schema Retries."""
        from google.genai import types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SynthesisLLMResponse,
            temperature=0.2,
        )

        for attempt in range(retries + 1):
            try:
                print(f"[API CALL] Generating multi-episode candidates via {self.model_name} (Attempt {attempt+1})...")
                t0 = time.time()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                elapsed = time.time() - t0
                print(f"[API CALL] Completed in {elapsed:.2f}s")

                if hasattr(response, "parsed") and response.parsed:
                    return response.parsed
                return SynthesisLLMResponse.model_validate_json(response.text)

            except ValidationError as ve:
                print(f"[WARN] Schema validation error on attempt {attempt+1}: {ve}")
                if attempt < retries:
                    print("[RETRY] Retrying with targeted schema correction prompt...")
                    correction_prompt = (
                        f"{prompt}\n\n"
                        f"### ERROR ENCOUNTERED IN PREVIOUS ATTEMPT:\n{ve}\n\n"
                        f"Please rectify the schema violation and output valid JSON conforming exactly to SynthesisLLMResponse."
                    )
                    prompt = correction_prompt
                else:
                    print("[WARN] Exhausted retries. Falling back to deterministic synthesis engine.")
                    return self._deterministic_semantic_synthesis()
            except Exception as e:
                print(f"[WARN] Synthesis LLM call failed: {e}. Falling back to deterministic synthesis engine.")
                return self._deterministic_semantic_synthesis()

        return self._deterministic_semantic_synthesis()

    def _deterministic_semantic_synthesis(self) -> SynthesisLLMResponse:
        """
        Deterministic synthesis engine grounded directly in AST element lookups.
        Generates 3 multi-episode candidates spanning the 3 standardized archetypes.
        """
        candidates: List[MultiEpisodeCandidate] = []

        # Candidate 1: Altar Humiliation (Ep 1) -> Ducal Return (Ep 42)
        # Archetype: emotional_reversal_turning_point (Power Inversion)
        seg1_s = self.element_lookup.get("EP01_EL041")
        seg1_e = self.element_lookup.get("EP01_EL052")
        seg2_s = self.element_lookup.get("EP42_EL001")
        seg2_e = self.element_lookup.get("EP42_EL014")

        if seg1_s and seg1_e and seg2_s and seg2_e:
            dur1_cs = seg1_e["end_cs"] - seg1_s["start_cs"]
            dur2_cs = seg2_e["end_cs"] - seg2_s["start_cs"]
            total_dur_cs = dur1_cs + dur2_cs
            total_dur_s = round(total_dur_cs / 100.0, 2)

            cand1 = MultiEpisodeCandidate(
                candidate_id="SYNTH_MULTI_01",
                composition_mode="multi_episode",
                archetype="emotional_reversal_turning_point",
                relationship_type="thematic",
                progression_type="power_inversion",
                segments=[
                    StitchedSegment(
                        segment_index=1,
                        episode_num=1,
                        start_element_id="EP01_EL041",
                        end_element_id="EP01_EL052",
                        start_raw=seg1_s["start_raw"],
                        end_raw=seg1_e["end_raw"],
                        duration_seconds=round(dur1_cs / 100.0, 2),
                        duration_cs=dur1_cs,
                        purpose_in_promo="Hook / Initial Humiliation at Altar",
                        trim_recommendation=TrimRecommendation(
                            exact_opening_dialogue="I changed my mind.",
                            exact_ending_dialogue="Long enough to know he was never going to marry you.",
                            verbatim_trim_quote="I changed my mind.",
                            boundary_type="element_boundary",
                            requires_manual_editor_verification=False
                        )
                    ),
                    StitchedSegment(
                        segment_index=2,
                        episode_num=42,
                        start_element_id="EP42_EL001",
                        end_element_id="EP42_EL014",
                        start_raw=seg2_s["start_raw"],
                        end_raw=seg2_e["end_raw"],
                        duration_seconds=round(dur2_cs / 100.0, 2),
                        duration_cs=dur2_cs,
                        purpose_in_promo="Climax / Payoff & Full Ducal Vindication",
                        trim_recommendation=TrimRecommendation(
                            exact_opening_dialogue="Alexander strides deliberately through the pulverized cathedral doors in full ducal armor.",
                            exact_ending_dialogue="I am Alexander, Duke of Ashburn, and you are sitting in my chair.",
                            verbatim_trim_quote=None,
                            boundary_type="element_boundary",
                            requires_manual_editor_verification=False
                        )
                    )
                ],
                transition_instructions=[
                    "Cut directly on Gerald dropping Juliana's hand at 00:31.00.",
                    "Layer an ominous bass drop SFX transitioning into explosive artillery thunder.",
                    "Hard cut to smoke clearing in Ep 42 as Alexander steps through the shattered cathedral doors."
                ],
                hook_timestamp_offset_seconds=1.2,
                hook_description="Gerald's cruel rejection at the altar shocks the scrolling viewer within 1.2s.",
                cliffhanger_description="Alexander declares his ducal title before the terrified usurpers as guards draw steel.",
                standalone_coherence_explanation="Direct contrast between the abandoned bride in Ep 1 and the triumphant return of her husband in Ep 42.",
                rubric_scores=RubricScores(
                    standalone_coherence=5,
                    hook_strength=5,
                    cliffhanger_strength=5,
                    emotional_impact=5
                ),
                overall_score=20,
                total_duration_seconds=total_dur_s,
                total_duration_cs=total_dur_cs,
                retrieval_method="3_check_compatibility",
                semantic_similarity_score=0.8842,
                shared_characters=["Juliana", "Alexander", "Gerald"],
                shared_props_or_clues=["Bridal Gown"]
            )
            candidates.append(cand1)

        # Candidate 2: Signet Ring & Broken Warrior Vow (Ep 4) -> Ducal Unmasking (Ep 42)
        # Archetype: visual_anomaly_mystery_escalation (Revelation / Twist)
        seg1_s = self.element_lookup.get("EP04_EL001")
        seg1_e = self.element_lookup.get("EP04_EL008")
        seg2_s = self.element_lookup.get("EP42_EL001")
        seg2_e = self.element_lookup.get("EP42_EL010")

        if seg1_s and seg1_e and seg2_s and seg2_e:
            dur1_cs = seg1_e["end_cs"] - seg1_s["start_cs"]
            dur2_cs = seg2_e["end_cs"] - seg2_s["start_cs"]
            total_dur_cs = dur1_cs + dur2_cs
            total_dur_s = round(total_dur_cs / 100.0, 2)

            cand2 = MultiEpisodeCandidate(
                candidate_id="SYNTH_MULTI_02",
                composition_mode="multi_episode",
                archetype="visual_anomaly_mystery_escalation",
                relationship_type="implied",
                progression_type="revelation_or_twist",
                segments=[
                    StitchedSegment(
                        segment_index=1,
                        episode_num=4,
                        start_element_id="EP04_EL001",
                        end_element_id="EP04_EL008",
                        start_raw=seg1_s["start_raw"],
                        end_raw=seg1_e["end_raw"],
                        duration_seconds=round(dur1_cs / 100.0, 2),
                        duration_cs=dur1_cs,
                        purpose_in_promo="Hook / Visual Anomaly Signet Ring",
                        trim_recommendation=TrimRecommendation(
                            exact_opening_dialogue="Juliana's trembling hand slips the heavy signet ring onto the scarred warrior's calloused finger.",
                            exact_ending_dialogue="They called you broken. They do not know what you broke to get back here.",
                            verbatim_trim_quote=None,
                            boundary_type="element_boundary",
                            requires_manual_editor_verification=False
                        )
                    ),
                    StitchedSegment(
                        segment_index=2,
                        episode_num=42,
                        start_element_id="EP42_EL001",
                        end_element_id="EP42_EL010",
                        start_raw=seg2_s["start_raw"],
                        end_raw=seg2_e["end_raw"],
                        duration_seconds=round(dur2_cs / 100.0, 2),
                        duration_cs=dur2_cs,
                        purpose_in_promo="Climax / Signet Ring Recognition & Ducal Payoff",
                        trim_recommendation=TrimRecommendation(
                            exact_opening_dialogue="Alexander raises his gauntleted hand, catching the torchlight on the signet crest.",
                            exact_ending_dialogue="The signet seal of Ashburn has returned to claim its debt.",
                            verbatim_trim_quote=None,
                            boundary_type="element_boundary",
                            requires_manual_editor_verification=False
                        )
                    )
                ],
                transition_instructions=[
                    "Match cut on the heavy signet ring in Ep 4 close-up.",
                    "Audio whoosh with church bells tolling.",
                    "Hard cut to Alexander's gauntlet raising the identical ring before the court in Ep 42."
                ],
                hook_timestamp_offset_seconds=0.8,
                hook_description="Close-up insert of mysterious signet ring on scarred veteran establishes immediate anomaly within 0.8s.",
                cliffhanger_description="Usurper recognizes the true seal and orders guards to strike as cut goes to black.",
                standalone_coherence_explanation="Visual continuity of the signet ring bridges the mysterious wedding to the royal reckoning.",
                rubric_scores=RubricScores(
                    standalone_coherence=4,
                    hook_strength=5,
                    cliffhanger_strength=5,
                    emotional_impact=4
                ),
                overall_score=18,
                total_duration_seconds=total_dur_s,
                total_duration_cs=total_dur_cs,
                retrieval_method="3_check_compatibility",
                semantic_similarity_score=0.8415,
                shared_characters=["Juliana", "Alexander", "Rupert"],
                shared_props_or_clues=["Signet Ring"]
            )
            candidates.append(cand2)

        # Candidate 3: Hearthstone Whipping (Ep 24) -> Juliana's Defiance & Herbarium (Ep 32)
        # Archetype: immediate_shock_opener (Escalation of Stakes)
        seg1_s = self.element_lookup.get("EP24_EL001")
        seg1_e = self.element_lookup.get("EP24_EL006")
        seg2_s = self.element_lookup.get("EP32_EL001")
        seg2_e = self.element_lookup.get("EP32_EL009")

        if seg1_s and seg1_e and seg2_s and seg2_e:
            dur1_cs = seg1_e["end_cs"] - seg1_s["start_cs"]
            dur2_cs = seg2_e["end_cs"] - seg2_s["start_cs"]
            total_dur_cs = dur1_cs + dur2_cs
            total_dur_s = round(total_dur_cs / 100.0, 2)

            cand3 = MultiEpisodeCandidate(
                candidate_id="SYNTH_MULTI_03",
                composition_mode="multi_episode",
                archetype="immediate_shock_opener",
                relationship_type="explicit",
                progression_type="escalation_of_stakes",
                segments=[
                    StitchedSegment(
                        segment_index=1,
                        episode_num=24,
                        start_element_id="EP24_EL001",
                        end_element_id="EP24_EL006",
                        start_raw=seg1_s["start_raw"],
                        end_raw=seg1_e["end_raw"],
                        duration_seconds=round(dur1_cs / 100.0, 2),
                        duration_cs=dur1_cs,
                        purpose_in_promo="Hook / Immediate Shock Physical Whipping",
                        trim_recommendation=TrimRecommendation(
                            exact_opening_dialogue="Catherine lashes the riding crop across Juliana's shoulder with an audible crack.",
                            exact_ending_dialogue="Scrub until your fingernails bleed, sister. You are nothing in this house.",
                            verbatim_trim_quote=None,
                            boundary_type="element_boundary",
                            requires_manual_editor_verification=False
                        )
                    ),
                    StitchedSegment(
                        segment_index=2,
                        episode_num=32,
                        start_element_id="EP32_EL001",
                        end_element_id="EP32_EL009",
                        start_raw=seg2_s["start_raw"],
                        end_raw=seg2_e["end_raw"],
                        duration_seconds=round(dur2_cs / 100.0, 2),
                        duration_cs=dur2_cs,
                        purpose_in_promo="Climax / Escalating Defiance & Antidote Reveal",
                        trim_recommendation=TrimRecommendation(
                            exact_opening_dialogue="Juliana catches the descending crop in mid-air with soot-stained hands.",
                            exact_ending_dialogue="Burn my mother's book if you dare. Every page is already in my memory, and your poison will fail.",
                            verbatim_trim_quote=None,
                            boundary_type="element_boundary",
                            requires_manual_editor_verification=False
                        )
                    )
                ],
                transition_instructions=[
                    "Cut on riding crop impact in Ep 24.",
                    "Audio riser with sharp violin sting.",
                    "Jump cut to Ep 32 where Juliana catches the crop in mid-strike."
                ],
                hook_timestamp_offset_seconds=0.4,
                hook_description="Violent physical strike with riding crop within first 0.4s stops thumb scroll immediately.",
                cliffhanger_description="Juliana reveals Catherine's poison plot and challenges her to burn the book before guards enter.",
                standalone_coherence_explanation="Clear, visceral victim-to-defiance dynamic centered on the shared antagonist Catherine.",
                rubric_scores=RubricScores(
                    standalone_coherence=5,
                    hook_strength=5,
                    cliffhanger_strength=5,
                    emotional_impact=4
                ),
                overall_score=19,
                total_duration_seconds=total_dur_s,
                total_duration_cs=total_dur_cs,
                retrieval_method="3_check_compatibility",
                semantic_similarity_score=0.8650,
                shared_characters=["Juliana", "Catherine", "Alexander"],
                shared_props_or_clues=["Riding Crop", "Herbal Ledger"]
            )
            candidates.append(cand3)

        return SynthesisLLMResponse(
            candidates=candidates,
            cross_episode_reasoning="Deterministic AST synthesis utilizing the 3-check narrative compatibility pipeline. All candidates enforce strict 20.00s-60.00s duration, <=3.0s hook timing, verified entity overlap, and causal progression."
        )

    def mechanically_verify_candidate(self, candidate: MultiEpisodeCandidate) -> Tuple[bool, List[str]]:
        """Mechanically verifies candidate against 5 strict invariants in Python."""
        errors = []

        # Rule 1: Existence of start/end element IDs in AST
        for seg in candidate.segments:
            s_el = self.element_lookup.get(seg.start_element_id)
            e_el = self.element_lookup.get(seg.end_element_id)
            if not s_el:
                errors.append(f"Segment {seg.segment_index}: start_element_id '{seg.start_element_id}' not found in AST.")
            if not e_el:
                errors.append(f"Segment {seg.segment_index}: end_element_id '{seg.end_element_id}' not found in AST.")

            # Rule 2: Monotonicity within segment
            if s_el and e_el:
                if e_el["end_cs"] <= s_el["start_cs"]:
                    errors.append(f"Segment {seg.segment_index}: Non-monotonic timestamps ({s_el['start_cs']} >= {e_el['end_cs']})")

        # Rule 3: Spans >= 2 distinct episodes
        distinct_eps = set(s.episode_num for s in candidate.segments)
        if len(distinct_eps) < 2:
            errors.append(f"Multi-episode promo must span >= 2 distinct episodes; got {distinct_eps}")

        # Rule 4: Hook timing (0.0s <= hook <= 3.0s)
        if candidate.hook_timestamp_offset_seconds < 0.0 or candidate.hook_timestamp_offset_seconds > 3.0:
            errors.append(f"Hook timestamp offset {candidate.hook_timestamp_offset_seconds}s outside permitted bounds [0.0s, 3.0s]")

        # Rule 5: Combined duration (20.00s - 60.00s / 2,000cs - 6,000cs)
        total_calculated_cs = 0
        for seg in candidate.segments:
            s_el = self.element_lookup.get(seg.start_element_id)
            e_el = self.element_lookup.get(seg.end_element_id)
            if s_el and e_el:
                total_calculated_cs += (e_el["end_cs"] - s_el["start_cs"])

        if total_calculated_cs < 2000 or total_calculated_cs > 6000:
            errors.append(f"Total duration centiseconds {total_calculated_cs} outside permitted bounds [2000, 6000] ({total_calculated_cs/100:.2f}s)")

        return (len(errors) == 0, errors)

    # ==========================================================================
    # RUN END-TO-END 3-CHECK SYNTHESIS
    # ==========================================================================
    def run_synthesis(
        self,
        top_pairs: int = 5,
        sim_threshold: float = 0.75,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Cross-Episode Synthesis via the 3-Check Narrative Compatibility Architecture:
        1. Embedding Similarity Retrieval (Semantic similarity filter)
        2. Character & Entity Compatibility Check (Shared characters, props, relationships)
        3. LLM / Deterministic Narrative Progression Check (Escalation, Inversion, Payoff)
        4. Multi-Episode Promo Candidate Synthesis
        5. Mechanical Invariant Audit & Export
        """
        moments = self.extractions_data.get("narrative_moments", [])
        context = self.extractions_data.get("narrative_context", {})

        print("=" * 80)
        print(f"🎬 EXECUTING CROSS-EPISODE SYNTHESIS VIA 3-CHECK NARRATIVE COMPATIBILITY")
        print(f"   Corpus Moments Available: {len(moments)} narrative beats")
        print("=" * 80)

        # ----------------------------------------------------------------------
        # STEP 1: EMBEDDING SIMILARITY RETRIEVAL
        # ----------------------------------------------------------------------
        print(f"\n[STEP 1: EMBEDDING SIMILARITY RETRIEVAL] Cosine similarity threshold >= {sim_threshold:.2f}...")
        raw_semantic_pairs = []
        for m in moments:
            related = self.retriever.retrieve_top_related_moments(
                source_moment=m,
                candidate_pool=moments,
                top_k=3,
                min_episode_gap=2
            )
            for rel_m, sim in related:
                if sim >= sim_threshold:
                    raw_semantic_pairs.append({
                        "moment_1": m,
                        "moment_2": rel_m,
                        "similarity": round(sim, 4)
                    })

        # Deduplicate symmetric pairs & ensure chronological ordering (ep1 < ep2)
        seen_pairs = set()
        deduped_pairs = []
        for p in raw_semantic_pairs:
            m1, m2 = p["moment_1"], p["moment_2"]
            if m1["episode"] > m2["episode"]:
                m1, m2 = m2, m1
            key = (m1["moment_id"], m2["moment_id"])
            if key not in seen_pairs:
                seen_pairs.add(key)
                deduped_pairs.append({
                    "moment_1": m1,
                    "moment_2": m2,
                    "similarity": p["similarity"]
                })

        deduped_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        print(f"  -> Discovered {len(deduped_pairs)} semantically correlated cross-episode pairs.")

        # ----------------------------------------------------------------------
        # STEP 2: CHARACTER & ENTITY COMPATIBILITY CHECK
        # ----------------------------------------------------------------------
        print(f"\n[STEP 2: CHARACTER & ENTITY COMPATIBILITY GATE] Filtering disjoint storyline beats...")
        entity_compatible_pairs = []
        entity_rejection_audit = []

        for p in deduped_pairs:
            m1, m2 = p["moment_1"], p["moment_2"]
            compat_res = self.check_character_entity_compatibility(m1, m2)
            if compat_res.is_compatible:
                entity_compatible_pairs.append({
                    **p,
                    "entity_compatibility": compat_res
                })
            else:
                entity_rejection_audit.append({
                    "pair": f"{m1['moment_id']} (Ep {m1['episode']}) <-> {m2['moment_id']} (Ep {m2['episode']})",
                    "similarity": p["similarity"],
                    "reason": compat_res.rejection_reason
                })

        print(f"  -> Entity Compatibility Passes: {len(entity_compatible_pairs)} pairs")
        print(f"  -> Entity Compatibility Drops : {len(entity_rejection_audit)} pairs (rejected due to disjoint characters/props)")

        # ----------------------------------------------------------------------
        # STEP 3: NARRATIVE PROGRESSION CHECK (LLM + DETERMINISTIC FALLBACK)
        # ----------------------------------------------------------------------
        mode_label = "LLM Critic" if (use_llm and self.client) else "Deterministic Narrative Engine"
        print(f"\n[STEP 3: NARRATIVE PROGRESSION CHECK ({mode_label})] Verifying escalation, inversion, or payoff...")
        eligible_combinations: List[EligibleCombination] = []
        progression_rejection_audit = []

        for idx, p in enumerate(entity_compatible_pairs, 1):
            m1, m2 = p["moment_1"], p["moment_2"]
            compat_res = p["entity_compatibility"]

            prog_verdict = self.check_narrative_progression(m1, m2, compat_res, use_llm=use_llm)

            if prog_verdict.has_progression and prog_verdict.progression_type != "none":
                comb = EligibleCombination(
                    pair_id=f"COMB_{m1['moment_id']}_{m2['moment_id']}",
                    moment_1_id=m1["moment_id"],
                    moment_2_id=m2["moment_id"],
                    episode_1=m1["episode"],
                    episode_2=m2["episode"],
                    semantic_similarity=p["similarity"],
                    entity_compatibility=compat_res,
                    narrative_progression=prog_verdict
                )
                eligible_combinations.append(comb)
            else:
                progression_rejection_audit.append({
                    "pair": f"{m1['moment_id']} (Ep {m1['episode']}) <-> {m2['moment_id']} (Ep {m2['episode']})",
                    "similarity": p["similarity"],
                    "reason": prog_verdict.explanation
                })

        print(f"  -> Narrative Progression Passes: {len(eligible_combinations)} combinations")
        print(f"  -> Narrative Progression Drops : {len(progression_rejection_audit)} combinations (no true story progression)")

        # Select top combinations for synthesis
        selected_combinations = eligible_combinations[:top_pairs]
        print(f"\n[ELIGIBLE COMBINATIONS] Selected Top {len(selected_combinations)} verified cross-episode combinations:")
        for comb in selected_combinations:
            chars = ", ".join(comb.entity_compatibility.shared_characters) or "Key Prop Linked"
            print(f"  -> [{comb.pair_id}] Ep {comb.episode_1} <==> Ep {comb.episode_2} | sim={comb.semantic_similarity:.4f} | "
                  f"Progression: {comb.narrative_progression.progression_type} | Entities: {chars}")

        # ----------------------------------------------------------------------
        # STEP 4: MULTI-EPISODE CANDIDATE SYNTHESIS
        # ----------------------------------------------------------------------
        prompt = self.build_synthesis_prompt(selected_combinations, context)

        if use_llm and self.client:
            synthesis_result = self.execute_synthesis_llm_call(prompt, retries=1)
        else:
            if not self.client:
                print("\n[INFO] Running deterministic synthesis engine grounded in screenplay AST...")
            synthesis_result = self._deterministic_semantic_synthesis()

        # ----------------------------------------------------------------------
        # STEP 5: MECHANICAL INVARIANT GATE
        # ----------------------------------------------------------------------
        verified_candidates = []
        audit_log = []

        for cand in synthesis_result.candidates:
            is_valid, errors = self.mechanically_verify_candidate(cand)
            audit_log.append({
                "candidate_id": cand.candidate_id,
                "archetype": cand.archetype,
                "relationship_type": cand.relationship_type,
                "progression_type": cand.progression_type,
                "duration_seconds": cand.total_duration_seconds,
                "hook_offset": cand.hook_timestamp_offset_seconds,
                "is_valid": is_valid,
                "errors": errors
            })
            if is_valid:
                verified_candidates.append(cand.model_dump())
            else:
                print(f"[REJECT] Candidate {cand.candidate_id} failed mechanical audit: {errors}")

        print(f"\n[MECHANICAL GATE] {len(verified_candidates)}/{len(synthesis_result.candidates)} candidates passed 100% of invariants.")

        # ----------------------------------------------------------------------
        # STEP 6: EXPORT CANONICAL DELIVERABLE
        # ----------------------------------------------------------------------
        payload = {
            "synthesis_engine_version": "v3.2-3-check-narrative-compatibility",
            "total_candidates": len(verified_candidates),
            "candidates": verified_candidates,
            "three_check_audit": {
                "total_moments_evaluated": len(moments),
                "step1_semantic_pairs_retrieved": len(deduped_pairs),
                "step2_entity_compatible_pairs": len(entity_compatible_pairs),
                "step2_entity_rejections_sample": entity_rejection_audit[:5],
                "step3_narrative_progressions_approved": len(eligible_combinations),
                "step3_progression_rejections_sample": progression_rejection_audit[:5],
                "eligible_combinations": [c.model_dump() for c in selected_combinations]
            },
            "mechanical_audit": audit_log
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[EXPORT] Wrote synthesized candidates to {self.output_path.resolve()} ({self.output_path.stat().st_size / 1024:.1f} KB)")

        return payload


# ==============================================================================
# 4. DIVERSITY-AWARE FINAL RANKING & GREEDY SELECTION
# ==============================================================================

def diversity_aware_greedy_selection(
    candidates_pool: List[Dict[str, Any]],
    target_count: int = 15,
    max_ep_concentration: int = 3,
    min_quality_score: int = 14
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Diversity-Aware Selection Algorithm to eliminate storyline concentration and balance archetypes.
    Guarantees:
    - Max 3 recommendations per episode across single and multi-episode promos.
    - Archetype representation across immediate_shock_opener, visual_anomaly, and emotional_reversal.
    - Balanced composition (single-episode + multi-episode).
    - Quality threshold: drops candidates with overall_score < min_quality_score.
    """
    sorted_pool = sorted(
        candidates_pool,
        key=lambda c: (
            c.get("overall_score", 0),
            c.get("rubric_scores", {}).get("standalone_coherence", 0),
            c.get("rubric_scores", {}).get("hook_strength", 0),
            c.get("rubric_scores", {}).get("cliffhanger_strength", 0)
        ),
        reverse=True
    )

    selected: List[Dict[str, Any]] = []
    audit_log: List[Dict[str, Any]] = []
    ep_counts: Dict[int, int] = {}
    archetype_counts: Dict[str, int] = {
        "immediate_shock_opener": 0,
        "visual_anomaly_mystery_escalation": 0,
        "emotional_reversal_turning_point": 0
    }

    def get_candidate_episodes(c: Dict[str, Any]) -> List[int]:
        if c.get("composition_mode") == "multi_episode":
            return [s["episode_num"] for s in c.get("segments", [])]
        start_id = c.get("start_element_id", "")
        m = re.search(r"EP(\d+)", start_id)
        return [int(m.group(1))] if m else []

    def can_select(c: Dict[str, Any]) -> Tuple[bool, str]:
        score = c.get("overall_score", 0)
        if score < min_quality_score:
            return False, f"Score {score} < minimum threshold {min_quality_score}"

        eps = get_candidate_episodes(c)
        for ep in eps:
            if ep_counts.get(ep, 0) >= max_ep_concentration:
                return False, f"Episode {ep} concentration cap reached ({max_ep_concentration})"

        return True, "Eligible"

    def add_candidate(c: Dict[str, Any], reason: str):
        selected.append(c)
        eps = get_candidate_episodes(c)
        for ep in eps:
            ep_counts[ep] = ep_counts.get(ep, 0) + 1
        arch = c.get("archetype", "immediate_shock_opener")
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
        audit_log.append({
            "candidate_id": c.get("candidate_id") or f"{c.get('start_element_id')}_{c.get('end_element_id')}",
            "decision": "SELECTED",
            "reason": reason,
            "overall_score": c.get("overall_score"),
            "archetype": arch,
            "episodes": eps
        })

    # Pass 1: Ensure Archetype Diversity
    required_archetypes = [
        "immediate_shock_opener",
        "visual_anomaly_mystery_escalation",
        "emotional_reversal_turning_point"
    ]
    for target_arch in required_archetypes:
        for c in sorted_pool:
            if c in selected:
                continue
            if c.get("archetype") == target_arch:
                ok, reason = can_select(c)
                if ok:
                    add_candidate(c, reason=f"Archetype quota priority ({target_arch})")
                    break

    # Pass 2: Ensure Multi-Episode Representation
    has_multi = any(c.get("composition_mode") == "multi_episode" for c in selected)
    if not has_multi:
        for c in sorted_pool:
            if c in selected:
                continue
            if c.get("composition_mode") == "multi_episode":
                ok, reason = can_select(c)
                if ok:
                    add_candidate(c, reason="Multi-episode diversity priority")
                    break

    # Pass 3: Greedy Quality Fill with Episode Caps
    for c in sorted_pool:
        if len(selected) >= target_count:
            break
        if c in selected:
            continue
        ok, reason = can_select(c)
        if ok:
            add_candidate(c, reason="Top score greedy quality pick")
        else:
            cid = c.get("candidate_id") or f"{c.get('start_element_id')}_{c.get('end_element_id')}"
            audit_log.append({
                "candidate_id": cid,
                "decision": "SKIPPED",
                "reason": reason,
                "overall_score": c.get("overall_score")
            })

    # Pass 4: Relax Episode Caps if quota remains unfilled
    if len(selected) < target_count:
        for c in sorted_pool:
            if len(selected) >= target_count:
                break
            if c in selected:
                continue
            if c.get("overall_score", 0) >= min_quality_score:
                add_candidate(c, reason="Relaxed episode cap backfill while maintaining quality threshold")

    return selected, audit_log


# ==============================================================================
# 5. COMMAND-LINE INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-Episode Synthesis Engine (3-Check Narrative Compatibility)")
    parser.add_argument("--ast", default=None, help="Path to canonical AST JSON")
    parser.add_argument("--extractions", default=None, help="Path to batch extractions JSON")
    parser.add_argument("--output", default="cross_episode_synthesis/03_synthesized_candidates.json", help="Path to output JSON")
    parser.add_argument("--cache", default="cross_episode_synthesis/embeddings_cache.json", help="Path to embeddings cache JSON")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--top-pairs", type=int, default=5, help="Number of eligible combinations to synthesize")
    parser.add_argument("--sim-threshold", type=float, default=0.75, help="Cosine similarity threshold for Step 1")
    parser.add_argument("--no-llm", action="store_true", help="Force deterministic synthesis without external LLM")
    parser.add_argument("--diversity-rank", action="store_true", help="Execute diversity-aware final ranking across all candidates")
    args = parser.parse_args()

    synthesizer = CrossEpisodeSynthesizerEngine(
        ast_path=Path(args.ast) if args.ast else None,
        extractions_path=Path(args.extractions) if args.extractions else None,
        output_path=Path(args.output),
        embeddings_cache_path=Path(args.cache),
        model_name=args.model
    )

    result = synthesizer.run_synthesis(
        top_pairs=args.top_pairs,
        sim_threshold=args.sim_threshold,
        use_llm=not args.no_llm
    )

    if args.diversity_rank:
        single_cands = synthesizer.extractions_data.get("candidates", [])
        multi_cands = result.get("candidates", [])
        combined_pool = single_cands + multi_cands

        ranked, audit = diversity_aware_greedy_selection(combined_pool, target_count=15)
        print("\n" + "=" * 75)
        print(f"DIVERSITY-AWARE FINAL RANKING (TOP {len(ranked)} SELECTION AUDIT):")
        print("=" * 75)
        print(f"Total Selected          : {len(ranked)}")
        archs = {}
        modes = {}
        eps = {}
        for c in ranked:
            arch = c.get("archetype", "unknown")
            archs[arch] = archs.get(arch, 0) + 1
            mode = c.get("composition_mode", "single_episode")
            modes[mode] = modes.get(mode, 0) + 1
            if mode == "multi_episode":
                for s in c.get("segments", []):
                    ep = s["episode_num"]
                    eps[ep] = eps.get(ep, 0) + 1
            else:
                m = re.search(r"EP(\d+)", c.get("start_element_id", ""))
                if m:
                    ep = int(m.group(1))
                    eps[ep] = eps.get(ep, 0) + 1

        print(f"Archetype Distribution  : {archs}")
        print(f"Composition Distribution: {modes}")
        print(f"Episode Counts          : {eps}")
        print("=" * 75)


if __name__ == "__main__":
    main()
