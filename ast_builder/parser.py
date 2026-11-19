"""
Screenplay Parser & Abstract Syntax Tree (AST) Generator (ast_builder/parser.py)
Production-grade screenplay ingestion engine for microdrama scripts.
Parses raw screenplay text files into a strongly-typed, verifiable JSON AST
with integer centisecond timestamps (1s = 100cs) and element categorization.
"""

import os
import sys
import re
import json
import argparse
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any, Union
from collections import Counter


# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class ElementType(str, Enum):
    CAMERA = "CAMERA"
    ACTION = "ACTION"
    SOUND_FX = "SOUND FX"
    DIALOGUE = "DIALOGUE"

    @classmethod
    def _missing_(cls, value: object):
        """
        Ensures case-insensitive lookup and format normalization (e.g. 'dialogue', 'SOUND_FX').
        Eliminates ValueError when strings are instantiated without prior .upper() normalization.
        """
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in ("SOUND_FX", "SOUND-FX"):
                normalized = "SOUND FX"
            for member in cls:
                if member.value == normalized:
                    return member
        return super()._missing_(value)


@dataclass(slots=True, frozen=True)
class DialogueMetadata:
    """Strongly-typed metadata for dialogue elements."""
    speaker: str
    modifier: Optional[str] = None   # e.g., 'O.S.', 'V.O.', 'Off-screen'
    delivery: Optional[str] = None   # e.g., 'Cold, Commanding', 'Grave, Alert'
    raw_tag: str = ""

    def __post_init__(self):
        if not self.speaker.strip():
            object.__setattr__(self, "speaker", "SPEAKER")


@dataclass(slots=True, frozen=True)
class ScreenplayElement:
    """Atomic screenplay unit with exact integer centisecond timing."""
    element_id: str                      # Format: EP{ep:02d}_EL{el:03d}
    episode_num: int
    scene_num: int
    element_type: ElementType
    start_raw: str                       # e.g., '00:11.30'
    end_raw: str                         # e.g., '00:13.10'
    start_cs: int                        # integer centiseconds (1s = 100cs)
    end_cs: int
    duration_cs: int                     # end_cs - start_cs
    duration_seconds: float              # duration_cs / 100.0
    content: str                         # verbatim dialogue or action text
    dialogue_meta: Optional[DialogueMetadata] = None
    raw_tag: str = ""

    def __post_init__(self):
        if isinstance(self.element_type, str) and not isinstance(self.element_type, ElementType):
            object.__setattr__(self, "element_type", ElementType(self.element_type))
        if not isinstance(self.element_type, ElementType):
            raise TypeError(f"Invalid element_type: {self.element_type}")
        if self.start_cs < 0:
            raise ValueError(f"start_cs must be non-negative: {self.start_cs}")
        if self.end_cs <= self.start_cs:
            raise ValueError(f"end_cs ({self.end_cs}) must be strictly greater than start_cs ({self.start_cs})")
        if self.duration_cs != (self.end_cs - self.start_cs):
            raise ValueError("duration_cs does not match (end_cs - start_cs)")


@dataclass(slots=True, frozen=True)
class Scene:
    """Scene boundary containing chronological screenplay elements."""
    episode_num: int
    scene_num: int
    heading: str
    start_raw: str
    end_raw: str
    start_cs: int
    end_cs: int
    duration_cs: int
    duration_seconds: float
    elements: Tuple[ScreenplayElement, ...]

    def __post_init__(self):
        if self.end_cs <= self.start_cs:
            raise ValueError(f"Scene end_cs ({self.end_cs}) must exceed start_cs ({self.start_cs})")


@dataclass(slots=True, frozen=True)
class EpisodeHeader:
    """Metadata header block for an episode."""
    episode_num: int
    raw_episode_title: str
    logline: str
    continuity_notes: str


@dataclass(slots=True, frozen=True)
class Episode:
    """Structured episode containing header, scenes, and ordered elements."""
    header: EpisodeHeader
    scenes: Tuple[Scene, ...]
    elements: Tuple[ScreenplayElement, ...]
    total_elements: int
    total_duration_cs: int

    @property
    def episode_num(self) -> int:
        return self.header.episode_num

    @property
    def total_duration_seconds(self) -> float:
        return self.total_duration_cs / 100.0


@dataclass(slots=True, frozen=True)
class ValidationAudit:
    """Validation audit record verifying mechanical integrity."""
    episode_num: int
    filename: str
    scene_count: int
    camera_count: int
    action_count: int
    sound_fx_count: int
    dialogue_count: int
    total_elements: int
    timestamp_errors: int
    containment_errors: int
    is_valid: bool
    errors: Tuple[str, ...] = ()


def parse_timestamp_to_cs(ts_str: str) -> int:
    """
    Converts MM:SS.ss or HH:MM:SS.ss string to integer centiseconds.
    Handles optional parentheses, 1-3 decimal digits, and variable whitespace.
    """
    clean = ts_str.strip().strip("()")
    match = re.search(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?", clean)
    if not match:
        raise ValueError(f"Invalid timestamp format: '{ts_str}'")

    h = int(match.group(1)) if match.group(1) else 0
    m = int(match.group(2))
    s = int(match.group(3))
    cs_raw = match.group(4) if match.group(4) else "00"

    if len(cs_raw) == 1:
        cs = int(cs_raw) * 10
    elif len(cs_raw) == 2:
        cs = int(cs_raw)
    else:  # 3 digits (milliseconds) -> round to centiseconds
        cs = round(int(cs_raw) / 10.0)

    return (h * 360000) + (m * 6000) + (s * 100) + cs


def format_cs_to_timestamp(cs: int) -> str:
    """Converts integer centiseconds back into MM:SS.ss string format."""
    total_sec = cs // 100
    centis = cs % 100
    m = total_sec // 60
    s = total_sec % 60
    return f"{m:02d}:{s:02d}.{centis:02d}"


class UniversalScreenplayParser:
    """
    Universal screenplay ingestion engine supporting:
    1. Microdrama series scripts with explicit timestamp tags (**TAG:**\nStart: ...|End: ...)
    2. Full Hollywood / spec screenplay formats (e.g. The Dark Knight, 10 Things I Hate About You, The Cellar)
       with complex scene headings (INT/EXT or implicit), dialogue blocks, parentheticals,
       extensions, camera directions, transitions, and sound effects.
    """

    # Screenplay dialect grammar patterns
    TIMES_OF_DAY = {
        "DAY", "NIGHT", "DAWN", "DUSK", "EVENING", "MORNING",
        "AFTERNOON", "CONTINUOUS", "LATER", "MOMENTS LATER",
        "MINUTES LATER", "SAME TIME", "SUNSET", "SUNRISE"
    }

    SLUG_PREFIX_RE = re.compile(
        r"^(?:(INT\.?\s*/\s*EXT\.?|EXT\.?\s*/\s*INT\.?|I/E\.?|INT\.?|EXT\.?))(?:\b|\s+)",
        re.IGNORECASE
    )

    CAMERA_CUES = (
        "CLOSE ON", "CLOSE UP ON", "CLOSE UP", "PULL BACK", "WIDE SHOT", "ANGLE ON",
        "ANOTHER ANGLE", "BACK TO", "POV", "POV SHOT", "FRONT ANGLE",
        "SERIES OF SHOTS", "INSERT", "INSERT CUT", "ON SCREEN",
        "CUTTING ", "PANNING ", "TRACKING ", "TILT "
    )

    SECONDARY_SLUG_KEYWORDS = {
        "ROOM", "BEDROOM", "BATHROOM", "KITCHEN", "MAIN ROOM", "STAIRWELL", "STAIRWAY",
        "HALLWAY", "BASEMENT", "CELLAR", "ATTIC", "PORCH", "GARAGE", "COURTYARD",
        "CLASSROOM", "LIVING ROOM", "CORRIDOR", "INSIDE", "OUTSIDE", "NEARBY",
        "CRAWL SPACE"
    }

    SECONDARY_SLUG_PREFIXES = (
        "ACROSS ", "AT THE ", "IN THE ", "NEAR ", "ON "
    )

    TRANSITION_RE = re.compile(
        r"^(?:(?:SMASH\s+)?CUT\s+TO|FADE\s+(?:IN|OUT|TO)|DISSOLVE\s+TO|"
        r"SMASH\s+TO\s+BLACK|CUT\s+TO\s+BLACK|ROLL\s+CREDITS|END|THE\s+END|"
        r"MATCH\s+CUT\s+TO|FLASH\s+CUT\s+TO)[\s\:\.\-]*$",
        re.IGNORECASE
    )

    PARENTHETICAL_RE = re.compile(r"^\s*\((.*?)\)\s*$")

    EXTENSION_RE = re.compile(
        r"\(\s*(V\.?O\.?|O\.?S\.?|0\.?S\.?|CONT'?D|CONTINUING|UNCONFIRMED|SUBTITLED|WHISPERING|SHOUTING|O\.?C\.?)\s*\)",
        re.IGNORECASE
    )

    PAGE_NUM_RE = re.compile(r"^\s*(?:=== PAGE \d+ ===|\d{1,3}\.?)\s*$")

    LIGHTING_ATMOSPHERE_CUES = {
        "BLACK", "DARKNESS", "SILENCE", "WHITE", "LIGHT", "SUPER", "TITLE", "FADE"
    }

    def __init__(self):
        self.ts_pattern = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?")

        # Element block pattern matching **TAG:**\nStart: (..)|End: (..)\nContent
        self.element_pattern = re.compile(
            r"^\*\*(?P<tag>.+?):\*\*\s*\n"
            r"Start:\s*\((?P<start>[^)]+)\)\s*\|\s*End:\s*\((?P<end>[^)]+)\)\s*\n"
            r"(?P<content>.*?)(?=(?:\n\*\*.+?:\*\*\s*\nStart:|\n(?:###\s*)?SCENE\s+\d+|\Z))",
            re.MULTILINE | re.DOTALL | re.IGNORECASE
        )

        # Scene block pattern
        self.scene_pattern = re.compile(
            r"^(?:###\s*)?(?:SCENE\s+(?P<num>\d+))\s*\n"
            r"(?:Scene heading:\s*(?P<heading>.+?)\s*\n)?"
            r"(?:Scene (?:start|window):\s*(?:\()?(?P<start>[0-9]{1,2}:[0-9]{2}(?:\.[0-9]{1,3})?)(?:\))?\s*\n)?"
            r"(?:Scene end:\s*(?:\()?(?P<end>[0-9]{1,2}:[0-9]{2}(?:\.[0-9]{1,3})?)(?:\))?\s*\n)?"
            r"(?:Actions and dialogue:\s*)?",
            re.MULTILINE | re.IGNORECASE
        )

        self.character_registry: set[str] = set()
        self.base_action_indent: int = 0
        self.has_indent_layout: bool = False
        self.action_max_indent: int = 0
        self.dialogue_min_indent: int = 0
        self.char_cue_min_indent: int = 0
        self.transition_min_indent: int = 45

    def time_to_centiseconds(self, ts_str: str) -> int:
        return parse_timestamp_to_cs(ts_str)

    def parse_header(self, text: str, fallback_ep_num: int) -> Tuple[EpisodeHeader, str]:
        """Extracts EPISODE NUMBER, LOGLINE, and CONTINUITY NOTES."""
        ep_num = fallback_ep_num
        num_m = re.search(r"^EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", text, re.MULTILINE | re.IGNORECASE)
        if num_m:
            ep_num = int(num_m.group(1))

        title_m = re.search(r"^EPISODE NUMBER:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
        raw_title = title_m.group(1).strip() if title_m else f"Episode {ep_num}"

        logline = ""
        log_m = re.search(r"^(?:EPISODE\s+)?LOGLINE:\s*(.+?)(?=\n[A-Z\s]+:|\n---|\Z)", text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if log_m:
            logline = " ".join(log_m.group(1).split())

        continuity = ""
        con_m = re.search(r"^(?:CONTINUITY\s+)?NOTES:\s*(.+?)(?=\n---|\nSCENE|\Z)", text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if con_m:
            continuity = " ".join(con_m.group(1).split())

        body = text
        split_marker = "--- SCRIPT START ---"
        body_idx = text.find(split_marker)
        if body_idx != -1:
            body = text[body_idx + len(split_marker):]

        header = EpisodeHeader(
            episode_num=ep_num,
            raw_episode_title=raw_title,
            logline=logline,
            continuity_notes=continuity
        )
        return header, body

    def decompose_tag(self, raw_tag_str: str) -> Tuple[ElementType, Optional[DialogueMetadata]]:
        """Decomposes raw tag into ElementType and structured DialogueMetadata."""
        tag = raw_tag_str.strip()
        tag_upper = tag.upper()

        if tag_upper == "CAMERA":
            return ElementType.CAMERA, None
        if tag_upper == "ACTION":
            return ElementType.ACTION, None
        if any(s in tag_upper for s in ["SOUND FX", "SOUND_FX", "SFX", "AUDIO"]):
            return ElementType.SOUND_FX, None

        # Dialogue: extract speaker, modifier, and delivery tone
        parens = re.findall(r"\((.*?)\)", tag)
        first_paren_idx = tag.find("(")
        speaker = tag[:first_paren_idx].strip() if first_paren_idx != -1 else tag.strip()

        modifier = None
        delivery = None
        if len(parens) == 1:
            p = parens[0].strip()
            if any(m in p.upper() for m in ["O.S.", "V.O.", "OFF-SCREEN", "OFF SCREEN", "VOICEOVER"]):
                modifier = p
            else:
                delivery = p
        elif len(parens) >= 2:
            modifier = parens[0].strip()
            delivery = " | ".join(p.strip() for p in parens[1:])

        meta = DialogueMetadata(
            speaker=speaker or "SPEAKER",
            modifier=modifier,
            delivery=delivery,
            raw_tag=tag
        )
        return ElementType.DIALOGUE, meta

    def parse_screenplay_file(
        self, 
        file_path: Path, 
        episode_idx: int = 1,
        target_pages_per_part: int = 20,
        threshold_pages: int = 25,
        auto_split_long_pdf: bool = True
    ) -> Union[Episode, List[Episode]]:
        """Parses a single screenplay file (microdrama or standard Hollywood format, .txt or .pdf) into Episode AST(s).
        If the file is a long PDF (> threshold_pages), automatically partitions it into balanced Episode parts.
        """
        is_pdf = file_path.suffix.lower() == ".pdf"
        total_pages = 1

        if is_pdf:
            import fitz
            doc = fitz.open(str(file_path))
            total_pages = len(doc)
            pages = []
            for pno, page in enumerate(doc, start=1):
                pages.append(f"\n=== PAGE {pno} ===\n" + page.get_text())
            doc.close()
            text = "\n".join(pages)
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        if self.element_pattern.search(text):
            return self._parse_microdrama_screenplay(text, episode_idx)
        else:
            single_ep = self._parse_standard_screenplay(text, episode_idx, file_name=file_path.name)
            if is_pdf and auto_split_long_pdf and total_pages > threshold_pages and len(single_ep.scenes) > 1:
                norm_lines = self._normalize_screenplay_text(text)
                _, meta_end_idx = self._extract_screenplay_metadata(norm_lines)
                lines_to_proc = norm_lines[meta_end_idx:]
                scene_start_pages = [
                    pno for raw, _, pno in lines_to_proc
                    if self.SLUG_PREFIX_RE.match(raw.strip()) or self._is_implicit_slugline(raw.strip())
                ]
                return partition_episode_into_parts(
                    ep=single_ep,
                    scene_start_pages=scene_start_pages,
                    start_episode_idx=episode_idx,
                    target_pages_per_part=target_pages_per_part
                )
            return single_ep

    def parse_screenplay(
        self, 
        text_or_path: Union[str, Path], 
        episode_idx: int = 1,
        target_pages_per_part: int = 20,
        threshold_pages: int = 25,
        auto_split_long_pdf: bool = True
    ) -> Union[Episode, List[Episode]]:
        """Convenience method to parse either raw text or a file path into an Episode AST (or List[Episode])."""
        if isinstance(text_or_path, Path):
            return self.parse_screenplay_file(
                text_or_path, 
                episode_idx=episode_idx,
                target_pages_per_part=target_pages_per_part,
                threshold_pages=threshold_pages,
                auto_split_long_pdf=auto_split_long_pdf
            )
        p = Path(text_or_path)
        if p.exists() and p.is_file():
            return self.parse_screenplay_file(
                p, 
                episode_idx=episode_idx,
                target_pages_per_part=target_pages_per_part,
                threshold_pages=threshold_pages,
                auto_split_long_pdf=auto_split_long_pdf
            )
        if self.element_pattern.search(text_or_path):
            return self._parse_microdrama_screenplay(text_or_path, episode_idx)
        else:
            return self._parse_standard_screenplay(text_or_path, episode_idx)

    def _parse_microdrama_screenplay(self, text: str, episode_idx: int) -> Episode:
        """Parses a timestamped microdrama screenplay into a typed Episode object."""
        header, body_text = self.parse_header(text, fallback_ep_num=episode_idx)
        ep_num = header.episode_num

        # Locate Scenes
        scene_matches = list(self.scene_pattern.finditer(body_text))
        if not scene_matches:
            # Fallback single scene for monolithic scripts without SCENE headers
            first_m = self.element_pattern.search(body_text)
            start_raw = first_m.group("start") if first_m else "00:00.00"
            scene_spans = [(0, len(body_text), 1, "INT./EXT. DEFAULT SCENE", start_raw, "00:00.00")]
        else:
            scene_spans = []
            for i, sm in enumerate(scene_matches):
                s_pos = sm.end()
                e_pos = scene_matches[i + 1].start() if i + 1 < len(scene_matches) else len(body_text)
                sc_num = int(sm.group("num"))
                raw_hd = (sm.group("heading") or "").strip()
                heading = re.sub(r"^Scene heading:\s*", "", raw_hd, flags=re.IGNORECASE).strip() or f"SCENE {sc_num}"
                sc_start_raw = sm.group("start")
                sc_end_raw = sm.group("end")
                scene_spans.append((s_pos, e_pos, sc_num, heading, sc_start_raw, sc_end_raw))

        parsed_scenes: List[Scene] = []
        all_elements: List[ScreenplayElement] = []
        element_counter = 0

        for s_pos, e_pos, sc_num, heading, sc_start_raw, sc_end_raw in scene_spans:
            scene_body = body_text[s_pos:e_pos]
            scene_elements: List[ScreenplayElement] = []

            for em in self.element_pattern.finditer(scene_body):
                element_counter += 1
                el_id = f"EP{ep_num:02d}_EL{element_counter:03d}"
                raw_tag = em.group("tag").strip()
                start_raw = em.group("start").strip()
                end_raw = em.group("end").strip()
                content = em.group("content").strip()

                start_cs = parse_timestamp_to_cs(start_raw)
                end_cs = parse_timestamp_to_cs(end_raw)
                duration_cs = end_cs - start_cs

                el_type, dlg_meta = self.decompose_tag(raw_tag)

                elem = ScreenplayElement(
                    element_id=el_id,
                    episode_num=ep_num,
                    scene_num=sc_num,
                    element_type=el_type,
                    start_raw=start_raw,
                    end_raw=end_raw,
                    start_cs=start_cs,
                    end_cs=end_cs,
                    duration_cs=duration_cs,
                    duration_seconds=duration_cs / 100.0,
                    content=content,
                    dialogue_meta=dlg_meta,
                    raw_tag=raw_tag
                )
                scene_elements.append(elem)
                all_elements.append(elem)

            if not scene_elements:
                continue

            # Determine scene boundaries
            sc_start_cs = parse_timestamp_to_cs(sc_start_raw) if sc_start_raw else scene_elements[0].start_cs
            sc_end_cs = parse_timestamp_to_cs(sc_end_raw) if sc_end_raw else scene_elements[-1].end_cs

            if sc_end_cs <= sc_start_cs and scene_elements:
                sc_end_cs = scene_elements[-1].end_cs
                sc_end_raw = scene_elements[-1].end_raw

            duration_cs = sc_end_cs - sc_start_cs
            scene = Scene(
                episode_num=ep_num,
                scene_num=sc_num,
                heading=heading,
                start_raw=sc_start_raw or scene_elements[0].start_raw,
                end_raw=sc_end_raw or scene_elements[-1].end_raw,
                start_cs=sc_start_cs,
                end_cs=sc_end_cs,
                duration_cs=duration_cs,
                duration_seconds=duration_cs / 100.0,
                elements=tuple(scene_elements)
            )
            parsed_scenes.append(scene)

        total_dur_cs = sum(s.duration_cs for s in parsed_scenes) if parsed_scenes else (all_elements[-1].end_cs if all_elements else 0)

        return Episode(
            header=header,
            scenes=tuple(parsed_scenes),
            elements=tuple(all_elements),
            total_elements=len(all_elements),
            total_duration_cs=total_dur_cs
        )

    def _normalize_screenplay_text(self, text: str) -> List[Tuple[str, int, int]]:
        lines = []
        page_counter = 1
        raw_lines = text.splitlines()

        for idx, line in enumerate(raw_lines, start=1):
            if "\x0c" in line:
                page_counter += 1
                line = line.replace("\x0c", "")

            norm_line = line.replace("’", "'").replace("‘", "'")
            norm_line = norm_line.replace("“", '"').replace("”", '"')
            norm_line = norm_line.replace("—", "--").replace("–", "--")

            if self.PAGE_NUM_RE.match(norm_line):
                m = re.search(r"\d+", norm_line)
                if m:
                    page_counter = int(m.group(0))
                continue

            lines.append((norm_line, idx, page_counter))

        # Stitch multi-line parentheticals
        merged_lines = []
        i = 0
        total_l = len(lines)
        while i < total_l:
            line_str, orig_idx, p_num = lines[i]
            s_str = line_str.strip()
            if s_str.startswith("(") and ")" not in s_str:
                stitched = line_str
                j = i + 1
                matched = False
                while j < min(i + 4, total_l):
                    next_str = lines[j][0].strip()
                    stitched += " " + next_str
                    if ")" in next_str:
                        matched = True
                        j += 1
                        break
                    j += 1
                if matched:
                    merged_lines.append((stitched, orig_idx, p_num))
                    i = j
                else:
                    merged_lines.append((line_str, orig_idx, p_num))
                    i += 1
            else:
                merged_lines.append((line_str, orig_idx, p_num))
                i += 1

        return merged_lines

    def _extract_screenplay_metadata(self, lines: List[Tuple[str, int, int]]) -> Tuple[EpisodeHeader, int]:
        author_cues = re.compile(r"^\s*(?:written\s+by|screenplay\s+by|story\s+by|by)\s*$", re.IGNORECASE)
        inline_author = re.compile(r"^\s*(?:written\s+by|screenplay\s+by|story\s+by|by)\s*[:\-]?\s*(.+)$", re.IGNORECASE)

        title = ""
        authors = []
        draft = ""
        source = ""
        metadata_end_idx = 0
        expecting_author = False
        expecting_source = False

        for i, (line_str, _, p_num) in enumerate(lines):
            txt = line_str.strip()
            if not txt:
                continue

            if title and p_num > 1:
                metadata_end_idx = i
                break

            txt_norm = txt.replace("&amp;", "&").replace("&quot;", '"')

            if re.search(r"\b(?:draft|revision|rev\.)\b", txt_norm, re.IGNORECASE):
                draft = txt_norm
                metadata_end_idx = i + 1
                continue

            if re.search(r"\b(?:based\s+on|adapted\s+from|created\s+by)\s*$", txt_norm, re.IGNORECASE):
                expecting_source = True
                source = txt_norm
                continue

            if expecting_source:
                source = (source + " " + txt_norm).strip()
                expecting_source = False
                metadata_end_idx = i + 1
                continue

            if re.search(r"\b(?:based\s+on|adapted\s+from|created\s+by)\b", txt_norm, re.IGNORECASE):
                source = txt_norm
                metadata_end_idx = i + 1
                continue

            if author_cues.match(txt_norm):
                expecting_author = True
                continue

            if expecting_author:
                names = re.split(r"\s+(?:and|&)\s+|\s*,\s*", txt_norm)
                for n in names:
                    n_clean = n.strip()
                    if n_clean and n_clean not in authors:
                        authors.append(n_clean)
                expecting_author = False
                metadata_end_idx = i + 1
                continue

            m_auth = inline_author.match(txt_norm)
            if m_auth:
                auth_str = m_auth.group(1).strip()
                names = re.split(r"\s+(?:and|&)\s+|\s*,\s*", auth_str)
                for n in names:
                    n_clean = n.strip()
                    if n_clean and n_clean not in authors:
                        authors.append(n_clean)
                metadata_end_idx = i + 1
                continue

            if not title and len(authors) == 0 and len(txt_norm.split()) <= 10:
                title = txt_norm
                metadata_end_idx = i + 1
                continue

            if title and authors:
                break

        author_str = ", ".join(authors) if authors else ""
        header = EpisodeHeader(
            episode_num=1,
            raw_episode_title=title or "Screenplay",
            logline=f"Written by {author_str}. {source}".strip() if author_str else source,
            continuity_notes=draft
        )
        return header, metadata_end_idx

    def _is_secondary_slug(self, stripped: str) -> bool:
        if not stripped.isupper():
            return False
        words = stripped.split()
        if len(words) > 6:
            return False
        t = stripped.upper()
        if t in self.SECONDARY_SLUG_KEYWORDS:
            return True
        if any(t.endswith(" " + k) or t.endswith("'S " + k) for k in self.SECONDARY_SLUG_KEYWORDS):
            return True
        if any(t.startswith(p) for p in self.SECONDARY_SLUG_PREFIXES):
            return True
        return False

    def _is_implicit_slugline(self, line: str) -> bool:
        if not line.isupper():
            return False
        parts = re.split(r"\s*[-–—]+\s*", line)
        if len(parts) >= 2:
            for part in parts[1:]:
                clean_p = part.strip().strip("()")
                if clean_p.upper() in self.TIMES_OF_DAY or "FLASHBACK" in clean_p.upper():
                    return True
        return False

    def _profile_screenplay_layout(self, norm_lines: List[Tuple[str, int, int]]) -> None:
        """
        Profiles the physical typography and horizontal indentation bands of the screenplay.
        Establishes indentation as the primary classifier across standard screenplay formats:
          - Action / Scene Heading band: indent <= action_max_indent
          - Dialogue text / parenthetical band: dialogue_min_indent <= indent < char_cue_min_indent
          - Character speaker cue band: char_cue_min_indent <= indent < transition_min_indent
          - Transition band: indent >= transition_min_indent
        """
        non_empty = [len(l[0]) - len(l[0].lstrip()) for l in norm_lines if l[0].strip()]
        if not non_empty:
            self.has_indent_layout = False
            self.base_action_indent = 0
            self.action_max_indent = 0
            self.dialogue_min_indent = 0
            self.char_cue_min_indent = 0
            self.transition_min_indent = 45
            return

        counts = Counter(non_empty)
        min_ind = min(non_empty)
        max_ind = max(non_empty)

        # Standard screenplay formats exhibit significant horizontal whitespace separation (>= 8 columns)
        if max_ind - min_ind >= 8:
            self.has_indent_layout = True
            low_indents = [ind for ind, _ in counts.most_common() if ind <= 15]
            self.base_action_indent = low_indents[0] if low_indents else min_ind
            self.action_max_indent = self.base_action_indent + 4
            self.dialogue_min_indent = self.base_action_indent + 6
            high_indents = [ind for ind in sorted(counts.keys()) if ind >= self.base_action_indent + 12]
            self.char_cue_min_indent = high_indents[0] if high_indents else (self.base_action_indent + 16)
            self.transition_min_indent = max(45, self.char_cue_min_indent + 15)
        else:
            self.has_indent_layout = False
            self.base_action_indent = 0
            self.action_max_indent = 0
            self.dialogue_min_indent = 0
            self.char_cue_min_indent = 0
            self.transition_min_indent = 45

    def _check_screenplay_cue(self, line: str, indent: int) -> Optional[Tuple[str, Optional[str]]]:
        # 1. Primary Indentation Classifier: Character cues MUST be centered in the character cue band
        if self.has_indent_layout:
            if indent < self.char_cue_min_indent:
                return None
        elif self.base_action_indent > 0 and indent < self.base_action_indent + 8:
            return None

        extensions = []
        for m in self.EXTENSION_RE.finditer(line):
            extensions.append(m.group(0))

        name_part = self.EXTENSION_RE.sub("", line).strip()
        if not name_part or not name_part.isupper() or len(name_part.split()) > 5:
            return None

        if any(c in name_part for c in [".", "!", "?", "--"]) and not re.match(r"^(?:MR|MRS|MS|DR)\.\s", name_part):
            return None

        if any(name_part.startswith(cue) for cue in self.CAMERA_CUES):
            return None

        if self._is_secondary_slug(name_part):
            return None

        if name_part.upper() in self.LIGHTING_ATMOSPHERE_CUES or len(name_part) < 2:
            return None

        tokens = name_part.split()
        if len(tokens) > 1 and all(len(t) == 1 for t in tokens):
            return None

        clean_name = re.sub(r"\s+", " ", name_part)
        modifier = " ".join(extensions) if extensions else None
        return clean_name, modifier

    def _parse_standard_screenplay(self, text: str, episode_idx: int = 1, file_name: str = "") -> Episode:
        """Parses traditional screenplay dialects (Dark Knight, 10 Things, Cloverfield) into an Episode AST."""
        norm_lines = self._normalize_screenplay_text(text)
        if not norm_lines:
            header = EpisodeHeader(episode_num=episode_idx, raw_episode_title=file_name or f"Episode {episode_idx}", logline="", continuity_notes="")
            return Episode(header=header, scenes=(), elements=(), total_elements=0, total_duration_cs=0)

        # Baseline indentation calculation & typography layout profiling
        self._profile_screenplay_layout(norm_lines)

        # Extract title page metadata
        header, meta_end_idx = self._extract_screenplay_metadata(norm_lines)
        header = EpisodeHeader(
            episode_num=episode_idx,
            raw_episode_title=header.raw_episode_title or file_name or f"Episode {episode_idx}",
            logline=header.logline,
            continuity_notes=header.continuity_notes
        )

        lines_to_process = norm_lines[meta_end_idx:]
        parsed_scenes: List[Scene] = []
        all_elements: List[ScreenplayElement] = []
        element_counter = 0
        current_cs = 0

        current_scene_num = 0
        current_scene_heading = "COLD OPEN / PROLOGUE"
        current_scene_elements: List[ScreenplayElement] = []

        def flush_scene():
            nonlocal current_scene_elements, current_scene_num, current_scene_heading
            if current_scene_elements:
                current_scene_num += 1
                sc_start_cs = current_scene_elements[0].start_cs
                sc_end_cs = current_scene_elements[-1].end_cs
                duration_cs = sc_end_cs - sc_start_cs
                scene = Scene(
                    episode_num=episode_idx,
                    scene_num=current_scene_num,
                    heading=current_scene_heading,
                    start_raw=current_scene_elements[0].start_raw,
                    end_raw=current_scene_elements[-1].end_raw,
                    start_cs=sc_start_cs,
                    end_cs=sc_end_cs,
                    duration_cs=duration_cs,
                    duration_seconds=duration_cs / 100.0,
                    elements=tuple(current_scene_elements)
                )
                parsed_scenes.append(scene)
            current_scene_elements = []

        def add_element(el_type: ElementType, content: str, dlg_meta: Optional[DialogueMetadata] = None, raw_tag: str = ""):
            nonlocal element_counter, current_cs, current_scene_elements, all_elements
            content_clean = content.strip()
            if not content_clean:
                return
            element_counter += 1
            el_id = f"EP{episode_idx:02d}_EL{element_counter:03d}"
            words = content_clean.split()
            dur_cs = max(100, len(words) * 30)
            start_cs = current_cs
            end_cs = current_cs + dur_cs
            current_cs = end_cs

            elem = ScreenplayElement(
                element_id=el_id,
                episode_num=episode_idx,
                scene_num=max(1, current_scene_num + 1),
                element_type=el_type,
                start_raw=format_cs_to_timestamp(start_cs),
                end_raw=format_cs_to_timestamp(end_cs),
                start_cs=start_cs,
                end_cs=end_cs,
                duration_cs=dur_cs,
                duration_seconds=dur_cs / 100.0,
                content=content_clean,
                dialogue_meta=dlg_meta,
                raw_tag=raw_tag
            )
            current_scene_elements.append(elem)
            all_elements.append(elem)

        accumulated_action: List[str] = []

        def flush_action():
            nonlocal accumulated_action
            if accumulated_action:
                action_text = " ".join(accumulated_action)
                add_element(ElementType.ACTION, action_text)
            accumulated_action = []

        idx = 0
        total_l = len(lines_to_process)

        while idx < total_l:
            raw_line, line_no, p_num = lines_to_process[idx]
            stripped = raw_line.strip()
            indent = len(raw_line) - len(raw_line.lstrip())

            if not stripped:
                flush_action()
                idx += 1
                continue

            # 1. Transitions
            is_transition = (
                self.TRANSITION_RE.match(stripped)
                or (stripped.endswith("TO:") and stripped.isupper())
                or (self.has_indent_layout and indent >= self.transition_min_indent and stripped.isupper())
            )
            if is_transition:
                flush_action()
                add_element(ElementType.ACTION, stripped, raw_tag="TRANSITION")
                idx += 1
                continue

            # 2. Scene Heading (Explicit or Implicit)
            if self.SLUG_PREFIX_RE.match(stripped) or self._is_implicit_slugline(stripped):
                flush_action()
                flush_scene()
                current_scene_heading = stripped
                idx += 1
                continue

            # 3. Secondary Slug / Camera Direction
            if any(stripped.upper().startswith(cue) for cue in self.CAMERA_CUES) or re.search(r"\bPOV\b", stripped) or self._is_secondary_slug(stripped):
                flush_action()
                add_element(ElementType.CAMERA, stripped, raw_tag="CAMERA")
                idx += 1
                continue

            # 4. Character Cue + Dialogue
            cue_res = self._check_screenplay_cue(stripped, indent)
            if cue_res:
                flush_action()
                char_name, modifier = cue_res
                parentheticals = []
                dialogue_lines = []
                idx += 1
                while idx < total_l:
                    next_raw, _, _ = lines_to_process[idx]
                    next_strip = next_raw.strip()
                    next_ind = len(next_raw) - len(next_raw.lstrip())
                    if not next_strip:
                        idx += 1
                        break
                    pm = self.PARENTHETICAL_RE.match(next_strip)
                    if pm:
                        parentheticals.append(pm.group(1).strip())
                        idx += 1
                    elif self.SLUG_PREFIX_RE.match(next_strip) or self._is_implicit_slugline(next_strip) or self.TRANSITION_RE.match(next_strip):
                        break
                    elif self._check_screenplay_cue(next_strip, next_ind):
                        break
                    elif self.has_indent_layout and next_ind <= self.action_max_indent:
                        # Dropping back to the base action margin terminates dialogue block immediately
                        break
                    else:
                        dialogue_lines.append(next_strip)
                        idx += 1

                dialogue_content = " ".join(dialogue_lines) if dialogue_lines else ""
                delivery = " | ".join(parentheticals) if parentheticals else None
                dlg_meta = DialogueMetadata(
                    speaker=char_name,
                    modifier=modifier,
                    delivery=delivery,
                    raw_tag=stripped
                )
                if dialogue_content:
                    add_element(ElementType.DIALOGUE, dialogue_content, dlg_meta=dlg_meta, raw_tag=stripped)
                continue

            # 5. Default: Action line
            accumulated_action.append(stripped)
            idx += 1

        flush_action()
        flush_scene()

        return Episode(
            header=header,
            scenes=tuple(parsed_scenes),
            elements=tuple(all_elements),
            total_elements=len(all_elements),
            total_duration_cs=current_cs
        )

    # Alias for backwards compatibility
    def parse_episode(self, file_path: Path) -> Episode:
        num_m = re.search(r"\d+", file_path.stem)
        idx = int(num_m.group(0)) if num_m else 1
        return self.parse_screenplay_file(file_path, episode_idx=idx)


def partition_episode_into_parts(
    ep: Episode,
    scene_start_pages: List[int],
    start_episode_idx: int = 1,
    target_pages_per_part: int = 20,
    min_pages_last_part: int = 6
) -> List[Episode]:
    """
    Partitions a long feature film Episode into balanced, scene-aligned Episode parts.
    Ensures:
    1. Cuts occur ONLY at scene boundaries (no severed scenes or dialogue volleys).
    2. Each part starts at 00:00.00 with monotonic, self-contained centisecond timestamps.
    3. Element IDs are cleanly namespaced per part: EP01_EL001, EP02_EL001, etc.
    """
    total_pages = max(scene_start_pages) if scene_start_pages else 1
    total_scenes = len(ep.scenes)
    if total_scenes != len(scene_start_pages):
        diff = total_scenes - len(scene_start_pages)
        if diff > 0:
            scene_start_pages = [1] * diff + scene_start_pages
        else:
            scene_start_pages = scene_start_pages[:total_scenes]

    partitions: List[List[int]] = []
    curr_part_indices: List[int] = []
    part_start_p = scene_start_pages[0] if scene_start_pages else 1

    for sc_idx, pno in enumerate(scene_start_pages):
        remaining_pages = total_pages - pno + 1
        if (
            curr_part_indices
            and (pno >= part_start_p + target_pages_per_part)
            and remaining_pages >= min_pages_last_part
        ):
            partitions.append(curr_part_indices)
            curr_part_indices = [sc_idx]
            part_start_p = pno
        else:
            curr_part_indices.append(sc_idx)

    if curr_part_indices:
        partitions.append(curr_part_indices)

    total_parts = len(partitions)
    base_title = ep.header.raw_episode_title or "Feature Screenplay"
    result_episodes: List[Episode] = []

    for part_offset, sc_indices in enumerate(partitions):
        part_num = start_episode_idx + part_offset
        first_sc_idx = sc_indices[0]
        last_sc_idx = sc_indices[-1]
        start_p = scene_start_pages[first_sc_idx]
        end_p = scene_start_pages[last_sc_idx]

        first_scene = ep.scenes[first_sc_idx]
        base_cs = first_scene.start_cs

        part_scenes: List[Scene] = []
        part_elements: List[ScreenplayElement] = []
        el_counter = 0

        for new_sc_idx, orig_sc_idx in enumerate(sc_indices, start=1):
            orig_scene = ep.scenes[orig_sc_idx]
            new_sc_elements: List[ScreenplayElement] = []

            for orig_el in orig_scene.elements:
                el_counter += 1
                new_start_cs = orig_el.start_cs - base_cs
                new_end_cs = orig_el.end_cs - base_cs
                new_dur_cs = new_end_cs - new_start_cs

                new_el = ScreenplayElement(
                    element_id=f"EP{part_num:02d}_EL{el_counter:03d}",
                    episode_num=part_num,
                    scene_num=new_sc_idx,
                    element_type=orig_el.element_type,
                    start_raw=format_cs_to_timestamp(new_start_cs),
                    end_raw=format_cs_to_timestamp(new_end_cs),
                    start_cs=new_start_cs,
                    end_cs=new_end_cs,
                    duration_cs=new_dur_cs,
                    duration_seconds=round(new_dur_cs / 100.0, 2),
                    content=orig_el.content,
                    dialogue_meta=orig_el.dialogue_meta,
                    raw_tag=orig_el.raw_tag
                )
                new_sc_elements.append(new_el)
                part_elements.append(new_el)

            new_sc_start_cs = new_sc_elements[0].start_cs if new_sc_elements else 0
            new_sc_end_cs = new_sc_elements[-1].end_cs if new_sc_elements else 0
            new_sc_dur_cs = new_sc_end_cs - new_sc_start_cs

            new_scene = Scene(
                episode_num=part_num,
                scene_num=new_sc_idx,
                heading=orig_scene.heading,
                start_raw=format_cs_to_timestamp(new_sc_start_cs),
                end_raw=format_cs_to_timestamp(new_sc_end_cs),
                start_cs=new_sc_start_cs,
                end_cs=new_sc_end_cs,
                duration_cs=new_sc_dur_cs,
                duration_seconds=round(new_sc_dur_cs / 100.0, 2),
                elements=tuple(new_sc_elements)
            )
            part_scenes.append(new_scene)

        total_part_dur_cs = part_scenes[-1].end_cs if part_scenes else 0
        part_header = EpisodeHeader(
            episode_num=part_num,
            raw_episode_title=f"{base_title} - Part {part_num:02d} (Scenes {first_sc_idx+1}-{last_sc_idx+1}, Pages {start_p}-{end_p})",
            logline=ep.header.logline,
            continuity_notes=f"Part {part_num} of {total_parts}. Covers scenes {first_sc_idx+1}-{last_sc_idx+1} (pages {start_p}-{end_p})."
        )

        part_ep = Episode(
            header=part_header,
            scenes=tuple(part_scenes),
            elements=tuple(part_elements),
            total_elements=len(part_elements),
            total_duration_cs=total_part_dur_cs
        )
        result_episodes.append(part_ep)

    return result_episodes


def split_pdf_to_parts(
    pdf_path: Path,
    output_dir: Optional[Path] = None,
    target_pages_per_part: int = 20,
    min_pages_last_part: int = 6
) -> List[Path]:
    """
    Physically splits a long screenplay PDF into balanced part PDF files.
    If output_dir is omitted, defaults to a 'parts' subfolder next to the PDF.
    """
    import fitz
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    if output_dir is None:
        output_dir = pdf_path.parent / "parts"
    output_dir.mkdir(parents=True, exist_ok=True)

    page_splits = []
    p = 1
    while p <= total_pages:
        end = min(p + target_pages_per_part - 1, total_pages)
        if (total_pages - end) < min_pages_last_part:
            end = total_pages
        page_splits.append((p, end))
        if end == total_pages:
            break
        p = end + 1

    part_files = []
    base_name = pdf_path.stem
    for idx, (sp, ep) in enumerate(page_splits, start=1):
        out_path = output_dir / f"{base_name}_part{idx:02d}.pdf"
        part_doc = fitz.open()
        part_doc.insert_pdf(doc, from_page=sp - 1, to_page=ep - 1)
        part_doc.save(str(out_path))
        part_doc.close()
        part_files.append(out_path)

    doc.close()
    return part_files


def discover_screenplay_files(folder: Path) -> List[Path]:
    """
    Discovers screenplay files (.txt and .pdf) naturally sorted by numeric episode number.
    Checks folder directly, then subdirectories (e.g. 'Sample script/'), and recursively.
    """
    if folder.is_file() and folder.suffix.lower() in [".txt", ".pdf"]:
        return [folder]

    def extract_num(p: Path) -> int:
        digits = re.findall(r"\d+", p.stem)
        return int(digits[0]) if digits else 999999

    # 1. Direct glob (excluding generated 'parts' subdirectories to avoid duplicates)
    files = [f for f in folder.glob("*.*") if f.suffix.lower() in [".txt", ".pdf"] and "parts" not in f.parts]
    if files:
        return sorted(files, key=extract_num)

    # 2. Check immediate subdirectories
    sub_files = [f for f in folder.glob("*/*.*") if f.suffix.lower() in [".txt", ".pdf"] and "parts" not in f.parts]
    if sub_files:
        return sorted(sub_files, key=extract_num)

    # 3. Recursive glob fallback
    rec_files = [f for f in folder.rglob("*.*") if f.suffix.lower() in [".txt", ".pdf"] and "parts" not in f.parts]
    return sorted(rec_files, key=extract_num)


def resolve_input_directory(candidate: Optional[Union[str, Path]] = None) -> Path:
    """Defensively resolves the directory containing screenplay files."""
    if candidate:
        p = Path(candidate)
        if p.exists():
            return p

    search_candidates = [
        Path("input_scripts/Sample script"),
        Path("input_scripts"),
        Path("../input_scripts/Sample script"),
        Path("../input_scripts"),
    ]

    for c in search_candidates:
        if c.exists() and (list(c.glob("*.txt")) or list(c.glob("*/*.txt"))):
            return c

    for c in search_candidates:
        if c.exists():
            return c

    return Path("input_scripts")


def build_and_export_ast(
    input_dir: Path,
    output_path: Path,
    series_title: str = "I Married a Broken Soldier. He was a Duke"
) -> Dict[str, Any]:
    """Parses all screenplays in input_dir and exports canonical JSON AST."""
    parser = UniversalScreenplayParser()
    files = discover_screenplay_files(input_dir)

    if not files:
        raise FileNotFoundError(f"No .txt screenplay files found in {input_dir.resolve()}")

    print(f"Discovered {len(files)} screenplay files in {input_dir.resolve()}")
    parsed_episodes: List[Episode] = []
    audit_records: List[ValidationAudit] = []

    for f in files:
        res = parser.parse_screenplay_file(f, episode_idx=len(parsed_episodes) + 1)
        eps = res if isinstance(res, list) else [res]
        for ep in eps:
            parsed_episodes.append(ep)

            camera = sum(1 for e in ep.elements if e.element_type == ElementType.CAMERA)
            action = sum(1 for e in ep.elements if e.element_type == ElementType.ACTION)
            sound = sum(1 for e in ep.elements if e.element_type == ElementType.SOUND_FX)
            dialogue = sum(1 for e in ep.elements if e.element_type == ElementType.DIALOGUE)

            t_errs = 0
            c_errs = 0
            errors = []

            for sc in ep.scenes:
                if sc.start_cs >= sc.end_cs:
                    t_errs += 1
                    errors.append(f"Scene {sc.scene_num} non-positive window: {sc.start_cs} >= {sc.end_cs}")
                for el in sc.elements:
                    if el.start_cs >= el.end_cs:
                        t_errs += 1
                        errors.append(f"Element {el.element_id} monotonic failure: {el.start_cs} >= {el.end_cs}")
                    if el.start_cs < sc.start_cs or el.end_cs > (sc.end_cs + 100):
                        c_errs += 1

            audit = ValidationAudit(
                episode_num=ep.episode_num,
                filename=f.name,
                scene_count=len(ep.scenes),
                camera_count=camera,
                action_count=action,
                sound_fx_count=sound,
                dialogue_count=dialogue,
                total_elements=ep.total_elements,
                timestamp_errors=t_errs,
                containment_errors=c_errs,
                is_valid=(len(errors) == 0 and t_errs == 0),
                errors=tuple(errors)
            )
            audit_records.append(audit)

    total_time_failures = sum(a.timestamp_errors for a in audit_records)
    if total_time_failures > 0:
        raise ValueError(f"Discovered {total_time_failures} timestamp monotonicity errors during parsing!")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_ast = {
        "corpus_title": series_title or input_dir.name.replace("_", " ").title(),
        "total_episodes": len(parsed_episodes),
        "total_elements": sum(ep.total_elements for ep in parsed_episodes),
        "element_distribution": {
            "CAMERA": sum(a.camera_count for a in audit_records),
            "ACTION": sum(a.action_count for a in audit_records),
            "SOUND_FX": sum(a.sound_fx_count for a in audit_records),
            "DIALOGUE": sum(a.dialogue_count for a in audit_records)
        },
        "episodes": [
            {
                "episode_num": ep.episode_num,
                "header": asdict(ep.header),
                "scenes": [
                    {
                        "scene_num": sc.scene_num,
                        "heading": sc.heading,
                        "start_cs": sc.start_cs,
                        "end_cs": sc.end_cs,
                        "start_raw": sc.start_raw,
                        "end_raw": sc.end_raw,
                        "duration_cs": sc.duration_cs,
                        "duration_seconds": sc.duration_seconds,
                        "elements": [asdict(e) for e in sc.elements]
                    }
                    for sc in ep.scenes
                ]
            }
            for ep in parsed_episodes
        ]
    }

    output_path.write_text(json.dumps(canonical_ast, indent=2), encoding="utf-8")
    file_size_kb = output_path.stat().st_size / 1024
    print("=" * 70)
    print(f"[OK] Successfully exported canonical AST to: {output_path.resolve()}")
    print(f"     Total Episodes Processed  : {len(parsed_episodes):5d}")
    print(f"     Total Elements Generated  : {canonical_ast['total_elements']:5d}")
    print(f"       - CAMERA Elements       : {canonical_ast['element_distribution']['CAMERA']:5d}")
    print(f"       - ACTION Elements       : {canonical_ast['element_distribution']['ACTION']:5d}")
    print(f"       - SOUND FX Elements     : {canonical_ast['element_distribution']['SOUND_FX']:5d}")
    print(f"       - DIALOGUE Elements     : {canonical_ast['element_distribution']['DIALOGUE']:5d}")
    print(f"     Timestamp Monotonic Errors: {total_time_failures:5d}")
    print(f"     JSON Deliverable Size     : {file_size_kb:.1f} KB")
    print("=" * 70)
    return canonical_ast


def main():
    parser = argparse.ArgumentParser(description="Screenplay AST Generator & Parser Engine")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing .txt screenplays")
    parser.add_argument("--output", type=Path, default=None, help="Path for canonical AST JSON export")
    parser.add_argument("--title", type=str, default="I Married a Broken Soldier. He was a Duke", help="Series title")
    args = parser.parse_args()

    input_dir = resolve_input_directory(args.input_dir)

    if args.output:
        output_path = args.output
    else:
        # Default next to script
        script_dir = Path(__file__).resolve().parent
        output_path = script_dir / "01_structured_episodes.json"

    build_and_export_ast(input_dir=input_dir, output_path=output_path, series_title=args.title)


if __name__ == "__main__":
    main()
