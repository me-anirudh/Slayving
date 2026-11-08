"""
Screenplay Parser & Abstract Syntax Tree (AST) Generator
Production-grade screenplay ingestion engine for microdrama scripts.
Parses raw screenplay text files into a strongly-typed, verifiable JSON AST
with integer centisecond timestamps (1s = 100cs) and element categorization.
"""

import os
import re
import json
import argparse
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any


class ElementType(str, Enum):
    CAMERA = "CAMERA"
    ACTION = "ACTION"
    SOUND_FX = "SOUND FX"
    DIALOGUE = "DIALOGUE"


@dataclass(slots=True, frozen=True)
class DialogueMetadata:
    """Strongly-typed metadata for dialogue elements."""
    speaker: str
    modifier: Optional[str] = None
    emotion_tone: Optional[str] = None


@dataclass(slots=True)
class ScreenplayElement:
    """Canonical atomic unit of a screenplay."""
    element_id: str
    episode_num: int
    scene_num: int
    element_type: ElementType
    raw_tag: str
    start_raw: str
    end_raw: str
    start_cs: int
    end_cs: int
    duration_cs: int
    content: str
    dialogue_meta: Optional[DialogueMetadata] = None


@dataclass(slots=True)
class Scene:
    """Logical scene containing ordered screenplay elements."""
    episode_num: int
    scene_num: int
    heading: str
    start_raw: str
    end_raw: str
    start_cs: int
    end_cs: int
    duration_cs: int
    elements: List[ScreenplayElement]


@dataclass(slots=True)
class EpisodeHeader:
    """Metadata extracted from the episode preamble."""
    episode_num: int
    logline: str
    continuity_notes: str


@dataclass(slots=True)
class Episode:
    """Complete parsed episode containing metadata and scenes."""
    episode_num: int
    header: EpisodeHeader
    scenes: List[Scene]

    @property
    def total_elements(self) -> int:
        return sum(len(s.elements) for s in self.scenes)

    @property
    def total_duration_cs(self) -> int:
        if not self.scenes:
            return 0
        return self.scenes[-1].end_cs - self.scenes[0].start_cs


@dataclass
class EpisodeAudit:
    """Audit verification record for a parsed episode."""
    episode_num: int
    scene_count: int
    total_elements: int
    camera_count: int
    action_count: int
    sound_fx_count: int
    dialogue_count: int
    duration_sec: float
    is_valid: bool
    issues: List[str]


def parse_timestamp_to_cs(ts_str: str) -> int:
    """Converts MM:SS.ss or HH:MM:SS.ss to integer centiseconds."""
    match = re.search(r"(?:(\d{1,2}):)?(\d{2}):(\d{2})\.(\d{2})", ts_str.strip())
    if not match:
        raise ValueError(f"Invalid timestamp format: '{ts_str}'")
    h = int(match.group(1)) if match.group(1) else 0
    m = int(match.group(2))
    s = int(match.group(3))
    cs = int(match.group(4))
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
    Deterministic parser for microdrama screenplay text files.
    """
    def __init__(self):
        self.ts_pattern = re.compile(r"(?:(\d{1,2}):)?(\d{2}):(\d{2})\.(\d{2})")
        self.element_pattern = re.compile(
            r"^\*\*(?P<tag>.+?):\*\*\s*\n"
            r"Start:\s*\((?P<start>[^)]+)\)\s*\|\s*End:\s*\((?P<end>[^)]+)\)\s*\n"
            r"(?P<content>(?:(?!\n\*\*|\Z).)+)",
            re.MULTILINE | re.DOTALL
        )

    def parse_header(self, text: str, fallback_ep_num: int) -> Tuple[EpisodeHeader, str]:
        """Extracts EPISODE NUMBER, LOGLINE, and CONTINUITY NOTES."""
        ep_num = fallback_ep_num
        num_m = re.search(r"^EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", text, re.MULTILINE | re.IGNORECASE)
        if num_m:
            ep_num = int(num_m.group(1))

        logline = ""
        log_m = re.search(r"^EPISODE LOGLINE:\s*(.+?)(?=\n[A-Z\s]+:|\n---|\Z)", text, re.MULTILINE | re.DOTALL)
        if log_m:
            logline = " ".join(log_m.group(1).split())

        continuity = ""
        con_m = re.search(r"^CONTINUITY NOTES:\s*(.+?)(?=\n---|\nSCENE|\Z)", text, re.MULTILINE | re.DOTALL)
        if con_m:
            continuity = " ".join(con_m.group(1).split())

        body = text
        start_sep = re.search(r"^---\s*SCRIPT START\s*---", text, re.MULTILINE)
        if start_sep:
            body = text[start_sep.end():]

        header = EpisodeHeader(
            episode_num=ep_num,
            logline=logline,
            continuity_notes=continuity
        )
        return header, body

    def parse_dialogue_metadata(self, raw_tag: str) -> Optional[DialogueMetadata]:
        """Parses speaker name, delivery modifier, and emotional tone."""
        clean = raw_tag.strip()
        speaker = clean
        modifier = None
        tone = None

        paren_m = re.findall(r"\((.*?)\)", clean)
        if paren_m:
            speaker = re.sub(r"\(.*?\)", "", clean).strip()
            details = [p.strip() for p in paren_m]
            for d in details:
                if any(m in d.upper() for m in ["O.S.", "V.O.", "OFF-SCREEN", "VOICEOVER"]):
                    modifier = d
                else:
                    tone = d if not tone else f"{tone}, {d}"

        return DialogueMetadata(
            speaker=speaker or "SPEAKER",
            modifier=modifier,
            emotion_tone=tone
        )

    def classify_element(self, raw_tag: str) -> Tuple[ElementType, Optional[DialogueMetadata]]:
        """Classifies element tag into one of four screenplay element types."""
        tag_upper = raw_tag.strip().upper()
        if tag_upper == "CAMERA":
            return ElementType.CAMERA, None
        elif tag_upper == "ACTION":
            return ElementType.ACTION, None
        elif tag_upper in ["SOUND FX", "SOUND_FX", "SFX", "AUDIO"]:
            return ElementType.SOUND_FX, None
        else:
            meta = self.parse_dialogue_metadata(raw_tag)
            return ElementType.DIALOGUE, meta

    def parse_episode(self, file_path: Path) -> Episode:
        """Parses a complete screenplay text file into an Episode object."""
        text = file_path.read_text(encoding="utf-8")
        filename_num = 1
        num_m = re.search(r"\d+", file_path.stem)
        if num_m:
            filename_num = int(num_m.group(0))

        header, body = self.parse_header(text, fallback_ep_num=filename_num)
        ep_num = header.episode_num

        scene_splits = list(re.finditer(
            r"^(?:###\s*)?SCENE\s+(\d+)[\s:—–-]*([^\n]*?)\s*\n"
            r"(?:.*?Scene (?:start|window):\s*\(([^)]+)\)\s*\|\s*Scene end:\s*\(([^)]+)\))?",
            body,
            re.MULTILINE | re.IGNORECASE
        ))

        scenes: List[Scene] = []
        global_el_idx = 1

        if not scene_splits:
            first_m = self.element_pattern.search(body)
            start_raw = first_m.group("start") if first_m else "00:00.00"
            elements = []
            for m in self.element_pattern.finditer(body):
                el_id = f"EP{ep_num:02d}_EL{global_el_idx:03d}"
                raw_tag = m.group("tag").strip()
                el_type, meta = self.classify_element(raw_tag)
                s_raw = m.group("start").strip()
                e_raw = m.group("end").strip()
                s_cs = parse_timestamp_to_cs(s_raw)
                e_cs = parse_timestamp_to_cs(e_raw)
                content = m.group("content").strip()

                el = ScreenplayElement(
                    element_id=el_id,
                    episode_num=ep_num,
                    scene_num=1,
                    element_type=el_type,
                    raw_tag=raw_tag,
                    start_raw=s_raw,
                    end_raw=e_raw,
                    start_cs=s_cs,
                    end_cs=e_cs,
                    duration_cs=e_cs - s_cs,
                    content=content,
                    dialogue_meta=meta
                )
                elements.append(el)
                global_el_idx += 1

            if elements:
                scenes.append(Scene(
                    episode_num=ep_num,
                    scene_num=1,
                    heading="INT./EXT. DEFAULT SCENE",
                    start_raw=elements[0].start_raw,
                    end_raw=elements[-1].end_raw,
                    start_cs=elements[0].start_cs,
                    end_cs=elements[-1].end_cs,
                    duration_cs=elements[-1].end_cs - elements[0].start_cs,
                    elements=elements
                ))
            return Episode(episode_num=ep_num, header=header, scenes=scenes)

        for i, sc_m in enumerate(scene_splits):
            sc_num = int(sc_m.group(1))
            heading = sc_m.group(2).strip() or f"SCENE {sc_num}"
            sc_start_raw = sc_m.group(3)
            sc_end_raw = sc_m.group(4)

            content_start = sc_m.end()
            content_end = scene_splits[i + 1].start() if i + 1 < len(scene_splits) else len(body)
            sc_body = body[content_start:content_end]

            sc_elements: List[ScreenplayElement] = []
            for el_m in self.element_pattern.finditer(sc_body):
                el_id = f"EP{ep_num:02d}_EL{global_el_idx:03d}"
                raw_tag = el_m.group("tag").strip()
                el_type, meta = self.classify_element(raw_tag)
                s_raw = el_m.group("start").strip()
                e_raw = el_m.group("end").strip()
                s_cs = parse_timestamp_to_cs(s_raw)
                e_cs = parse_timestamp_to_cs(e_raw)
                content = el_m.group("content").strip()

                el = ScreenplayElement(
                    element_id=el_id,
                    episode_num=ep_num,
                    scene_num=sc_num,
                    element_type=el_type,
                    raw_tag=raw_tag,
                    start_raw=s_raw,
                    end_raw=e_raw,
                    start_cs=s_cs,
                    end_cs=e_cs,
                    duration_cs=e_cs - s_cs,
                    content=content,
                    dialogue_meta=meta
                )
                sc_elements.append(el)
                global_el_idx += 1

            if not sc_elements:
                continue

            act_start_cs = parse_timestamp_to_cs(sc_start_raw) if sc_start_raw else sc_elements[0].start_cs
            act_end_cs = parse_timestamp_to_cs(sc_end_raw) if sc_end_raw else sc_elements[-1].end_cs

            scene = Scene(
                episode_num=ep_num,
                scene_num=sc_num,
                heading=heading,
                start_raw=sc_start_raw or sc_elements[0].start_raw,
                end_raw=sc_end_raw or sc_elements[-1].end_raw,
                start_cs=act_start_cs,
                end_cs=act_end_cs,
                duration_cs=act_end_cs - act_start_cs,
                elements=sc_elements
            )
            scenes.append(scene)

        return Episode(episode_num=ep_num, header=header, scenes=scenes)


def discover_screenplay_files(folder: Path) -> List[Path]:
    """Discovers and naturally sorts screenplay text files."""
    txt_files = list(folder.glob("*.txt"))
    def extract_num(p: Path) -> int:
        digits = re.findall(r"\d+", p.stem)
        return int(digits[0]) if digits else 999999
    return sorted(txt_files, key=extract_num)


def build_and_export_ast(
    input_dir: Path,
    output_path: Path,
    series_title: str = "I Married a Broken Soldier. He was a Duke"
) -> Dict[str, Any]:
    """Parses all screenplays in input_dir and exports canonical JSON AST."""
    parser = UniversalScreenplayParser()
    files = discover_screenplay_files(input_dir)

    if not files:
        raise FileNotFoundError(f"No .txt screenplay files found in {input_dir}")

    parsed_episodes: List[Episode] = []
    audit_records: List[EpisodeAudit] = []

    for f in files:
        ep = parser.parse_episode(f)
        parsed_episodes.append(ep)

        camera = sum(1 for s in ep.scenes for e in s.elements if e.element_type == ElementType.CAMERA)
        action = sum(1 for s in ep.scenes for e in s.elements if e.element_type == ElementType.ACTION)
        sound = sum(1 for s in ep.scenes for e in s.elements if e.element_type == ElementType.SOUND_FX)
        dialogue = sum(1 for s in ep.scenes for e in s.elements if e.element_type == ElementType.DIALOGUE)

        issues = []
        for s in ep.scenes:
            for e in s.elements:
                if e.end_cs <= e.start_cs:
                    issues.append(f"Non-positive duration in {e.element_id}")

        audit = EpisodeAudit(
            episode_num=ep.episode_num,
            scene_count=len(ep.scenes),
            total_elements=ep.total_elements,
            camera_count=camera,
            action_count=action,
            sound_fx_count=sound,
            dialogue_count=dialogue,
            duration_sec=round(ep.total_duration_cs / 100.0, 2),
            is_valid=len(issues) == 0,
            issues=issues
        )
        audit_records.append(audit)

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
                        "duration_seconds": sc.duration_cs / 100.0,
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
    print(f"[OK] Successfully exported AST to: {output_path.resolve()}")
    print(f"     Total Episodes : {len(parsed_episodes)}")
    print(f"     Total Elements : {canonical_ast['total_elements']}")
    print(f"     File Size      : {file_size_kb:.1f} KB")
    return canonical_ast


def main():
    parser = argparse.ArgumentParser(description="Screenplay AST Generator")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing .txt screenplays")
    parser.add_argument("--output", type=Path, default=None, help="Path for canonical AST JSON export")
    parser.add_argument("--title", type=str, default="I Married a Broken Soldier. He was a Duke", help="Series title")
    args = parser.parse_args()

    # Defensive path resolution
    if args.input_dir:
        input_dir = args.input_dir
    elif Path("input_scripts").exists():
        input_dir = Path("input_scripts")
    elif Path("../input_scripts").exists():
        input_dir = Path("../input_scripts")
    else:
        input_dir = Path("input_scripts")

    if args.output:
        output_path = args.output
    else:
        # Default next to script
        script_dir = Path(__file__).resolve().parent
        output_path = script_dir / "01_structured_episodes.json"

    build_and_export_ast(input_dir=input_dir, output_path=output_path, series_title=args.title)


if __name__ == "__main__":
    main()
