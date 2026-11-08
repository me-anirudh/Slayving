"""
Sample Screenplay Format Inspector & Structural Validator
Inspects screenplay text files (e.g., ep01.txt, ep25.txt, ep50.txt) to verify:
1. Episode headers (EPISODE NUMBER, EPISODE LOGLINE, CONTINUITY NOTES)
2. Scene boundaries (SCENE <N>, Scene heading, Scene start, Scene end)
3. Timestamps (MM:SS.ss format, validity, start < end monotonicity)
4. Screenplay element tags (DIALOGUE, ACTION, CAMERA, SOUND FX)
"""

import re
from pathlib import Path

SAMPLE_FILES = ["ep01.txt", "ep25.txt", "ep50.txt"]
INPUT_DIR = Path("input_scripts") if Path("input_scripts").exists() else Path("../input_scripts")

def inspect_file(filename: str):
    file_path = INPUT_DIR / filename
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return

    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    print("\n" + "=" * 80)
    print(f"INSPECTING SAMPLE SCREENPLAY: {filename} (Total lines: {len(lines)}, Size: {len(text)} bytes)")
    print("=" * 80)

    # 1. Header Verification
    print("\n--- 1. HEADER INSPECTION ---")
    header_patterns = {
        "EPISODE NUMBER": r"^EPISODE NUMBER:\s*(.+)$",
        "EPISODE LOGLINE": r"^EPISODE LOGLINE:\s*(.+)$",
        "CONTINUITY NOTES": r"^CONTINUITY NOTES:\s*(.+)$",
    }
    for header, pat in header_patterns.items():
        match = re.search(pat, text, re.MULTILINE)
        if match:
            val = match.group(1).strip()
            # truncate display if too long
            disp_val = val[:100] + "..." if len(val) > 100 else val
            print(f"  [FOUND] {header}: \"{disp_val}\"")
        else:
            print(f"  [MISSING] {header}")

    # Separator check
    separator = re.search(r"^---\s*SCRIPT START\s*---", text, re.MULTILINE)
    if separator:
        print("  [FOUND] Script start separator: '--- SCRIPT START ---'")
    else:
        print("  [INFO] No explicit '--- SCRIPT START ---' marker")

    # 2. Scene Boundary Inspection
    print("\n--- 2. SCENE BOUNDARY INSPECTION ---")
    scene_matches = list(re.finditer(
        r"^(SCENE\s+\d+)\s*\nScene heading:\s*(.+)\s*\nScene start:\s*([0-9]{2}:[0-9]{2}\.[0-9]{2})\s*\nScene end:\s*([0-9]{2}:[0-9]{2}\.[0-9]{2})",
        text,
        re.MULTILINE
    ))
    print(f"  Total scenes detected: {len(scene_matches)}")
    for sc in scene_matches:
        s_id, heading, start_t, end_t = sc.groups()
        print(f"  -> {s_id} | Heading: {heading.strip()} | Window: [{start_t} -> {end_t}]")

    # 3. Element Tags & Timestamps Inspection
    print("\n--- 3. ELEMENT TAGS & TIMESTAMPS (4 Screenplay Element Types) ---")
    element_pattern = re.compile(
        r"^\*\*([A-Z0-9_\s\.\-\'\,\/]+(?:\(.*?\))?):\*\*\s*\n"
        r"Start:\s*\(([0-9]{2}:[0-9]{2}\.[0-9]{2})\)\s*\|\s*End:\s*\(([0-9]{2}:[0-9]{2}\.[0-9]{2})\)",
        re.MULTILINE
    )

    matches = list(element_pattern.finditer(text))
    print(f"  Total timed element blocks detected: {len(matches)}")

    counts = {"CAMERA": 0, "ACTION": 0, "SOUND FX": 0, "DIALOGUE": 0}
    timestamp_errors = 0
    sample_dialogues = []
    sample_camera = []
    sample_action = []
    sample_sound = []

    time_regex = re.compile(r"^([0-9]{2}):([0-9]{2})\.([0-9]{2})$")

    def parse_cs(ts_str : str):
        m = time_regex.match(ts_str)
        if not m:
            return None
        mm, ss, cs = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return mm * 6000 + ss * 100 + cs

    for m in matches:
        raw_tag = m.group(1).strip()
        start_ts = m.group(2)
        end_ts = m.group(3)

        start_cs = parse_cs(start_ts)
        end_cs = parse_cs(end_ts)

        if start_cs is None or end_cs is None or start_cs >= end_cs:
            timestamp_errors += 1

        # Classify tag
        if raw_tag == "CAMERA":
            counts["CAMERA"] += 1
            if len(sample_camera) < 1:
                sample_camera.append((raw_tag, start_ts, end_ts))
        elif raw_tag == "ACTION":
            counts["ACTION"] += 1
            if len(sample_action) < 1:
                sample_action.append((raw_tag, start_ts, end_ts))
        elif "SOUND FX" in raw_tag:
            counts["SOUND FX"] += 1
            if len(sample_sound) < 1:
                sample_sound.append((raw_tag, start_ts, end_ts))
        else:
            counts["DIALOGUE"] += 1
            if len(sample_dialogues) < 2:
                sample_dialogues.append((raw_tag, start_ts, end_ts))

    print(f"  Element Distribution:")
    print(f"    - CAMERA:   {counts['CAMERA']:3d} elements")
    print(f"    - ACTION:   {counts['ACTION']:3d} elements")
    print(f"    - SOUND FX: {counts['SOUND FX']:3d} elements")
    print(f"    - DIALOGUE: {counts['DIALOGUE']:3d} elements")
    print(f"  Timestamp Format & Validity:")
    print(f"    - Regex check: MM:SS.ss (minutes:seconds.centiseconds)")
    print(f"    - Timestamp format / monotonic errors: {timestamp_errors}")

    print("\n--- 4. VERBATIM SAMPLES ---")
    if sample_camera:
        print(f"  [CAMERA Sample]: **{sample_camera[0][0]}:** | Start: ({sample_camera[0][1]}) | End: ({sample_camera[0][2]})")
    if sample_action:
        print(f"  [ACTION Sample]: **{sample_action[0][0]}:** | Start: ({sample_action[0][1]}) | End: ({sample_action[0][2]})")
    if sample_sound:
        print(f"  [SOUND FX Sample]: **{sample_sound[0][0]}:** | Start: ({sample_sound[0][1]}) | End: ({sample_sound[0][2]})")
    for d in sample_dialogues:
        print(f"  [DIALOGUE Sample]: **{d[0]}:** | Start: ({d[1]}) | End: ({d[2]})")

def main():
    for f in SAMPLE_FILES:
        inspect_file(f)

if __name__ == "__main__":
    main()
