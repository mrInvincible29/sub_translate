#!/usr/bin/env python3
"""
Combine two subtitle files into a bilingual ASS subtitle.
EN on bottom, second language above it. Optimized for Plex TV apps.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

KNOWN_SUFFIX_TAGS = {
    "hi", "sdh", "forced", "cc", "subs", "sub", "dub", "commentary", "chs", "cht"
}


@dataclass
class Segment:
    num: int
    timestamp: str
    lines: List[str]


def read_text_with_fallback(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_srt(text: str) -> List[Segment]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", text.strip())
    segs: List[Segment] = []
    for block in blocks:
        lines = block.split("\n")
        if len(lines) >= 2 and re.fullmatch(r"\d+", lines[0].strip()) and "-->" in lines[1]:
            num = int(lines[0].strip())
            ts = lines[1].strip()
            body = lines[2:]
            segs.append(Segment(num=num, timestamp=ts, lines=body))
        else:
            segs.append(Segment(num=0, timestamp="", lines=lines))
    return segs


def is_lang_srt(path: Path, lang: str) -> bool:
    name = path.name.lower()
    return name.endswith(".srt") and (f".{lang}." in name or name.endswith(f".{lang}.srt"))


def strip_lang_and_tags(name: str) -> str:
    parts = name.split(".")
    if len(parts) < 2 or parts[-1].lower() != "srt":
        return name
    i = len(parts) - 2
    while i >= 0 and parts[i].lower() in KNOWN_SUFFIX_TAGS:
        i -= 1
    if i >= 0 and re.fullmatch(r"[A-Za-z]{2,3}([_-][A-Za-z]{2})?", parts[i]):
        parts.pop(i)
    return ".".join(parts)


def find_best_pair(dir_path: Path, lang_a: str, lang_b: str) -> Tuple[Path, Path]:
    srts = sorted(dir_path.glob("*.srt"))
    a = [p for p in srts if is_lang_srt(p, lang_a)]
    b = [p for p in srts if is_lang_srt(p, lang_b)]
    if not a or not b:
        raise FileNotFoundError(f"Missing .{lang_a}.srt or .{lang_b}.srt in directory")

    a_map = {strip_lang_and_tags(p.name): p for p in a}
    b_map = {strip_lang_and_tags(p.name): p for p in b}
    common = sorted(set(a_map.keys()) & set(b_map.keys()))
    if common:
        key = common[0]
        return a_map[key], b_map[key]

    return a[0], b[0]


def resolve_tmdb_dir(tmdb_id: str, movies_root: Path) -> Path:
    if not tmdb_id.isdigit():
        raise FileNotFoundError(tmdb_id)
    tag = f"{{tmdb-{tmdb_id}}}"
    candidates = [p for p in movies_root.glob(f"*{tag}*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No directory with {tag} under {movies_root}")
    return sorted(candidates)[0]


def parse_srt_timestamp(ts: str) -> Tuple[str, str]:
    parts = ts.split("-->")
    if len(parts) != 2:
        raise ValueError(f"Bad timestamp: {ts}")
    return parts[0].strip(), parts[1].strip()


def srt_to_ass_time(t: str) -> str:
    # "00:01:02,345" -> "0:01:02.34"
    hh, mm, rest = t.split(":")
    ss, ms = rest.split(",")
    cs = str(int(ms) // 10).rjust(2, "0")
    return f"{int(hh)}:{mm}:{ss}.{cs}"


ASS_OVERRIDE_RE = re.compile(r"(\{\\[^}]*\})")


def ass_escape(text: str) -> str:
    parts = ASS_OVERRIDE_RE.split(text)
    out: List[str] = []
    for part in parts:
        if ASS_OVERRIDE_RE.fullmatch(part):
            out.append(part)
            continue
        part = part.replace("\\", r"\\")
        part = part.replace("{", r"\{").replace("}", r"\}")
        part = part.replace("\n", r"\N")
        out.append(part)
    return "".join(out)


def html_to_ass(text: str) -> str:
    # Basic HTML tag conversion for common subtitle tags.
    text = re.sub(r"<i>", r"{\\i1}", text, flags=re.IGNORECASE)
    text = re.sub(r"</i>", r"{\\i0}", text, flags=re.IGNORECASE)
    text = re.sub(r"<b>", r"{\\b1}", text, flags=re.IGNORECASE)
    text = re.sub(r"</b>", r"{\\b0}", text, flags=re.IGNORECASE)
    text = re.sub(r"<u>", r"{\\u1}", text, flags=re.IGNORECASE)
    text = re.sub(r"</u>", r"{\\u0}", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*an([1-9])\s*>", r"{\\an\1}", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*\\an([1-9])\s*>", r"{\\an\1}", text, flags=re.IGNORECASE)

    def font_tag_repl(match: re.Match) -> str:
        attrs = match.group(1) or ""
        parts: List[str] = []
        color = re.search(r'color\s*=\s*"(#?[0-9A-Fa-f]{6})"', attrs)
        if color:
            hexval = color.group(1).lstrip("#")
            bb = hexval[4:6]
            gg = hexval[2:4]
            rr = hexval[0:2]
            parts.append(rf"{{\c&H{bb}{gg}{rr}&}}")
        face = re.search(r'face\s*=\s*"([^"]+)"', attrs)
        if face:
            parts.append(rf"{{\fn{face.group(1)}}}")
        size = re.search(r'size\s*=\s*"(\d+)"', attrs)
        if size:
            parts.append(rf"{{\fs{size.group(1)}}}")
        return "".join(parts)

    text = re.sub(r"<font\s+([^>]*)>", font_tag_repl, text, flags=re.IGNORECASE)
    text = re.sub(r"</font>", r"{\\r}", text, flags=re.IGNORECASE)
    return text


def combine_to_ass(a_segs: List[Segment], b_segs: List[Segment]) -> List[str]:
    a_map: Dict[int, Segment] = {s.num: s for s in a_segs if s.timestamp}
    b_map: Dict[int, Segment] = {s.num: s for s in b_segs if s.timestamp}

    missing = sorted(set(a_map.keys()) ^ set(b_map.keys()))
    if missing:
        raise ValueError(f"Segment mismatch; missing ids: {missing[:10]}")

    events: List[str] = []
    for s in a_segs:
        if not s.timestamp:
            continue
        b = b_map.get(s.num)
        if not b:
            raise ValueError(f"Missing segment id {s.num} in second language")
        start_srt, end_srt = parse_srt_timestamp(s.timestamp)
        start = srt_to_ass_time(start_srt)
        end = srt_to_ass_time(end_srt)
        a_text = ass_escape(html_to_ass("\n".join(s.lines).strip()))
        b_text = ass_escape(html_to_ass("\n".join(b.lines).strip()))
        if a_text:
            events.append(f"Dialogue: 0,{start},{end},A,,0,0,0,,{a_text}")
        if b_text:
            events.append(f"Dialogue: 0,{start},{end},B,,0,0,0,,{b_text}")
    return events


def ass_header(play_res_x: int, play_res_y: int, margin_v_a: int, margin_v_b: int) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: A,Arial,50,&H00FFFFFF,&H000000FF,&H00111111,&H90000000,0,0,0,0,100,100,0,0,1,2,0,2,40,40,{margin_v_a},1
Style: B,Tahoma,48,&H00C8F4FF,&H000000FF,&H00111111,&H90000000,0,0,0,0,100,100,0,0,1,2,0,2,40,40,{margin_v_b},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Combine two subtitle languages into one ASS subtitle.")
    ap.add_argument("input", help="tmdb id, or directory containing subtitle files")
    ap.add_argument("--movies-root", default="/home/h2/media/Movies",
                    help="Root folder to search when input is a tmdb id (default: /home/h2/media/Movies)")
    ap.add_argument("--lang-a", default="en", help="Primary language code (default: en)")
    ap.add_argument("--lang-b", default="th", help="Secondary language code (default: th)")
    ap.add_argument("--a", dest="a_path", help="Explicit primary .srt path (overrides search)")
    ap.add_argument("--b", dest="b_path", help="Explicit secondary .srt path (overrides search)")
    ap.add_argument("-o", "--output", help="Output .ass path (default: <base>.<A>+<B>.ass)")
    ap.add_argument("--playres-x", type=int, default=1920, help="ASS PlayResX (default: 1920)")
    ap.add_argument("--playres-y", type=int, default=1080, help="ASS PlayResY (default: 1080)")
    ap.add_argument("--margin-a", type=int, default=30, help="Bottom margin for primary language (default: 30)")
    ap.add_argument("--margin-b", type=int, default=80, help="Bottom margin for secondary language (default: 80)")

    args = ap.parse_args()

    lang_a = args.lang_a.lower()
    lang_b = args.lang_b.lower()
    if args.a_path and args.b_path:
        a_path = Path(args.a_path).expanduser()
        b_path = Path(args.b_path).expanduser()
    else:
        movies_root = Path(args.movies_root).expanduser()
        if args.input.isdigit():
            dir_path = resolve_tmdb_dir(args.input, movies_root)
        else:
            dir_path = Path(args.input).expanduser()
        if not dir_path.is_dir():
            print(f"Not a directory: {dir_path}", file=sys.stderr)
            return 2
        a_path, b_path = find_best_pair(dir_path, lang_a, lang_b)

    if not a_path.exists() or not b_path.exists():
        print("Both subtitle files must exist.", file=sys.stderr)
        return 2

    a_text = read_text_with_fallback(a_path)
    b_text = read_text_with_fallback(b_path)
    a_segs = parse_srt(a_text)
    b_segs = parse_srt(b_text)

    try:
        events = combine_to_ass(a_segs, b_segs)
    except ValueError as e:
        print(f"Combine failed: {e}", file=sys.stderr)
        return 2

    if args.output:
        out_path = Path(args.output).expanduser()
    else:
        base = strip_lang_and_tags(a_path.name)
        tag = f".{lang_a.upper()}+{lang_b.upper()}.ass"
        out_path = a_path.with_name(base.replace(".srt", tag))

    header = ass_header(args.playres_x, args.playres_y, args.margin_a, args.margin_b)
    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"WROTE: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
