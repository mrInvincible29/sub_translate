#!/usr/bin/env python3
"""
Combine two subtitle files into one bilingual .srt (EN first, then TH).
Optimized for Plex: keeps original timestamps; combines lines per segment.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def format_srt(segments: List[Segment]) -> str:
    out: List[str] = []
    need_renumber = any(s.num == 0 or not s.timestamp for s in segments)
    next_num = 1
    for s in segments:
        if s.timestamp:
            num = next_num if need_renumber else s.num
            out.append(str(num))
            out.append(s.timestamp)
            out.extend(s.lines)
            out.append("")
            next_num += 1
        else:
            out.extend(s.lines)
            out.append("")
    return "\n".join(out).rstrip() + "\n"


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


def combine_segments(a_segs: List[Segment], b_segs: List[Segment]) -> List[Segment]:
    a_map: Dict[int, Segment] = {s.num: s for s in a_segs if s.timestamp}
    b_map: Dict[int, Segment] = {s.num: s for s in b_segs if s.timestamp}

    missing = sorted(set(a_map.keys()) ^ set(b_map.keys()))
    if missing:
        raise ValueError(f"Segment mismatch; missing ids: {missing[:10]}")

    out: List[Segment] = []
    for s in a_segs:
        if not s.timestamp:
            out.append(s)
            continue
        b = b_map.get(s.num)
        if not b:
            raise ValueError(f"Missing segment id {s.num} in second language")
        lines = s.lines + [""] + b.lines
        out.append(Segment(num=s.num, timestamp=s.timestamp, lines=lines))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Combine two subtitle languages into one bilingual .srt.")
    ap.add_argument("input", help="tmdb id, or directory containing subtitle files")
    ap.add_argument("--movies-root", default="/home/h2/media/Movies",
                    help="Root folder to search when input is a tmdb id (default: /home/h2/media/Movies)")
    ap.add_argument("--lang-a", default="en", help="Primary language code (default: en)")
    ap.add_argument("--lang-b", default="th", help="Secondary language code (default: th)")
    ap.add_argument("--a", dest="a_path", help="Explicit primary .srt path (overrides search)")
    ap.add_argument("--b", dest="b_path", help="Explicit secondary .srt path (overrides search)")
    ap.add_argument("-o", "--output", help="Output .srt path (default: <base>.<A>+<B>.srt)")

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
        combined = combine_segments(a_segs, b_segs)
    except ValueError as e:
        print(f"Combine failed: {e}", file=sys.stderr)
        return 2

    if args.output:
        out_path = Path(args.output).expanduser()
    else:
        base = strip_lang_and_tags(a_path.name)
        tag = f".{lang_a.upper()}+{lang_b.upper()}.srt"
        out_path = a_path.with_name(base.replace(".srt", tag))

    out_path.write_text(format_srt(combined), encoding="utf-8")
    print(f"WROTE: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
