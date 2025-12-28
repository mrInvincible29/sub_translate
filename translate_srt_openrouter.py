#!/usr/bin/env python3
"""
Translate .srt subtitle files via OpenRouter, with optional per-segment context and post-translation validation/QC.

- Translates subtitle blocks (segments), not individual words.
- Renames output language code in filename (e.g. *.en.hi.srt -> *.th.hi.srt).
- Context window: include previous/next segments in each request to improve continuity.
- Validation: structural checks (tags, speaker labels, line breaks, "still English" heuristics).
- Optional LLM QC: ask a judge model to score/flag segments and emit a JSON report.

Quick start:
  pip install requests
  export OPENROUTER_API_KEY="..."
  python3 translate_srt_openrouter.py "/path/Movie.en.hi.srt" -t th --json-mode --context-window 2 --validate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

KNOWN_SUFFIX_TAGS = {
    "hi", "sdh", "forced", "cc", "subs", "sub", "dub", "commentary", "chs", "cht"
}

TRANSLATOR_TAG_RE = re.compile(r"^\s*#\s*Subtitles translated with .*#\s*$", re.IGNORECASE)

TARGET_LANG_NAMES = {
    "th": "Thai",
    "en": "English",
}

SYSTEM_PROMPT_TRANSLATE = """You are a professional subtitle translator.
Translate English subtitle text into {target_lang_name}.

Input is JSON:
{{
  "target_language":"{target_lang_code}",
  "segments":[
    {{
      "id":"7",
      "text":"ATLAS: <i>Did you think\\nwe abandoned you?</i>",
      "context_prev":["<i>Horsemen! Horsemen!</i>"],
      "context_next":["(AUDIENCE CHEERING LOUDLY)"]
    }},
    ...
  ]
}}

Output MUST be ONLY valid JSON:
{{
  "segments":[
    {{"id":"7","text":"..."}},
    ...
  ]
}}

Rules:
- Preserve ALL markup tags exactly as-is (e.g., <i>, </i>, <b>, <font ...>). Do not add/remove/rename tags.
- Do not change anything inside angle brackets <>.
- Keep line breaks in each segment exactly (if the input "text" contains \\n, output the same number of lines).
- Keep ALL-CAPS speaker labels ending with ':' unchanged (e.g., ATLAS:, AUDIENCE:).
- Use context_prev/context_next only to resolve meaning/pronouns/continuity; translate ONLY "text".
"""

SYSTEM_PROMPT_QC = """You are a bilingual subtitle QA reviewer.
You will receive JSON with English source segments and their {target_lang_name} translations.

Return ONLY valid JSON:
{{
  "segments":[
    {{"id":"7","score":1,"issues":["..."],"notes":"..."}},
    ...
  ]
}}

Scoring:
- 5: excellent and natural
- 4: good, minor nit
- 3: understandable but noticeable issues
- 2: meaning likely wrong or very unnatural
- 1: clearly wrong / not translated / broken formatting

Rules:
- Keep notes short (max ~120 chars).
- Flag formatting problems (lost/changed tags, wrong line breaks, changed speaker label).
- Flag meaning drift, missing lines, wrong tone, mistranslation, untranslated English.
"""

TAG_RE = re.compile(r"<[^>]+>")
LEADING_TAGS_RE = re.compile(r"^\s*(?:<[^>]+>\s*)+")
SPEAKER_RE = re.compile(r"^([A-Z][A-Z0-9 .'\-]{0,40}):")


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


def normalize_lines(lines: List[str], strip_translator_tag: bool) -> List[str]:
    if not strip_translator_tag:
        return lines
    return [ln for ln in lines if not TRANSLATOR_TAG_RE.match(ln)]


def segment_text(seg: Segment, strip_translator_tag: bool) -> str:
    lines = normalize_lines(seg.lines, strip_translator_tag)
    return "\n".join(lines).strip()


def swap_lang_in_filename(filename: str, new_lang: str) -> str:
    parts = filename.split(".")
    if len(parts) < 2 or parts[-1].lower() != "srt":
        return filename

    i = len(parts) - 2
    while i >= 0 and parts[i].lower() in KNOWN_SUFFIX_TAGS:
        i -= 1

    if i >= 0 and re.fullmatch(r"[A-Za-z]{2,3}([_-][A-Za-z]{2})?", parts[i]):
        parts[i] = new_lang
        return ".".join(parts)

    parts.insert(len(parts) - 1, new_lang)
    return ".".join(parts)


def item_size_chars(it: Dict[str, object]) -> int:
    total = 0
    for v in it.values():
        if isinstance(v, str):
            total += len(v)
        elif isinstance(v, list):
            total += sum(len(x) for x in v if isinstance(x, str))
    return total


def chunk_by_chars(items: List[Dict[str, object]], max_chars: int) -> Iterable[List[Dict[str, object]]]:
    chunk: List[Dict[str, object]] = []
    total = 0
    for it in items:
        size = item_size_chars(it)
        if chunk and total + size > max_chars:
            yield chunk
            chunk = []
            total = 0
        chunk.append(it)
        total += size
    if chunk:
        yield chunk


def safe_json_loads(s: str) -> dict:
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise
    return json.loads(m.group(0))


def openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    app_url: str,
    app_title: str,
    response_format: Optional[dict],
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_title:
        headers["X-Title"] = app_title

    payload: Dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    r = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def translate_batch(
    *,
    api_key: str,
    model: str,
    target_lang: str,
    batch: List[Dict[str, object]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    app_url: str,
    app_title: str,
    response_format: Optional[dict],
    attempts: int,
) -> Dict[str, str]:
    target_name = TARGET_LANG_NAMES.get(target_lang, target_lang)
    req_obj = {"target_language": target_lang, "segments": batch}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TRANSLATE.format(target_lang_name=target_name, target_lang_code=target_lang)},
        {"role": "user", "content": json.dumps(req_obj, ensure_ascii=False)},
    ]

    last_err: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            resp = openrouter_chat(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                app_url=app_url,
                app_title=app_title,
                response_format=response_format,
            )
            content = resp["choices"][0]["message"]["content"]
            obj = safe_json_loads(content)
            segs = obj.get("segments", [])
            out: Dict[str, str] = {}
            for s in segs:
                sid = str(s.get("id"))
                out[sid] = s.get("text", "")
            return out
        except Exception as e:
            last_err = e
            time.sleep(min(10.0, 0.8 * (2 ** attempt)))

    raise last_err or RuntimeError("translation failed")


def translate_items_recursive(
    *,
    api_key: str,
    model: str,
    target_lang: str,
    items: List[Dict[str, object]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    app_url: str,
    app_title: str,
    response_format: Optional[dict],
    attempts: int,
) -> Dict[str, str]:
    if not items:
        return {}

    got = translate_batch(
        api_key=api_key,
        model=model,
        target_lang=target_lang,
        batch=items,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        app_url=app_url,
        app_title=app_title,
        response_format=response_format,
        attempts=attempts,
    )

    want_ids = {str(it["id"]) for it in items}
    got_ids = set(got.keys())
    if want_ids.issubset(got_ids) and all(got.get(i, "") != "" for i in want_ids):
        return got

    if len(items) == 1:
        only = items[0]
        return {str(only["id"]): str(got.get(str(only["id"]), only.get("text", "")))}

    mid = len(items) // 2
    left = translate_items_recursive(
        api_key=api_key, model=model, target_lang=target_lang, items=items[:mid],
        temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s,
        app_url=app_url, app_title=app_title, response_format=response_format, attempts=attempts
    )
    right = translate_items_recursive(
        api_key=api_key, model=model, target_lang=target_lang, items=items[mid:],
        temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s,
        app_url=app_url, app_title=app_title, response_format=response_format, attempts=attempts
    )
    left.update(right)
    return left


def build_items_with_context(
    segments: List[Segment],
    *,
    strip_translator_tag: bool,
    context_window: int,
) -> List[Dict[str, object]]:
    timed: List[Tuple[int, str]] = []
    for s in segments:
        if not s.timestamp:
            continue
        txt = segment_text(s, strip_translator_tag)
        if txt:
            timed.append((s.num, txt))

    items: List[Dict[str, object]] = []
    n = len(timed)
    for idx, (sid, txt) in enumerate(timed):
        prev_ctx: List[str] = []
        next_ctx: List[str] = []
        if context_window > 0:
            for j in range(max(0, idx - context_window), idx):
                prev_ctx.append(timed[j][1])
            for j in range(idx + 1, min(n, idx + 1 + context_window)):
                next_ctx.append(timed[j][1])

        it: Dict[str, object] = {"id": str(sid), "text": txt}
        if context_window > 0:
            it["context_prev"] = prev_ctx
            it["context_next"] = next_ctx
        items.append(it)
    return items


def apply_translations(
    segments: List[Segment],
    translated: Dict[str, str],
    strip_translator_tag: bool,
) -> List[Segment]:
    out: List[Segment] = []
    for s in segments:
        if not s.timestamp:
            out.append(s)
            continue

        key = str(s.num)
        orig_lines = normalize_lines(s.lines, strip_translator_tag)

        if key in translated:
            new_lines = str(translated[key]).split("\n")
            out.append(Segment(num=s.num, timestamp=s.timestamp, lines=new_lines))
        else:
            out.append(Segment(num=s.num, timestamp=s.timestamp, lines=orig_lines))
    return out


def iter_input_files(inputs: List[str], recursive: bool) -> List[Path]:
    paths = [Path(p).expanduser() for p in inputs]
    files: List[Path] = []
    for p in paths:
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.srt") if recursive else p.glob("*.srt")))
        else:
            raise FileNotFoundError(str(p))
    uniq: Dict[str, Path] = {str(f.resolve()): f for f in files}
    return list(uniq.values())


def extract_tags(text: str) -> List[str]:
    return TAG_RE.findall(text)


def strip_leading_tags(line: str) -> str:
    line = LEADING_TAGS_RE.sub("", line)
    return line.lstrip()


def detect_speaker_label(text: str) -> Optional[str]:
    first = text.split("\n", 1)[0].strip()
    first = strip_leading_tags(first)
    m = SPEAKER_RE.match(first)
    return m.group(1) + ":" if m else None


def thai_ratio(text: str) -> float:
    thai = sum(1 for ch in text if "\u0E00" <= ch <= "\u0E7F")
    letters = sum(1 for ch in text if ch.isalpha())
    return thai / max(letters, 1)


def ascii_alpha_ratio(text: str) -> float:
    ascii_alpha = sum(1 for ch in text if ch.isalpha() and ch.isascii())
    letters = sum(1 for ch in text if ch.isalpha())
    return ascii_alpha / max(letters, 1)


def validate_translation(
    *,
    original_segments: List[Segment],
    translated_segments: List[Segment],
    translated_ids: set[str],
    target_lang: str,
    strip_translator_tag: bool,
) -> Dict[str, object]:
    orig_map: Dict[str, str] = {}
    orig_linecount: Dict[str, int] = {}
    for s in original_segments:
        if not s.timestamp:
            continue
        txt = segment_text(s, strip_translator_tag)
        if not txt:
            continue
        sid = str(s.num)
        orig_map[sid] = txt
        orig_linecount[sid] = len(txt.split("\n"))

    trans_map: Dict[str, str] = {}
    trans_linecount: Dict[str, int] = {}
    for s in translated_segments:
        if not s.timestamp:
            continue
        txt = "\n".join(s.lines).strip()
        if not txt:
            continue
        sid = str(s.num)
        trans_map[sid] = txt
        trans_linecount[sid] = len(txt.split("\n"))

    issues: List[Dict[str, object]] = []

    expected_ids = set(orig_map.keys())
    missing = sorted(expected_ids - set(translated_ids), key=lambda x: int(x))
    if missing:
        issues.append({"type": "missing_model_outputs", "count": len(missing), "ids": missing[:50]})

    for sid, src in orig_map.items():
        dst = trans_map.get(sid, "")
        if not dst:
            continue

        if extract_tags(src) != extract_tags(dst):
            issues.append({"id": sid, "type": "tag_mismatch"})

        src_label = detect_speaker_label(src)
        if src_label:
            dst_label = detect_speaker_label(dst)
            if dst_label != src_label:
                issues.append({"id": sid, "type": "speaker_label_changed", "expected": src_label, "got": dst_label})

        if orig_linecount.get(sid) != trans_linecount.get(sid):
            issues.append({"id": sid, "type": "linebreak_count_changed", "expected": orig_linecount.get(sid), "got": trans_linecount.get(sid)})

        if target_lang == "th":
            src_ascii_alpha = sum(1 for ch in src if ch.isalpha() and ch.isascii())
            if src_ascii_alpha >= 15:
                tr_th = thai_ratio(dst)
                tr_ascii = ascii_alpha_ratio(dst)
                if tr_th < 0.20 and tr_ascii > 0.60:
                    issues.append({"id": sid, "type": "likely_untranslated", "thai_ratio": round(tr_th, 3), "ascii_alpha_ratio": round(tr_ascii, 3)})

    ok = (len([i for i in issues if i.get("type") != "missing_model_outputs"]) == 0) and (not missing)
    return {"ok": ok, "issues": issues, "total_segments": len(orig_map), "expected_translations": len(expected_ids)}


def qc_judge_batch(
    *,
    api_key: str,
    model: str,
    target_lang: str,
    batch: List[Dict[str, object]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    app_url: str,
    app_title: str,
    response_format: Optional[dict],
    attempts: int,
) -> Dict[str, Dict[str, object]]:
    target_name = TARGET_LANG_NAMES.get(target_lang, target_lang)
    req_obj = {"target_language": target_lang, "segments": batch}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_QC.format(target_lang_name=target_name)},
        {"role": "user", "content": json.dumps(req_obj, ensure_ascii=False)},
    ]

    last_err: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            resp = openrouter_chat(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                app_url=app_url,
                app_title=app_title,
                response_format=response_format,
            )
            content = resp["choices"][0]["message"]["content"]
            obj = safe_json_loads(content)
            segs = obj.get("segments", [])
            out: Dict[str, Dict[str, object]] = {}
            for s in segs:
                sid = str(s.get("id"))
                out[sid] = {
                    "score": int(s.get("score", 0) or 0),
                    "issues": s.get("issues", []),
                    "notes": s.get("notes", ""),
                }
            return out
        except Exception as e:
            last_err = e
            time.sleep(min(10.0, 0.8 * (2 ** attempt)))

    raise last_err or RuntimeError("qc failed")


def select_qc_ids(all_ids: List[str], flagged_ids: List[str], qc_limit: int) -> List[str]:
    seen = set()
    selected: List[str] = []

    for sid in flagged_ids:
        if sid not in seen:
            selected.append(sid)
            seen.add(sid)

    if qc_limit == 0:
        for sid in all_ids:
            if sid not in seen:
                selected.append(sid)
                seen.add(sid)
        return selected

    if len(selected) >= qc_limit:
        return selected[:qc_limit]

    remaining = qc_limit - len(selected)
    pool = [sid for sid in all_ids if sid not in seen]
    if not pool:
        return selected

    step = max(1, len(pool) // remaining)
    idx = 0
    while len(selected) < qc_limit and idx < len(pool):
        selected.append(pool[idx])
        idx += step
    return selected


def run_llm_qc(
    *,
    api_key: str,
    judge_model: str,
    target_lang: str,
    original_segments: List[Segment],
    translated_segments: List[Segment],
    strip_translator_tag: bool,
    qc_limit: int,
    flagged_ids: List[str],
    max_chars: int,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    app_url: str,
    app_title: str,
    response_format: Optional[dict],
    attempts: int,
) -> Dict[str, object]:
    src_map: Dict[str, str] = {}
    for s in original_segments:
        if not s.timestamp:
            continue
        txt = segment_text(s, strip_translator_tag)
        if txt:
            src_map[str(s.num)] = txt

    dst_map: Dict[str, str] = {}
    for s in translated_segments:
        if not s.timestamp:
            continue
        txt = "\n".join(s.lines).strip()
        if txt:
            dst_map[str(s.num)] = txt

    all_ids = sorted(set(src_map.keys()) & set(dst_map.keys()), key=lambda x: int(x))
    chosen_ids = select_qc_ids(all_ids, flagged_ids, qc_limit)

    id_to_pos = {sid: i for i, sid in enumerate(all_ids)}
    qc_items: List[Dict[str, object]] = []

    for sid in chosen_ids:
        pos = id_to_pos.get(sid, 0)
        prev_ctx = [src_map[all_ids[j]] for j in range(max(0, pos - 2), pos)]
        next_ctx = [src_map[all_ids[j]] for j in range(pos + 1, min(len(all_ids), pos + 3))]
        qc_items.append({
            "id": sid,
            "source": src_map[sid],
            "translation": dst_map[sid],
            "context_prev": prev_ctx,
            "context_next": next_ctx,
        })

    results: Dict[str, Dict[str, object]] = {}
    for batch in chunk_by_chars(qc_items, max_chars=max_chars):
        out = qc_judge_batch(
            api_key=api_key,
            model=judge_model,
            target_lang=target_lang,
            batch=batch,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            app_url=app_url,
            app_title=app_title,
            response_format=response_format,
            attempts=attempts,
        )
        results.update(out)

    scores = [v["score"] for v in results.values() if isinstance(v.get("score"), int) and v["score"] > 0]
    avg = sum(scores) / max(len(scores), 1)

    return {
        "average_score": round(avg, 3),
        "qc_count": len(results),
        "qc_limit": qc_limit,
        "by_score": {str(s): sum(1 for v in results.values() if v.get("score") == s) for s in range(1, 6)},
        "segments": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Translate .srt subtitles using OpenRouter (+ context + validation/QC).")
    ap.add_argument("inputs", nargs="+", help="Input .srt file(s) or directories")
    ap.add_argument("-t", "--target-lang", default="th", help="Target language code (default: th)")

    ap.add_argument("--model", default=os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview"),
                    help="Model id for translation (default: google/gemini-3-flash-preview)")
    ap.add_argument("--judge-model", default=os.getenv("OPENROUTER_JUDGE_MODEL", "google/gemini-3-flash-preview"),
                    help="Model id for QC (default: google/gemini-3-flash-preview)")

    ap.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"),
                    help="OpenRouter API key (or env OPENROUTER_API_KEY)")

    ap.add_argument("--max-chars", type=int, default=8000, help="Max approx characters per request (default: 8000)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="max_tokens for translation output (default: 4096)")
    ap.add_argument("--qc-max-tokens", type=int, default=4096, help="max_tokens for QC output (default: 4096)")

    ap.add_argument("--temperature", type=float, default=0.0, help="Translation temperature (default: 0.0)")
    ap.add_argument("--qc-temperature", type=float, default=0.0, help="QC temperature (default: 0.0)")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds (default: 120)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    ap.add_argument("--recursive", action="store_true", help="If an input is a directory, recurse into subdirs")
    ap.add_argument("--strip-translator-tag", action="store_true",
                    help="Remove lines like '# Subtitles translated with ... #' before translating")

    ap.add_argument("--context-window", type=int, default=2,
                    help="Include N previous/next segments as context for each segment (default: 2)")
    ap.add_argument("--validate", action="store_true", help="Run heuristic validation after translation")
    ap.add_argument("--llm-qc", action="store_true", help="Run judge-model QC and emit report JSON")
    ap.add_argument("--qc-limit", type=int, default=250,
                    help="How many segments to QC (0 = all). Default: 250. Prioritizes flagged segments first.")

    ap.add_argument("--app-url", default=os.getenv("OPENROUTER_APP_URL", ""), help="Optional HTTP-Referer header value")
    ap.add_argument("--app-title", default=os.getenv("OPENROUTER_APP_TITLE", "srt-translator"), help="Optional X-Title header value")
    ap.add_argument("--json-mode", action="store_true",
                    help="Send response_format={type:'json_object'} (recommended for reliable parsing)")
    ap.add_argument("--attempts", type=int, default=5, help="Retry attempts per request (default: 5)")

    args = ap.parse_args()

    if not args.api_key:
        print("Missing API key. Set OPENROUTER_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    in_files = iter_input_files(args.inputs, recursive=args.recursive)
    if not in_files:
        print("No .srt files found.", file=sys.stderr)
        return 1

    response_format = {"type": "json_object"} if args.json_mode else None

    for in_path in in_files:
        out_name = swap_lang_in_filename(in_path.name, args.target_lang)
        out_path = in_path.with_name(out_name)

        if out_path.exists() and not args.overwrite:
            print(f"SKIP exists: {out_path}")
            continue

        src_text = read_text_with_fallback(in_path)
        src_segments = parse_srt(src_text)

        items = build_items_with_context(
            src_segments,
            strip_translator_tag=args.strip_translator_tag,
            context_window=max(0, args.context_window),
        )

        translated_all: Dict[str, str] = {}
        batches = list(chunk_by_chars(items, max_chars=args.max_chars))
        for bi, batch in enumerate(batches, start=1):
            translated = translate_items_recursive(
                api_key=args.api_key,
                model=args.model,
                target_lang=args.target_lang,
                items=batch,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
                app_url=args.app_url,
                app_title=args.app_title,
                response_format=response_format,
                attempts=args.attempts,
            )
            translated_all.update(translated)
            print(f"{in_path.name}: batch {bi}/{len(batches)} done, total {len(translated_all)}/{len(items)}")

        out_segments = apply_translations(src_segments, translated_all, strip_translator_tag=args.strip_translator_tag)
        out_text = format_srt(out_segments)
        out_path.write_text(out_text, encoding="utf-8")
        print(f"WROTE: {out_path}")

        heur = None
        flagged_ids: List[str] = []
        if args.validate or args.llm_qc:
            heur = validate_translation(
                original_segments=src_segments,
                translated_segments=out_segments,
                translated_ids=set(translated_all.keys()),
                target_lang=args.target_lang,
                strip_translator_tag=args.strip_translator_tag,
            )
            for it in heur["issues"]:
                if isinstance(it, dict) and "id" in it:
                    flagged_ids.append(str(it["id"]))

        if args.validate and heur is not None:
            print(f"VALIDATE: ok={heur['ok']} total={heur['total_segments']} expected_translations={heur['expected_translations']}")
            if not heur["ok"]:
                for it in heur["issues"][:25]:
                    print("  ISSUE:", it)

        if args.llm_qc:
            qc = run_llm_qc(
                api_key=args.api_key,
                judge_model=args.judge_model,
                target_lang=args.target_lang,
                original_segments=src_segments,
                translated_segments=out_segments,
                strip_translator_tag=args.strip_translator_tag,
                qc_limit=max(0, args.qc_limit),
                flagged_ids=flagged_ids,
                max_chars=args.max_chars,
                temperature=args.qc_temperature,
                max_tokens=args.qc_max_tokens,
                timeout_s=args.timeout,
                app_url=args.app_url,
                app_title=args.app_title,
                response_format=response_format,
                attempts=args.attempts,
            )
            qc_path = out_path.with_suffix(out_path.suffix + ".qc.json")
            qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"QC: avg_score={qc['average_score']} by_score={qc['by_score']} report={qc_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
