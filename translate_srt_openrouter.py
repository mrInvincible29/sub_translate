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
import concurrent.futures
import threading
import subprocess
import shutil
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
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
SEGMENT_PAIR_RE = re.compile(
    r'"id"\s*:\s*(?:"([^"]+)"|(\d+))\s*,\s*"text"\s*:\s*"((?:\\.|[^"\\])*)"',
    re.DOTALL,
)


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


def extract_first_json_object(s: str) -> Optional[str]:
    start = s.find("{")
    if start < 0:
        return None

    in_str = False
    escape = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]

    return None


def sanitize_json_text(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = CODE_FENCE_RE.sub("", s).strip()
    return s


def repair_json_text(s: str) -> str:
    # Remove trailing commas before closing braces/brackets.
    return re.sub(r",(\s*[}\]])", r"\1", s)


def safe_json_loads(s: str) -> dict:
    s = sanitize_json_text(s)
    first_err: Optional[Exception] = None
    try:
        return json.loads(s)
    except json.JSONDecodeError as err:
        first_err = err

    candidate = extract_first_json_object(s)
    if not candidate:
        if first_err is not None:
            raise first_err
        raise json.JSONDecodeError("No JSON object found", s, 0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = repair_json_text(candidate)
        return json.loads(repaired)


def extract_segments_best_effort(content: str) -> Dict[str, str]:
    # Best-effort parse of id/text pairs from malformed JSON.
    out: Dict[str, str] = {}
    for m in SEGMENT_PAIR_RE.finditer(content):
        sid = m.group(1) or m.group(2) or ""
        raw_text = m.group(3) or ""
        if not sid:
            continue
        try:
            text = json.loads(f'"{raw_text}"')
        except json.JSONDecodeError:
            text = raw_text.replace('\\"', '"').replace("\\n", "\n")
        out[str(sid)] = text
    return out


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
    progress_label: str,
    progress_state: Optional[Dict[str, object]],
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
            if progress_state is None or progress_state.get("verbose"):
                print(f"{progress_label}: request attempt {attempt + 1}/{attempts}...", flush=True)
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
            try:
                obj = safe_json_loads(content)
                segs = obj.get("segments", [])
                out: Dict[str, str] = {}
                for s in segs:
                    sid = str(s.get("id"))
                    out[sid] = s.get("text", "")
                if progress_state is not None:
                    progress_state["last_json_error"] = False
                    progress_state["last_salvaged"] = False
                return out
            except json.JSONDecodeError as e:
                if progress_state is not None:
                    progress_state["last_json_error"] = True
                best = extract_segments_best_effort(content or "")
                if best:
                    if progress_state is not None:
                        progress_state["last_salvaged"] = True
                    if progress_state is None or progress_state.get("verbose"):
                        print(
                            f"{progress_label}: warning: response JSON malformed; salvaged {len(best)} segments",
                            file=sys.stderr,
                            flush=True,
                        )
                    return best
                snippet = (content or "").replace("\n", " ")[:160]
                raise json.JSONDecodeError(
                    f"{e.msg} (content_len={len(content or '')}, snippet={snippet!r})",
                    e.doc,
                    e.pos,
                ) from e
        except json.JSONDecodeError as e:
            # JSON parse errors usually mean truncated/invalid model output; split batch instead of retrying.
            print(f"{progress_label}: attempt {attempt + 1} failed: {e}", file=sys.stderr, flush=True)
            raise
        except Exception as e:
            last_err = e
            print(f"{progress_label}: attempt {attempt + 1} failed: {e}", file=sys.stderr, flush=True)
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
    progress_label: str,
    depth: int = 0,
    progress_state: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    if not items:
        return {}

    if progress_state is None:
        progress_state = {}

    if progress_state.get("adaptive_limits"):
        cur_max_chars = int(progress_state.get("max_chars", 0) or 0)
        if cur_max_chars > 0 and len(items) > 1:
            total_chars = sum(item_size_chars(it) for it in items)
            if total_chars > cur_max_chars:
                chunks = list(chunk_by_chars(items, max_chars=cur_max_chars))
                out: Dict[str, str] = {}
                for i, chunk in enumerate(chunks, start=1):
                    sub_label = f"{progress_label} [chunk {i}/{len(chunks)}]"
                    got = translate_items_recursive(
                        api_key=api_key,
                        model=model,
                        target_lang=target_lang,
                        items=chunk,
                        temperature=temperature,
                        max_tokens=int(progress_state.get("max_tokens", max_tokens) or max_tokens),
                        timeout_s=timeout_s,
                        app_url=app_url,
                        app_title=app_title,
                        response_format=response_format,
                        attempts=attempts,
                        progress_label=sub_label,
                        depth=depth,
                        progress_state=progress_state,
                    )
                    out.update(got)
                return out

    try:
        got = translate_batch(
            api_key=api_key,
            model=model,
            target_lang=target_lang,
            batch=items,
            temperature=temperature,
            max_tokens=int(progress_state.get("max_tokens", max_tokens) or max_tokens),
            timeout_s=timeout_s,
            app_url=app_url,
            app_title=app_title,
            response_format=response_format,
            attempts=attempts,
            progress_label=progress_label,
            progress_state=progress_state,
        )
    except json.JSONDecodeError:
        if len(items) == 1:
            raise
        if not progress_state.get("split_explain_printed") and progress_state.get("verbose"):
            print(
                f"{progress_label}: response JSON invalid; splitting batch into smaller requests",
                file=sys.stderr,
                flush=True,
            )
            progress_state["split_explain_printed"] = True
        adjust_adaptive_limits(
            progress_state,
            direction="down",
            reason="json_error",
            label=progress_label,
        )
        mid = len(items) // 2
        left = translate_items_recursive(
            api_key=api_key, model=model, target_lang=target_lang, items=items[:mid],
            temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s,
            app_url=app_url, app_title=app_title, response_format=response_format, attempts=attempts,
            progress_label=f"{progress_label} [split L]",
            depth=depth + 1,
            progress_state=progress_state,
        )
        right = translate_items_recursive(
            api_key=api_key, model=model, target_lang=target_lang, items=items[mid:],
            temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s,
            app_url=app_url, app_title=app_title, response_format=response_format, attempts=attempts,
            progress_label=f"{progress_label} [split R]",
            depth=depth + 1,
            progress_state=progress_state,
        )
        left.update(right)
        return left

    want_ids = {str(it["id"]) for it in items}
    got_ids = set(got.keys())
    if want_ids.issubset(got_ids) and all(got.get(i, "") != "" for i in want_ids):
        if progress_state.get("adaptive_limits"):
            progress_state["stable_ok"] = int(progress_state.get("stable_ok", 0) or 0) + 1
            adjust_adaptive_limits(
                progress_state,
                direction="up",
                reason="stable_ok",
                label=progress_label,
            )
        return got

    if got and got_ids:
        missing = [it for it in items if str(it["id"]) not in got_ids or not got.get(str(it["id"]), "")]
        if missing:
            missing_ids = [str(it["id"]) for it in missing]
            sample = ", ".join(missing_ids[:5])
            total = len(items)
            got_count = len(got_ids)
            if not progress_state.get("missing_explain_printed") and progress_state.get("verbose"):
                print(
                    f"{progress_label}: model did not return all requested segments; "
                    f"we re-request only the missing segment ids (ids are SRT block numbers)",
                    file=sys.stderr,
                    flush=True,
                )
                progress_state["missing_explain_printed"] = True
            if progress_state.get("verbose"):
                print(
                    f"{progress_label}: requested {total} segments, got {got_count}, "
                    f"missing {len(missing_ids)} (e.g. ids {sample})",
                    file=sys.stderr,
                    flush=True,
                )
            adjust_adaptive_limits(
                progress_state,
                direction="down",
                reason="missing_segments",
                label=progress_label,
            )
            rest = translate_items_recursive(
                api_key=api_key,
                model=model,
                target_lang=target_lang,
                items=missing,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                app_url=app_url,
                app_title=app_title,
                response_format=response_format,
                attempts=attempts,
                progress_label=(
                    f"{progress_label} [retry depth={depth + 1} "
                    f"missing {len(missing)}/{len(items)}]"
                ),
                depth=depth + 1,
                progress_state=progress_state,
            )
            got.update(rest)
        if progress_state.get("last_salvaged"):
            adjust_adaptive_limits(
                progress_state,
                direction="down",
                reason="salvaged_json",
                label=progress_label,
            )
            progress_state["last_salvaged"] = False
        return got

    if len(items) == 1:
        only = items[0]
        return {str(only["id"]): str(got.get(str(only["id"]), only.get("text", "")))}

    mid = len(items) // 2
    left = translate_items_recursive(
        api_key=api_key, model=model, target_lang=target_lang, items=items[:mid],
        temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s,
        app_url=app_url, app_title=app_title, response_format=response_format, attempts=attempts,
        progress_label=f"{progress_label} [split L]",
        depth=depth + 1,
        progress_state=progress_state,
    )
    right = translate_items_recursive(
        api_key=api_key, model=model, target_lang=target_lang, items=items[mid:],
        temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s,
        app_url=app_url, app_title=app_title, response_format=response_format, attempts=attempts,
        progress_label=f"{progress_label} [split R]",
        depth=depth + 1,
        progress_state=progress_state,
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


def build_items_map_with_context(
    segments: List[Segment],
    *,
    strip_translator_tag: bool,
    context_window: int,
) -> Dict[str, Dict[str, object]]:
    items = build_items_with_context(
        segments,
        strip_translator_tag=strip_translator_tag,
        context_window=context_window,
    )
    return {str(it["id"]): it for it in items}


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


def load_progress(path: Path) -> Dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict) and isinstance(data.get("translated"), dict):
        return {str(k): str(v) for k, v in data["translated"].items()}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def save_progress(path: Path, translated: Dict[str, str]) -> None:
    payload = {"translated": translated}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def render_progress(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[----------] 0/0 (0%)"
    filled = int(width * done / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    pct = int(100 * done / max(total, 1))
    return f"[{bar}] {done}/{total} ({pct}%)"


def progress_line(prefix: str, done: int, total: int, width: int = 30) -> str:
    return f"{prefix} {render_progress(done, total, width)}"


def colorize(text: str, kind: str, use_color: bool) -> str:
    if not use_color:
        return text
    codes = {
        "info": "\033[36m",
        "ok": "\033[32m",
        "warn": "\033[33m",
        "err": "\033[31m",
        "reset": "\033[0m",
    }
    return f"{codes.get(kind, '')}{text}{codes['reset']}"


def adjust_adaptive_limits(
    progress_state: Dict[str, object],
    *,
    direction: str,
    reason: str,
    label: str,
) -> None:
    if not progress_state.get("adaptive_limits"):
        return
    lock: Optional[threading.Lock] = progress_state.get("lock")
    if lock:
        lock.acquire()
    try:
        max_chars = int(progress_state.get("max_chars", 0) or 0)
        max_tokens = int(progress_state.get("max_tokens", 0) or 0)
        min_chars = int(progress_state.get("min_chars", 0) or 0)
        min_tokens = int(progress_state.get("min_tokens", 0) or 0)
        max_chars_cap = int(progress_state.get("max_chars_cap", max_chars) or max_chars)
        max_tokens_cap = int(progress_state.get("max_tokens_cap", max_tokens) or max_tokens)
        stable = int(progress_state.get("stable_ok", 0) or 0)

        new_chars = max_chars
        new_tokens = max_tokens
        if direction == "down":
            new_chars = max(min_chars, int(max_chars * 0.8))
            new_tokens = max(min_tokens, int(max_tokens * 0.85))
            stable = 0
        elif direction == "up" and stable >= 3:
            new_chars = min(max_chars_cap, int(max_chars * 1.1))
            new_tokens = min(max_tokens_cap, int(max_tokens * 1.1))
            stable = 0

        if new_chars != max_chars or new_tokens != max_tokens:
            progress_state["max_chars"] = new_chars
            progress_state["max_tokens"] = new_tokens
            if progress_state.get("verbose"):
                print(
                    f"{label}: ADAPT {direction} -> max_chars={new_chars} max_tokens={new_tokens} "
                    f"(reason: {reason})",
                    file=sys.stderr,
                    flush=True,
                )
        progress_state["stable_ok"] = stable
    finally:
        if lock:
            lock.release()


def force_linebreaks(text: str, target_lines: int) -> str:
    if target_lines <= 1:
        return " ".join(text.splitlines()).strip()
    plain = " ".join(text.splitlines()).strip()
    if not plain:
        return text
    total_len = len(plain)
    chunk = max(1, total_len // target_lines)
    lines: List[str] = []
    idx = 0
    for i in range(target_lines - 1):
        cut = min(total_len, idx + chunk)
        # Try to cut on a nearby space to avoid mid-word splits.
        lo = max(idx + 1, cut - 10)
        hi = min(total_len, cut + 10)
        space = plain.rfind(" ", lo, hi)
        if space == -1:
            space = cut
        lines.append(plain[idx:space].strip())
        idx = space + 1 if space < total_len else total_len
    lines.append(plain[idx:].strip())
    return "\n".join(line for line in lines if line != "")


def apply_translations_with_linebreaks(
    segments: List[Segment],
    translated: Dict[str, str],
    strip_translator_tag: bool,
    enforce_ids: set[str],
) -> List[Segment]:
    out: List[Segment] = []
    for s in segments:
        if not s.timestamp:
            out.append(s)
            continue

        key = str(s.num)
        orig_lines = normalize_lines(s.lines, strip_translator_tag)

        if key in translated:
            text = str(translated[key])
            if key in enforce_ids:
                text = force_linebreaks(text, max(1, len(orig_lines)))
            new_lines = text.split("\n")
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


def is_english_srt(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".srt") and (".en." in name or name.endswith(".en.srt"))


def pick_english_srt(dir_path: Path) -> List[Path]:
    all_srts = sorted(dir_path.glob("*.srt"))
    en = [p for p in all_srts if is_english_srt(p)]
    return en


def list_video_files(dir_path: Path) -> List[Path]:
    exts = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".ts", ".m2ts"}
    return [p for p in sorted(dir_path.iterdir()) if p.suffix.lower() in exts and p.is_file()]


def require_ffmpeg_tools() -> None:
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")
    if not shutil.which("ffprobe"):
        missing.append("ffprobe")
    if missing:
        hint = "Install with: sudo apt install ffmpeg"
        raise FileNotFoundError(f"Missing tools: {', '.join(missing)}. {hint}")


def ffprobe_subtitle_streams(video_path: Path) -> List[Dict[str, str]]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "s",
        "-show_entries", "stream=index:stream_tags=language,title:stream=codec_name",
        "-of", "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout or "{}")
    streams = []
    for s in data.get("streams", []):
        tags = s.get("tags") or {}
        streams.append({
            "index": str(s.get("index")),
            "language": str(tags.get("language", "")),
            "title": str(tags.get("title", "")),
            "codec": str(s.get("codec_name", "")),
        })
    return streams


def pick_stream_interactive(streams: List[Dict[str, str]], prompt: str) -> Dict[str, str]:
    for i, s in enumerate(streams, start=1):
        lang = s.get("language", "")
        title = s.get("title", "")
        codec = s.get("codec", "")
        print(f"{i}) index={s['index']} lang={lang} codec={codec} title={title}")
    while True:
        sel = input(prompt).strip()
        if not sel.isdigit():
            print("Enter a number from the list.")
            continue
        idx = int(sel)
        if 1 <= idx <= len(streams):
            return streams[idx - 1]
        print("Invalid selection.")


def extract_english_srt_from_dir(dir_path: Path) -> Path:
    require_ffmpeg_tools()
    videos = list_video_files(dir_path)
    if not videos:
        raise FileNotFoundError(f"No video files found in {dir_path}")
    if len(videos) > 1:
        print("Multiple video files found:")
        for i, v in enumerate(videos, start=1):
            print(f"{i}) {v.name}")
        while True:
            sel = input("Select video to extract EN subtitles from: ").strip()
            if not sel.isdigit():
                print("Enter a number from the list.")
                continue
            idx = int(sel)
            if 1 <= idx <= len(videos):
                video = videos[idx - 1]
                break
            print("Invalid selection.")
    else:
        video = videos[0]

    streams = ffprobe_subtitle_streams(video)
    en_streams = [s for s in streams if s.get("language", "").lower() in {"en", "eng"}]
    if not en_streams:
        raise FileNotFoundError(f"No English subtitle streams found in {video.name}")
    if len(en_streams) > 1:
        print("Multiple English subtitle streams found:")
        chosen = pick_stream_interactive(en_streams, "Select EN subtitle stream: ")
    else:
        chosen = en_streams[0]

    out_path = dir_path / (video.stem + ".en.srt")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-map", f"0:{chosen['index']}",
        "-c:s", "srt",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def resolve_tmdb_input(tmdb_id: str, movies_root: Path) -> List[Path]:
    if not tmdb_id.isdigit():
        raise FileNotFoundError(tmdb_id)
    tag = f"{{tmdb-{tmdb_id}}}"
    candidates = [p for p in movies_root.glob(f"*{tag}*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No directory with {tag} under {movies_root}")
    # Prefer exact match if multiple, otherwise first.
    target_dir = sorted(candidates)[0]
    en = pick_english_srt(target_dir)
    if not en:
        extracted = extract_english_srt_from_dir(target_dir)
        return [extracted]
    return en


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
            try:
                obj = safe_json_loads(content)
            except json.JSONDecodeError as e:
                snippet = (content or "").replace("\n", " ")[:160]
                raise json.JSONDecodeError(
                    f"{e.msg} (content_len={len(content or '')}, snippet={snippet!r})",
                    e.doc,
                    e.pos,
                ) from e
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
    ap.add_argument("inputs", nargs="+",
                    help="Input .srt file(s), directories, or a tmdb id (e.g. 1197137)")
    ap.add_argument("-t", "--target-lang", default="th", help="Target language code (default: th)")

    ap.add_argument("--model", default=os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview"),
                    help="Model id for translation (default: google/gemini-3-flash-preview)")
    ap.add_argument("--judge-model", default=os.getenv("OPENROUTER_JUDGE_MODEL", "google/gemini-3-flash-preview"),
                    help="Model id for QC (default: google/gemini-3-flash-preview)")

    ap.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"),
                    help="OpenRouter API key (or env OPENROUTER_API_KEY)")

    ap.add_argument("--max-chars", type=int, default=2000, help="Max approx characters per request (default: 2000)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="max_tokens for translation output (default: 1024)")
    ap.add_argument("--qc-max-tokens", type=int, default=4096, help="max_tokens for QC output (default: 4096)")

    ap.add_argument("--temperature", type=float, default=0.0, help="Translation temperature (default: 0.0)")
    ap.add_argument("--qc-temperature", type=float, default=0.0, help="QC temperature (default: 0.0)")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds (default: 120)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    ap.add_argument("--overwrite-translated", action="store_true",
                    help="Overwrite existing translated .srt (alias for --overwrite)")
    ap.add_argument("--recursive", action="store_true", help="If an input is a directory, recurse into subdirs")
    ap.add_argument("--strip-translator-tag", action="store_true", default=True,
                    help="Remove lines like '# Subtitles translated with ... #' before translating (default: on)")
    ap.add_argument("--no-strip-translator-tag", dest="strip_translator_tag", action="store_false",
                    help="Do not strip translator tag lines")

    ap.add_argument("--context-window", type=int, default=0,
                    help="Include N previous/next segments as context for each segment (default: 0)")
    ap.add_argument("--validate", action="store_true", default=True,
                    help="Run heuristic validation after translation (default: on)")
    ap.add_argument("--no-validate", dest="validate", action="store_false",
                    help="Disable heuristic validation")
    ap.add_argument("--llm-qc", action="store_true", help="Run judge-model QC and emit report JSON")
    ap.add_argument("--qc-limit", type=int, default=250,
                    help="How many segments to QC (0 = all). Default: 250. Prioritizes flagged segments first.")
    ap.add_argument("--auto-fix", action="store_true", default=True,
                    help="Auto-fix flagged segments (retranslate + linebreak enforcement) (default: on)")
    ap.add_argument("--no-auto-fix", dest="auto_fix", action="store_false",
                    help="Disable auto-fix")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Resume from saved progress file to avoid re-translating (default: on)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Disable resume/progress caching")

    ap.add_argument("--app-url", default=os.getenv("OPENROUTER_APP_URL", ""), help="Optional HTTP-Referer header value")
    ap.add_argument("--app-title", default=os.getenv("OPENROUTER_APP_TITLE", "srt-translator"), help="Optional X-Title header value")
    ap.add_argument("--json-mode", action="store_true", default=True,
                    help="Send response_format={type:'json_object'} (default: on)")
    ap.add_argument("--no-json-mode", dest="json_mode", action="store_false",
                    help="Disable response_format json_object")
    ap.add_argument("--attempts", type=int, default=5, help="Retry attempts per request (default: 5)")
    ap.add_argument("--parallel", type=int, default=4,
                    help="Parallel batch requests (default: 4)")
    ap.add_argument("--progress-bar", action="store_true", default=True,
                    help="Show progress bar output (default: on)")
    ap.add_argument("--no-progress-bar", dest="progress_bar", action="store_false",
                    help="Disable progress bar output")
    ap.add_argument("--verbose", action="store_true", default=False,
                    help="Verbose logging (default: off)")
    ap.add_argument("--adaptive-limits", action="store_true", default=False,
                    help="Auto-tune max-chars/max-tokens based on response quality (default: off)")
    ap.add_argument("--no-adaptive-limits", dest="adaptive_limits", action="store_false",
                    help="Disable auto-tuning of limits")
    ap.add_argument("--color", action="store_true", default=True,
                    help="Colorize logs (default: on)")
    ap.add_argument("--no-color", dest="color", action="store_false",
                    help="Disable colored logs")
    ap.add_argument("--movies-root", default="/home/h2/media/Movies",
                    help="Root folder to search when input is a tmdb id (default: /home/h2/media/Movies)")

    args = ap.parse_args()

    if not args.api_key:
        print("Missing API key. Set OPENROUTER_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    argv = set(sys.argv[1:])
    if args.model == "openai/gpt-4o-mini":
        if "--max-chars" not in argv:
            args.max_chars = max(args.max_chars, 4000)
        if "--max-tokens" not in argv:
            args.max_tokens = max(args.max_tokens, 1536)

    inputs: List[str] = []
    movies_root = Path(args.movies_root).expanduser()
    for raw in args.inputs:
        if raw.isdigit():
            inputs.extend(str(p) for p in resolve_tmdb_input(raw, movies_root))
        else:
            p = Path(raw).expanduser()
            if p.is_dir():
                en = pick_english_srt(p)
                if not en:
                    extracted = extract_english_srt_from_dir(p)
                    inputs.append(str(extracted))
                else:
                    inputs.append(raw)
            else:
                inputs.append(raw)

    in_files = iter_input_files(inputs, recursive=args.recursive)
    if not in_files:
        print("No .srt files found.", file=sys.stderr)
        return 1

    response_format = {"type": "json_object"} if args.json_mode else None

    for in_path in in_files:
        out_name = swap_lang_in_filename(in_path.name, args.target_lang)
        out_path = in_path.with_name(out_name)

        if out_path.exists() and not (args.overwrite or args.overwrite_translated):
            print(f"SKIP exists: {out_path}")
            continue

        src_text = read_text_with_fallback(in_path)
        src_segments = parse_srt(src_text)

        items_full = build_items_with_context(
            src_segments,
            strip_translator_tag=args.strip_translator_tag,
            context_window=max(0, args.context_window),
        )

        progress_path = out_path.with_suffix(out_path.suffix + ".progress.json")
        translated_all: Dict[str, str] = {}
        if args.resume and progress_path.exists():
            translated_all = load_progress(progress_path)
            if translated_all:
                print(colorize(f"RESUME: loaded {len(translated_all)} segments from {progress_path}", "info", args.color))

        items = [it for it in items_full if str(it["id"]) not in translated_all or not translated_all.get(str(it["id"]))] 
        if not items:
            print(colorize(f"SKIP all translated: {in_path.name}", "info", args.color))
        else:
            print(colorize(f"{in_path.name}: remaining {len(items)}/{len(items_full)} segments", "info", args.color))

        batches = list(chunk_by_chars(items, max_chars=args.max_chars))
        total_batches = len(batches)
        parallel = max(1, args.parallel)
        if args.verbose:
            print(
                f"{in_path.name}: translate {len(items_full)} segments, "
                f"{total_batches} batches, parallel={parallel}"
            )
            print(
                f"{in_path.name}: a segment = one subtitle block; ids are the SRT numbers; "
                f"batches group multiple segments per request"
            )
            if parallel > 1:
                print(f"{in_path.name}: note: batch completion logs may appear out of order (parallel)")

        progress_state: Dict[str, object] = {}
        progress_state["adaptive_limits"] = args.adaptive_limits
        progress_state["max_chars"] = args.max_chars
        progress_state["max_tokens"] = args.max_tokens
        progress_state["max_chars_cap"] = args.max_chars
        progress_state["max_tokens_cap"] = args.max_tokens
        progress_state["min_chars"] = max(500, args.max_chars // 4)
        progress_state["min_tokens"] = max(256, args.max_tokens // 4)
        progress_state["stable_ok"] = 0
        progress_state["verbose"] = args.verbose
        if parallel > 1:
            progress_state["lock"] = threading.Lock()
        total_segments = len(items_full)
        if args.progress_bar:
            done = len(translated_all)
            sys.stdout.write("\r" + progress_line(in_path.name, done, total_segments))
            sys.stdout.flush()
        if parallel == 1:
            for bi, batch in enumerate(batches, start=1):
                progress_label = f"{in_path.name}: batch {bi}/{total_batches}"
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
                    progress_label=progress_label,
                    progress_state=progress_state,
                )
                translated_all.update(translated)
                if args.resume:
                    save_progress(progress_path, translated_all)
                if args.verbose:
                    print(f"{in_path.name}: batch {bi}/{total_batches} done, total {len(translated_all)}/{len(items)}")
                if args.progress_bar:
                    done = len(translated_all)
                    sys.stdout.write("\r" + progress_line(in_path.name, done, total_segments))
                    sys.stdout.flush()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
                futures: Dict[concurrent.futures.Future[Dict[str, str]], int] = {}
                for bi, batch in enumerate(batches, start=1):
                    progress_label = f"{in_path.name}: batch {bi}/{total_batches}"
                    fut = ex.submit(
                        translate_items_recursive,
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
                        progress_label=progress_label,
                        progress_state=progress_state,
                    )
                    futures[fut] = bi

                for fut in concurrent.futures.as_completed(futures):
                    bi = futures[fut]
                    translated = fut.result()
                    translated_all.update(translated)
                    if args.resume:
                        save_progress(progress_path, translated_all)
                    if args.verbose:
                        print(
                            f"{in_path.name}: batch {bi}/{total_batches} done, "
                            f"total {len(translated_all)}/{len(items)}"
                        )
                    if args.progress_bar:
                        done = len(translated_all)
                        sys.stdout.write("\r" + progress_line(in_path.name, done, total_segments))
                        sys.stdout.flush()

        if args.progress_bar:
            sys.stdout.write("\n")
        out_segments = apply_translations(src_segments, translated_all, strip_translator_tag=args.strip_translator_tag)
        out_text = format_srt(out_segments)
        out_path.write_text(out_text, encoding="utf-8")
        print(colorize(f"WROTE: {out_path}", "ok", args.color))
        if args.resume:
            save_progress(progress_path, translated_all)

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
                if isinstance(it, dict) and it.get("type") == "missing_model_outputs":
                    for sid in it.get("ids", []):
                        flagged_ids.append(str(sid))

        if args.validate and heur is not None:
            status = "ok" if heur["ok"] else "warn"
            print(colorize(
                f"VALIDATE: ok={heur['ok']} total={heur['total_segments']} expected_translations={heur['expected_translations']}",
                status,
                args.color,
            ))
            if not heur["ok"]:
                type_counts: Dict[str, int] = {}
                for it in heur["issues"]:
                    if isinstance(it, dict):
                        t = str(it.get("type", "unknown"))
                        type_counts[t] = type_counts.get(t, 0) + 1
                if type_counts:
                    top = sorted(type_counts.items(), key=lambda x: (-x[1], x[0]))[:6]
                    top_str = ", ".join(f"{k}={v}" for k, v in top)
                    print(colorize(f"  SUMMARY: {top_str}", "warn", args.color))
                for it in heur["issues"][:25]:
                    print(colorize(f"  ISSUE: {it}", "warn", args.color))

        if args.auto_fix and heur is not None and flagged_ids:
            fix_rounds = 2
            for round_idx in range(1, fix_rounds + 1):
                unique_ids = sorted(set(flagged_ids), key=lambda x: int(x))
                if not unique_ids:
                    break
                sample = ", ".join(unique_ids[:8])
                print(colorize(
                    f"AUTO-FIX: round {round_idx}/{fix_rounds}, retranslate {len(unique_ids)} segments (e.g. {sample})",
                    "info",
                    args.color,
                ))
                items_map = build_items_map_with_context(
                    src_segments,
                    strip_translator_tag=args.strip_translator_tag,
                    context_window=max(0, args.context_window),
                )
                fix_items = [items_map[sid] for sid in unique_ids if sid in items_map]
                if fix_items:
                    fix_max_chars = min(args.max_chars, 2000)
                    fix_max_tokens = min(args.max_tokens, 1024)
                    fix_batches = list(chunk_by_chars(fix_items, max_chars=fix_max_chars))
                    for bi, batch in enumerate(fix_batches, start=1):
                        progress_label = f"{in_path.name}: autofix batch {bi}/{len(fix_batches)}"
                        fixed = translate_items_recursive(
                            api_key=args.api_key,
                            model=args.model,
                            target_lang=args.target_lang,
                            items=batch,
                            temperature=args.temperature,
                            max_tokens=fix_max_tokens,
                            timeout_s=args.timeout,
                            app_url=args.app_url,
                            app_title=args.app_title,
                            response_format=response_format,
                            attempts=args.attempts,
                            progress_label=progress_label,
                            progress_state=progress_state,
                        )
                        translated_all.update(fixed)

                    out_segments = apply_translations_with_linebreaks(
                        src_segments,
                        translated_all,
                        strip_translator_tag=args.strip_translator_tag,
                        enforce_ids=set(unique_ids),
                    )
                    out_text = format_srt(out_segments)
                    out_path.write_text(out_text, encoding="utf-8")
                    print(colorize(f"WROTE: {out_path}", "ok", args.color))

                    if args.resume:
                        save_progress(progress_path, translated_all)
                    if args.validate:
                        heur = validate_translation(
                            original_segments=src_segments,
                            translated_segments=out_segments,
                            translated_ids=set(translated_all.keys()),
                            target_lang=args.target_lang,
                            strip_translator_tag=args.strip_translator_tag,
                        )
                        status = "ok" if heur["ok"] else "warn"
                        print(colorize(
                            f"AUTO-FIX VALIDATE: ok={heur['ok']} total={heur['total_segments']} "
                            f"expected_translations={heur['expected_translations']}",
                            status,
                            args.color,
                        ))
                        if not heur["ok"]:
                            type_counts = {}
                            flagged_ids = []
                            for it in heur["issues"]:
                                if isinstance(it, dict):
                                    t = str(it.get("type", "unknown"))
                                    type_counts[t] = type_counts.get(t, 0) + 1
                                    if "id" in it:
                                        flagged_ids.append(str(it["id"]))
                            if type_counts:
                                top = sorted(type_counts.items(), key=lambda x: (-x[1], x[0]))[:6]
                                top_str = ", ".join(f"{k}={v}" for k, v in top)
                                print(colorize(f"  SUMMARY: {top_str}", "warn", args.color))
                            for it in heur["issues"][:25]:
                                print(colorize(f"  ISSUE: {it}", "warn", args.color))
                        else:
                            flagged_ids = []
                else:
                    break

        if args.resume and progress_path.exists():
            if args.validate and heur is not None and heur.get("ok"):
                progress_path.unlink()

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
