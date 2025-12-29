# Subtitle Tools

Two scripts:
- `translate_srt_openrouter.py` - translate `.srt` via OpenRouter with validation, auto-fix, resume, and parallel batches.
- `combine_subs_bilingual_ass.py` - combine two subtitle languages into a bilingual `.ass` for Plex TV apps.

## translate_srt_openrouter.py

### Quick start
```bash
export OPENROUTER_API_KEY="..."
python3 translate_srt_openrouter.py 1197137 -t th
```

### Inputs
- Accepts `.srt` file paths, directories, or a TMDB id (e.g. `1197137`).
- TMDB id searches under `/home/h2/media/Movies` by default.

### Common flags
- `--model`: OpenRouter model id.
- `--parallel`: concurrent batch requests (default: 4).
- `--max-chars`, `--max-tokens`: per-batch limits.
- `--resume`: save/load progress to avoid re-translation (default: on).
- `--overwrite-translated`: overwrite existing translated `.srt`.
- `--validate`, `--auto-fix`: validation + auto-fix (default: on).
- `--progress-bar`: persistent progress bar (default: on).
- `--verbose`: print detailed batch/missing logs.
- `--no-color`: disable colored logs.
- `--movies-root`: override TMDB lookup root.
- Auto-extract: if no English `.srt` is found in a directory/TMDB folder, the script uses `ffprobe/ffmpeg` to extract an English subtitle stream when available.

### Examples
Translate a file:
```bash
python3 translate_srt_openrouter.py "/path/Movie.en.srt" -t th
```

Translate by TMDB id:
```bash
python3 translate_srt_openrouter.py 1197137 -t th
```

Force overwrite:
```bash
python3 translate_srt_openrouter.py 1197137 -t th --overwrite-translated
```

Verbose logs:
```bash
python3 translate_srt_openrouter.py 1197137 -t th --verbose
```

## combine_subs_bilingual_ass.py

Creates a single bilingual `.ass` subtitle with two lines per segment:
primary language on the bottom, secondary above it. Uses ASS positioning so
Plex TV apps render both lines reliably.

### Examples
From TMDB id:
```bash
python3 combine_subs_bilingual_ass.py 1197137
```

Custom language pair:
```bash
python3 combine_subs_bilingual_ass.py 1197137 --lang-a en --lang-b th
```

Explicit files:
```bash
python3 combine_subs_bilingual_ass.py whatever \
  --a "/path/movie.en.srt" \
  --b "/path/movie.th.srt"
```

Adjust placement:
```bash
python3 combine_subs_bilingual_ass.py 1197137 --margin-a 30 --margin-b 80
```
