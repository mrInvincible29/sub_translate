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

### All flags
- `-t, --target-lang`: target language code.
- `--model`: translation model id.
- `--judge-model`: QC model id (used with `--llm-qc`).
- `--api-key`: OpenRouter API key override.
- `--max-chars`: approx input chars per request.
- `--max-tokens`: output tokens per request.
- `--qc-max-tokens`: output tokens for QC.
- `--temperature`: translation temperature.
- `--qc-temperature`: QC temperature.
- `--timeout`: HTTP timeout seconds.
- `--overwrite`, `--overwrite-translated`: overwrite output if exists.
- `--recursive`: recurse into subfolders when input is a directory.
- `--strip-translator-tag`, `--no-strip-translator-tag`: remove translator tag lines.
- `--context-window`: include N previous/next segments as context.
- `--validate`, `--no-validate`: heuristic validation.
- `--llm-qc`: run judge model QC and save report JSON.
- `--qc-limit`: QC segment limit (0 = all).
- `--auto-fix`, `--no-auto-fix`: auto-fix flagged segments.
- `--resume`, `--no-resume`: save/load progress file.
- `--app-url`: HTTP-Referer header for OpenRouter.
- `--app-title`: X-Title header for OpenRouter.
- `--json-mode`, `--no-json-mode`: use response_format json_object.
- `--attempts`: retry attempts per request.
- `--parallel`: parallel batch requests.
- `--progress-bar`, `--no-progress-bar`: persistent progress bar.
- `--verbose`: detailed batch/missing logs.
- `--adaptive-limits`, `--no-adaptive-limits`: auto-tune limits.
- `--color`, `--no-color`: colorized logs.
- `--movies-root`: TMDB lookup root.

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

Use cases:
- Auto translate a movie folder by TMDB id and resume safely:
```bash
python3 translate_srt_openrouter.py 1197137 -t th
```
- Overwrite an existing translated file:
```bash
python3 translate_srt_openrouter.py 1197137 -t th --overwrite-translated
```
- Lower batch sizes for stricter JSON:
```bash
python3 translate_srt_openrouter.py 1197137 -t th --max-chars 1500 --max-tokens 768
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

### All flags
- `input`: TMDB id or directory containing subtitle files.
- `--movies-root`: TMDB lookup root.
- `--lang-a`, `--lang-b`: primary/secondary language codes.
- `--a`, `--b`: explicit subtitle file paths.
- `-o, --output`: output `.ass` path.
- `--playres-x`, `--playres-y`: ASS resolution.
- `--margin-a`, `--margin-b`: vertical placement margins.

Use cases:
- Create EN+TH ASS for Plex TV apps:
```bash
python3 combine_subs_bilingual_ass.py 1197137
```
- Use a different pair and custom output:
```bash
python3 combine_subs_bilingual_ass.py 1197137 --lang-a en --lang-b es -o "/path/Movie.EN+ES.ass"
```
