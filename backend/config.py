"""
AutoClip Configuration Module
"""
import json
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"

# Hook-First Settings (Defaults)
hook_first = True
min_hook_score = 7
min_clip_sec = 15
max_clip_sec = 45
preroll_sec = 0.2
cold_open = False
vertical = False
burn_captions = False

# Pipeline & Processing Defaults
chunk_size = 5000
min_score_threshold = 7.0
max_clips_per_collection = 10

# Load from data/settings.json if present
if SETTINGS_FILE.exists():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            _loaded = json.load(f)
            if isinstance(_loaded, dict):
                hook_first = _loaded.get("hook_first", hook_first)
                min_hook_score = _loaded.get("min_hook_score", min_hook_score)
                min_clip_sec = _loaded.get("min_clip_sec", min_clip_sec)
                max_clip_sec = _loaded.get("max_clip_sec", max_clip_sec)
                preroll_sec = _loaded.get("preroll_sec", preroll_sec)
                cold_open = _loaded.get("cold_open", cold_open)
                vertical = _loaded.get("vertical", vertical)
                burn_captions = _loaded.get("burn_captions", burn_captions)
                chunk_size = _loaded.get("chunk_size", chunk_size)
                min_score_threshold = _loaded.get("min_score_threshold", min_score_threshold)
                max_clips_per_collection = _loaded.get("max_clips_per_collection", max_clips_per_collection)
    except Exception:
        pass
