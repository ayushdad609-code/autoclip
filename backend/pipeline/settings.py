"""Read saved processing settings at the start of each pipeline step."""
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _read_setting_direct(name: str, default: Any = None) -> Any:
    """尝试通过 LLMManager 获取设置；若环境缺少重型依赖，直接回退读取 data/settings.json。"""
    try:
        from ..core.llm_manager import get_llm_manager
        return get_llm_manager().get_processing_setting(name, default)
    except Exception:
        pass

    try:
        settings_path = Path(__file__).resolve().parent.parent.parent / "data" / "settings.json"
        if settings_path.exists():
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and name in data:
                return data[name]
    except Exception:
        pass
    return default


def processing_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = _read_setting_direct(name, default)
    try:
        if isinstance(value, bool) or int(value) != float(value):
            return default
        number = int(value)
        return number if minimum <= number <= maximum else default
    except (ValueError, TypeError, OverflowError):
        return default


def processing_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = _read_setting_direct(name, default)
    try:
        if isinstance(value, bool):
            return default
        f = float(value)
        return f if minimum <= f <= maximum else default
    except (ValueError, TypeError, OverflowError):
        return default


def processing_bool(name: str, default: bool) -> bool:
    value = _read_setting_direct(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(value, (int, float)):
        return bool(value)
    return default



def get_hook_first(default: bool = True) -> bool:
    return processing_bool("hook_first", default)


def get_min_hook_score(default: float = 7.0) -> float:
    return processing_float("min_hook_score", default, 0.0, 10.0)


def get_min_clip_sec(default: int = 15) -> int:
    return processing_int("min_clip_sec", default, 1, 300)


def get_max_clip_sec(default: int = 45) -> int:
    return processing_int("max_clip_sec", default, 5, 600)


def get_preroll_sec(default: float = 0.2) -> float:
    return processing_float("preroll_sec", default, 0.0, 5.0)


def get_cold_open(default: bool = False) -> bool:
    return processing_bool("cold_open", default)


def get_vertical(default: bool = False) -> bool:
    return processing_bool("vertical", default)


def get_burn_captions(default: bool = False) -> bool:
    return processing_bool("burn_captions", default)

