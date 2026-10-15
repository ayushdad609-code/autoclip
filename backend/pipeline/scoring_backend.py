"""Step 3 scoring contract; provider selection remains with LLMClient by default."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Protocol, Sequence, TypedDict

logger = logging.getLogger(__name__)


class ScoreResult(TypedDict, total=False):
    id: str
    outline: str
    final_score: float
    recommend_reason: str
    hook_start: str
    hook_text: str
    hook_score: float
    standalone_score: float
    emotion_score: float
    payoff_score: float


class ScoringBackend(Protocol):
    """Evaluate clips identified by id/outline, returning final_score and reason.

    Input includes transcript, content and absolute start/end times. Return a
    JSON-compatible list; scores use 0–1 or 0-10. Missing/invalid results are handled by
    the pipeline's alignment and fallback guards. Implementations must retain
    clip identifiers; they do not select, reorder or cut the source video.
    """

    def score(self, clips: Sequence[Dict[str, Any]]) -> List[ScoreResult]: ...


SCORING_CONTRACT = '''

## 本次评分输出契约（覆盖上文输入/输出示例）
你的任务是评估每个片段的内容价值，依据 transcript 原文、content 和 outline。
返回 JSON 数组，每个输入片段对应一个对象，保留原 id 和 outline：
[{"id":"原 id","outline":"原 outline","final_score":0.8,"recommend_reason":"基于原文的简短理由"}]
final_score 必须是 0 到 1 的有限数值；没有足够内容依据时不要编造高分。
不得只返回 id 到推荐理由的字典，不得省略分数，不要改变 id 或 outline。
'''

SCORING_HOOK_CONTRACT = '''

## 本次评分输出契约（覆盖上文输入/输出示例）
你的任务是评估每个候选短视频片段的爆款潜力和 Hook 抓人度，依据 transcript 原文、content、outline、hook_start、hook_text。
返回 JSON 数组，每个输入片段对应一个对象，保留原 id 和 outline：
[
  {
    "id": "原 id",
    "outline": "原 outline",
    "hook_start": "00:01:23,456",
    "hook_text": "核心抓人金句原文",
    "hook_score": 8.5,
    "standalone_score": 7.5,
    "emotion_score": 8.0,
    "payoff_score": 8.5,
    "final_score": 8.1,
    "recommend_reason": "基于原文的简短推荐理由（15-30字）"
  }
]
评分标准（0-10分）：
- hook_score (0-10): 开头最抓人的一句话的吸引力（10=瞬间驻足，0=平淡寒暄）。若 hook_start 或 hook_text 未精准对齐最抓人语句，请校正。
- standalone_score (0-10): 脱离前后文的独立可理解性（>=6为及格，<6说明严重依赖前置背景）。
- emotion_score (0-10): 情绪刺激、观点鲜明度与共鸣度。
- payoff_score (0-10): 结尾交付、高潮或悬念的满足感。
- final_score: 加权总分 = hook*0.4 + standalone*0.2 + emotion*0.2 + payoff*0.2 (0-10分区间)。
- recommend_reason: 15-30字推荐理由，精准点出爆点。
必须返回纯有效 JSON 数组，必须用英文双引号，不要包含任何解释文字。
'''


class LLMScoringBackend:
    """Current LLM scoring with a consistent schema for every category prompt."""

    def __init__(self, client: Any, prompt: str):
        self.client = client
        from .settings import get_hook_first
        from ..core.shared_config import inject_hook_rubric
        if get_hook_first():
            prompt_with_rubric = inject_hook_rubric(prompt, step="scoring")
            self.prompt = prompt_with_rubric + SCORING_HOOK_CONTRACT
        else:
            self.prompt = prompt + SCORING_CONTRACT

    def score(self, clips: Sequence[Dict[str, Any]]) -> List[ScoreResult]:
        from .settings import get_hook_first
        is_hook_first = get_hook_first()
        attempts = 3  # 1 request + 2 retries
        for attempt in range(attempts):
            try:
                response = self.client.call_with_retry(self.prompt, list(clips))
                if not response:
                    logger.warning(f"评分响应为空 (attempt {attempt + 1}/{attempts})")
                    continue
                results = self.client.parse_json_response(response)
                if isinstance(results, dict):
                    for k in ("clips", "scored_clips", "results"):
                        if isinstance(results.get(k), list):
                            results = results[k]
                            break
                if isinstance(results, list) and results:
                    validated: List[ScoreResult] = []
                    for r in results:
                        if not isinstance(r, dict):
                            continue
                        item = dict(r)
                        if is_hook_first:
                            h = item.get("hook_score")
                            s = item.get("standalone_score")
                            e = item.get("emotion_score")
                            p = item.get("payoff_score")
                            if all(v is not None and not isinstance(v, bool) for v in (h, s, e, p)):
                                try:
                                    calc_score = float(h) * 0.4 + float(s) * 0.2 + float(e) * 0.2 + float(p) * 0.2
                                    item["final_score"] = round(calc_score, 2)
                                    item["final_score_10"] = round(calc_score, 2)
                                except (TypeError, ValueError):
                                    pass
                            else:
                                logger.warning(f"片段 {item.get('id', '未知')} 缺少部分 Hook 评分维度，回退使用 final_score")
                        validated.append(item)
                    if validated:
                        return validated
            except Exception as e:
                logger.warning(f"评分解析第 {attempt + 1}/{attempts} 次出错: {e}")
        return []
