"""
Unit tests for Hook-First video pipeline.
Tests cover:
1. Hook-start snapping to SRT line boundaries and word timing fallback.
2. Duration clamping (15-45s) in timeline refinement.
3. Strict JSON validation and retries.
4. Fallback behavior when hook fields or models fail.
5. VideoProcessor cutting commands (standard, vertical 9:16, cold-open, ASS subtitles).
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.pipeline.quality import DurationProfile, profile_for, refine_timeline, to_seconds, to_srt_time
from backend.pipeline.scoring_backend import LLMScoringBackend
from backend.pipeline.step2_timeline import TimelineExtractor
from backend.pipeline.step3_scoring import ClipScorer
from backend.utils.video_processor import VideoProcessor
from backend.utils.text_processor import TextProcessor


def _make_cue(start_sec: float, end_sec: float, text: str = "Test subtitle"):
    return {
        "start_time": to_srt_time(start_sec),
        "end_time": to_srt_time(end_sec),
        "text": text,
        "index": 1,
    }


class TestHookFirstSnapping(unittest.TestCase):
    """Test hook-start snapping to SRT lines and silence padding."""

    def test_snap_start_to_srt_boundary(self):
        cues = [
            _make_cue(0.0, 8.0, "Introductory remarks"),
            _make_cue(10.0, 20.0, "This is the hook sentence!"),
            _make_cue(20.0, 30.0, "Here is the explanation."),
        ]
        # Candidate start is 10.3s (within cue starting at 10.0s)
        snapped = VideoProcessor.snap_start_to_srt(10.3, srt_entries=cues)
        self.assertAlmostEqual(snapped, 10.0, places=2)

        # Candidate start is 9.6s (within 1s threshold of cue 2 starting at 10.0s)
        snapped_near = VideoProcessor.snap_start_to_srt(9.6, srt_entries=cues)
        self.assertAlmostEqual(snapped_near, 10.0, places=2)

    def test_snap_mid_line_fallback(self):
        cues = [
            _make_cue(10.0, 25.0, "A very long sentence where the hook is buried inside."),
        ]
        # Candidate start is 14.0s (> 0.8s into cue).
        # Without word timing sidecar, it falls back safely to the cue boundary (10.0s).
        snapped = VideoProcessor.snap_start_to_srt(14.0, srt_entries=cues)
        self.assertEqual(snapped, 10.0)

    def test_snap_mid_line_with_word_timing(self):
        from backend.utils.word_timing import write_word_timing
        cues = [
            _make_cue(10.0, 25.0, "First word second word third word fourth word"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_path = Path(tmp_dir) / "test.srt"
            srt_path.write_text("1\n00:00:10,000 --> 00:00:25,000\nFirst word second word third word fourth word\n", encoding="utf-8")
            segments = [
                {
                    "start": 10.0,
                    "end": 25.0,
                    "text": "First word second word third word fourth word",
                    "words": [
                        {"word": "First", "text": "First", "start": 10.0, "end": 10.8},
                        {"word": "word", "text": "word", "start": 10.8, "end": 11.2},
                        {"word": "second", "text": "second", "start": 11.2, "end": 12.0},
                        {"word": "word", "text": "word", "start": 12.0, "end": 12.5},
                        {"word": "third", "text": "third", "start": 13.9, "end": 14.5},
                        {"word": "word", "text": "word", "start": 14.5, "end": 14.9},
                        {"word": "fourth", "text": "fourth", "start": 14.9, "end": 15.5},
                        {"word": "word", "text": "word", "start": 15.5, "end": 16.0},
                    ]
                }
            ]
            write_word_timing(srt_path, segments, source='faster-whisper')

            snapped = VideoProcessor.snap_start_to_srt(14.0, srt_path=srt_path, srt_entries=cues)
            # Snaps to word 'third' starting at 13.9s
            self.assertAlmostEqual(snapped, 13.9, places=2)

    def test_adjust_start_with_silence_preroll(self):
        fake_video = Path("/nonexistent/video.mp4")
        # With non-existent video, ffmpeg probe will safely fall back to cand_start
        adjusted = VideoProcessor.adjust_start_with_silence(fake_video, 15.0, preroll_sec=0.2)
        self.assertAlmostEqual(adjusted, 14.8, places=2)

        adjusted_zero = VideoProcessor.adjust_start_with_silence(fake_video, 0.1, preroll_sec=0.2)
        self.assertEqual(adjusted_zero, 0.0)


class TestDurationClamping(unittest.TestCase):
    """Test duration clamping for 15s - 45s short-form clips."""

    def test_refine_timeline_duration_clamping(self):
        # 60 cues of 1 second each (0 to 60s)
        cues = [_make_cue(i, i + 1, f"cue_{i}") for i in range(60)]
        profile = DurationProfile(
            tier="short",
            total_sec=60,
            min_clip_sec=15,
            target_clip_sec=(15, 45),
            max_clip_sec=45,
            topics_hint=(1, 5),
            min_keep=1,
            max_clips=5,
        )

        items = [
            # Clip 1: 5s (too short to reach 15s if at end of boundary, or extendable)
            {
                "outline": "Too Short NonExtendable",
                "start_time": "00:00:58,000",
                "end_time": "00:01:00,000",
                "hook_start": "00:00:58,000",
            },
            # Clip 2: 55s (exceeds max_clip_sec 45s -> must be trimmed)
            {
                "outline": "Too Long Clip",
                "start_time": "00:00:05,000",
                "end_time": "00:01:00,000",
                "hook_start": "00:00:05,000",
            },
            # Clip 3: 20s (valid short clip)
            {
                "outline": "Perfect Clip",
                "start_time": "00:00:10,000",
                "end_time": "00:00:30,000",
                "hook_start": "00:00:10,000",
            },
        ]

        refined, report = refine_timeline(items, cues, profile)
        # Clip 1 cannot reach 15s because video ends at 60s and it starts at 58s -> dropped
        outlines = [x["outline"] for x in refined]
        self.assertNotIn("Too Short NonExtendable", outlines)

        # All surviving clips must satisfy min_clip_sec <= dur <= max_clip_sec
        for clip in refined:
            dur = clip["duration_sec"]
            self.assertGreaterEqual(dur, 15)
            self.assertLessEqual(dur, 45)

        # Clip 2 must have been trimmed to <= 45s
        long_clip = next(x for x in refined if x["outline"] == "Too Long Clip")
        self.assertLessEqual(long_clip["duration_sec"], 45)


class TestJSONValidationAndScoring(unittest.TestCase):
    """Test JSON validation, weighted final_score computation, and retry logic."""

    def test_scoring_backend_weighted_sum(self):
        mock_client = MagicMock()
        backend = LLMScoringBackend(mock_client, "prompt")

        # Mock LLM returning 4 hook dimensions
        mock_client.call_with_retry.return_value = json.dumps([
            {
                "id": "1",
                "outline": "Great Hook Clip",
                "hook_start": "00:00:12,000",
                "hook_text": "The shocking secret!",
                "hook_score": 9.0,
                "standalone_score": 8.0,
                "emotion_score": 7.0,
                "payoff_score": 8.0,
                "recommend_reason": "High viral potential",
            }
        ])
        mock_client.parse_json_response.side_effect = lambda s: json.loads(s)

        with patch("backend.pipeline.settings.get_hook_first", return_value=True):
            results = backend.score([{"id": "1", "outline": "Great Hook Clip"}])

        self.assertEqual(len(results), 1)
        # Expected: 9.0*0.4 + 8.0*0.2 + 7.0*0.2 + 8.0*0.2 = 3.6 + 1.6 + 1.4 + 1.6 = 8.2
        self.assertAlmostEqual(results[0]["final_score"], 8.2, places=2)
        self.assertEqual(results[0]["hook_score"], 9.0)

    def test_scoring_backend_retries_on_invalid_json(self):
        mock_client = MagicMock()
        backend = LLMScoringBackend(mock_client, "prompt")

        # Attempt 1 returns broken JSON, Attempt 2 returns valid JSON
        mock_client.call_with_retry.side_effect = [
            "Not valid json {",
            json.dumps([{"id": "1", "final_score": 8.0, "hook_score": 8.0, "standalone_score": 7.0, "emotion_score": 7.0, "payoff_score": 7.0}]),
        ]
        def parse_helper(s):
            return json.loads(s)
        mock_client.parse_json_response.side_effect = [ValueError("Invalid JSON"), [{"id": "1", "final_score": 8.0, "hook_score": 8.0, "standalone_score": 7.0, "emotion_score": 7.0, "payoff_score": 7.0}]]

        with patch("backend.pipeline.settings.get_hook_first", return_value=True):
            results = backend.score([{"id": "1"}])

        self.assertEqual(len(results), 1)
        self.assertEqual(mock_client.call_with_retry.call_count, 2)

    def test_step2_hook_start_replaces_start_time(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            extractor = TimelineExtractor.__new__(TimelineExtractor)
            extractor.text_processor = TextProcessor()
            extractor.metadata_dir = Path(tmp_dir)
            extractor.llm_raw_output_dir = Path(tmp_dir)
            extractor.llm_client = MagicMock()
            extractor.llm_client.parse_json_response.side_effect = lambda s: json.loads(s)
            extractor._last_parse_diagnostics = {}

            raw_response = json.dumps([
                {
                    "outline": "Explosive Discovery",
                    "start_time": "00:00:05,000",
                    "end_time": "00:00:35,000",
                    "hook_start": "00:00:15,000",
                    "hook_text": "This will change everything you know!",
                    "hook_score": 9.5,
                    "standalone_score": 8.0,
                }
            ])

            with patch("backend.pipeline.settings.get_hook_first", return_value=True):
                parsed = extractor._parse_and_validate_response(
                    raw_response, chunk_start="00:00:00,000", chunk_end="00:01:00,000", chunk_index=0
                )

            self.assertEqual(len(parsed), 1)
            item = parsed[0]
            # In hook-first mode, clip start snaps to hook_start
            self.assertEqual(item["start_time"], "00:00:15,000")
            self.assertEqual(item["topic_start_time"], "00:00:05,000")
            self.assertEqual(item["hook_start"], "00:00:15,000")
            self.assertEqual(item["hook_score"], 9.5)


class TestFallbackBehavior(unittest.TestCase):
    """Test fallbacks when LLM lacks hook fields or scores fall below thresholds."""

    def test_step2_fallback_when_hook_start_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            extractor = TimelineExtractor.__new__(TimelineExtractor)
            extractor.text_processor = TextProcessor()
            extractor.metadata_dir = Path(tmp_dir)
            extractor.llm_raw_output_dir = Path(tmp_dir)
            extractor.llm_client = MagicMock()
            extractor.llm_client.parse_json_response.side_effect = lambda s: json.loads(s)
            extractor._last_parse_diagnostics = {}

            # LLM omits hook_start and hook_text
            raw_response = json.dumps([
                {
                    "outline": "Legacy Segment",
                    "start_time": "00:00:10,000",
                    "end_time": "00:00:40,000",
                }
            ])

            with patch("backend.pipeline.settings.get_hook_first", return_value=True):
                parsed = extractor._parse_and_validate_response(
                    raw_response, chunk_start="00:00:00,000", chunk_end="00:01:00,000", chunk_index=0
                )

            self.assertEqual(len(parsed), 1)
            item = parsed[0]
            # Gracefully falls back to start_time
            self.assertEqual(item["start_time"], "00:00:10,000")
            self.assertEqual(item["hook_start"], "00:00:10,000")
            self.assertEqual(item["hook_text"], "")

    def test_step3_filter_rejects_low_hook_score(self):
        scorer = ClipScorer.__new__(ClipScorer)
        scorer.scoring_backend = MagicMock()
        scorer.metadata_dir = None
        scorer._srt_entries = []

        timeline_data = [
            {
                "id": "1",
                "chunk_index": 0,
                "outline": "Low Hook Clip",
                "start_time": "00:00:05,000",
                "end_time": "00:00:30,000",
                "hook_start": "00:00:10,000",
            },
            {
                "id": "2",
                "chunk_index": 0,
                "outline": "High Hook Clip",
                "start_time": "00:00:35,000",
                "end_time": "00:01:00,000",
                "hook_start": "00:00:40,000",
            },
        ]

        scored_mock = [
            {
                "id": "1",
                "chunk_index": 0,
                "outline": "Low Hook Clip",
                "start_time": "00:00:05,000",
                "end_time": "00:00:30,000",
                "hook_start": "00:00:10,000",
                "hook_score": 4.0,  # Below threshold 7.0
                "standalone_score": 7.0,
                "final_score": 5.0,
            },
            {
                "id": "2",
                "chunk_index": 0,
                "outline": "High Hook Clip",
                "start_time": "00:00:35,000",
                "end_time": "00:01:00,000",
                "hook_start": "00:00:40,000",
                "hook_score": 8.5,  # Passes >= 7.0
                "standalone_score": 8.0,  # Passes >= 6.0
                "final_score": 8.3,
            },
        ]

        scorer._get_llm_evaluation = MagicMock(return_value=scored_mock)

        with patch("backend.pipeline.settings.get_hook_first", return_value=True), \
             patch("backend.pipeline.settings.get_min_hook_score", return_value=7.0):
            selected = scorer.score_clips(timeline_data)

        # Clip 1 must be filtered out, Clip 2 kept
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], "2")
        # Start time must be hook_start
        self.assertEqual(selected[0]["start_time"], "00:00:40,000")

    def test_step3_filter_fallback_when_all_below_threshold(self):
        scorer = ClipScorer.__new__(ClipScorer)
        scorer.scoring_backend = MagicMock()
        scorer.metadata_dir = None
        scorer._srt_entries = []

        timeline_data = [
            {"id": "1", "chunk_index": 0, "outline": "Low 1", "start_time": "00:00:00,000", "end_time": "00:00:20,000"},
            {"id": "2", "chunk_index": 0, "outline": "Low 2", "start_time": "00:00:20,000", "end_time": "00:00:40,000"},
        ]

        scored_mock = [
            {"id": "1", "chunk_index": 0, "outline": "Low 1", "hook_score": 5.0, "standalone_score": 5.0, "final_score": 5.0, "start_time": "00:00:00,000"},
            {"id": "2", "chunk_index": 0, "outline": "Low 2", "hook_score": 6.0, "standalone_score": 5.5, "final_score": 5.8, "start_time": "00:00:20,000"},
        ]

        scorer._get_llm_evaluation = MagicMock(return_value=scored_mock)

        with patch("backend.pipeline.settings.get_hook_first", return_value=True), \
             patch("backend.pipeline.settings.get_min_hook_score", return_value=7.0):
            selected = scorer.score_clips(timeline_data)

        # Robustness: pipeline never crashes or returns empty, keeps best candidates as fallback
        self.assertEqual(len(selected), 2)


class TestVideoProcessorFeatures(unittest.TestCase):
    """Test command construction for re-encoding, vertical 9:16, cold-open, and ASS subtitles."""

    def test_build_extract_clip_command_standard(self):
        cmd = VideoProcessor.build_extract_clip_command(
            ffmpeg_bin="ffmpeg",
            input_video=Path("in.mp4"),
            output_path=Path("out.mp4"),
            start_time="00:00:10.000",
            duration=25.0,
            vertical=False,
        )
        cmd_str = " ".join(cmd)
        self.assertIn("-ss 00:00:10.000", cmd_str)
        self.assertIn("-t 25.000", cmd_str)
        self.assertIn("-c:v libx264", cmd_str)
        self.assertIn("-c:a aac", cmd_str)
        self.assertNotIn("crop=ih*(9/16):ih", cmd_str)

    def test_build_extract_clip_command_vertical(self):
        cmd = VideoProcessor.build_extract_clip_command(
            ffmpeg_bin="ffmpeg",
            input_video=Path("in.mp4"),
            output_path=Path("out.mp4"),
            start_time="00:00:10.000",
            duration=25.0,
            vertical=True,
        )
        cmd_str = " ".join(cmd)
        self.assertIn("crop=ih*(9/16):ih,scale=1080:1920", cmd_str)

    def test_build_extract_clip_command_cold_open(self):
        cold_params = {
            "hook_start": 20.0,
            "hook_dur": 3.0,
            "setup_start": 0.0,
            "setup_dur": 22.0,
        }
        cmd = VideoProcessor.build_extract_clip_command(
            ffmpeg_bin="ffmpeg",
            input_video=Path("in.mp4"),
            output_path=Path("out.mp4"),
            start_time="00:00:20.000",
            duration=25.0,
            cold_open_params=cold_params,
        )
        cmd_str = " ".join(cmd)
        self.assertIn("-filter_complex", cmd)
        self.assertIn("trim=start=20.000:end=23.000", cmd_str)
        self.assertIn("trim=start=0.000:end=22.000", cmd_str)
        self.assertIn("concat=n=2:v=1:a=1", cmd_str)

    def test_generate_clip_ass_subtitles(self):
        cues = [
            _make_cue(10.0, 14.0, "The single most gripping moment in the entire story"),
            _make_cue(14.0, 18.0, "And then the shocking outcome happened"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_ass = Path(tmp_dir) / "sub.ass"
            ok = VideoProcessor.generate_clip_ass_subtitles(
                srt_path=None,
                srt_entries=cues,
                clip_start_sec=10.0,
                clip_end_sec=18.0,
                output_ass_path=out_ass,
                is_vertical=True,
            )
            self.assertTrue(ok)
            content = out_ass.read_text(encoding="utf-8")
            self.assertIn("PlayResX: 1080", content)
            self.assertIn("PlayResY: 1920", content)
            self.assertIn(",40,40,280,1", content)
            self.assertIn("Dialogue: 0,", content)


if __name__ == "__main__":
    unittest.main()
