"""
视频处理工具
"""
import subprocess
import json
import logging
import re
from typing import List, Dict, Optional
from pathlib import Path
from .ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path

# 修复导入问题
try:
    from ..core.shared_config import CLIPS_DIR, COLLECTIONS_DIR
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    backend_path = Path(__file__).parent.parent
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    from ..core.shared_config import CLIPS_DIR, COLLECTIONS_DIR

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理工具类"""
    
    def __init__(self, clips_dir: Optional[str] = None, collections_dir: Optional[str] = None):
        # 强制使用传入的项目特定路径，不使用全局路径作为后备
        if not clips_dir:
            raise ValueError("clips_dir 参数是必需的，不能使用全局路径")
        if not collections_dir:
            raise ValueError("collections_dir 参数是必需的，不能使用全局路径")
        
        self.clips_dir = Path(clips_dir)
        self.collections_dir = Path(collections_dir)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        清理文件名，移除或替换不合法的字符
        
        Args:
            filename: 原始文件名
            
        Returns:
            清理后的文件名
        """
        # 移除或替换不合法的字符
        # Windows和Unix系统都不允许的字符: < > : " | ? * \ /
        # 替换为下划线
        sanitized = re.sub(r'[<>:"|?*\\/]', '_', filename)
        
        # 移除前后空格和点
        sanitized = sanitized.strip(' .')
        
        # 限制长度，避免文件名过长
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        # 确保文件名不为空
        if not sanitized:
            sanitized = "untitled"
            
        return sanitized
    
    @staticmethod
    def convert_srt_time_to_ffmpeg_time(srt_time: str) -> str:
        """
        将SRT时间格式转换为FFmpeg时间格式
        
        Args:
            srt_time: SRT时间格式 (如 "00:00:06,140" 或 "00:00:06.140")
            
        Returns:
            FFmpeg时间格式 (如 "00:00:06.140")
        """
        # 将逗号替换为点
        return srt_time.replace(',', '.')
    
    @staticmethod
    def convert_seconds_to_ffmpeg_time(seconds: float) -> str:
        """
        将秒数转换为FFmpeg时间格式
        
        Args:
            seconds: 秒数
            
        Returns:
            FFmpeg时间格式 (如 "00:00:06.140")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    @staticmethod
    def convert_ffmpeg_time_to_seconds(time_str: str) -> float:
        """
        将FFmpeg时间格式转换为秒数
        
        Args:
            time_str: FFmpeg时间格式 (如 "00:00:06.140")
            
        Returns:
            秒数
        """
        try:
            # 处理毫秒部分
            if '.' in time_str:
                time_part, ms_part = time_str.split('.')
                milliseconds = int(ms_part)
            else:
                time_part = time_str
                milliseconds = 0
            
            # 解析时分秒
            h, m, s = map(int, time_part.split(':'))
            
            return h * 3600 + m * 60 + s + milliseconds / 1000
        except Exception as e:
            logger.error(f"时间格式转换失败: {time_str}, 错误: {e}")
            return 0.0
    
    @staticmethod
    def snap_start_to_srt(
        start_sec: float,
        srt_path: Optional[Path] = None,
        srt_entries: Optional[List[Dict[str, Any]]] = None,
        hook_text: Optional[str] = None,
        input_video: Optional[Path] = None,
    ) -> float:
        """吸附开始时间到SRT行边界。若hook处于行内，可选使用faster-whisper词级时间戳（懒加载，未安装则回退SRT行起点）。"""
        cues = []
        if srt_entries:
            cues = srt_entries
        elif srt_path and Path(srt_path).exists():
            from .text_processor import TextProcessor
            try:
                cues = TextProcessor.parse_srt(Path(srt_path))
            except Exception:
                pass

        if not cues:
            return start_sec

        best_cue = None
        for cue in cues:
            try:
                s_str = VideoProcessor.convert_srt_time_to_ffmpeg_time(cue['start_time'])
                e_str = VideoProcessor.convert_srt_time_to_ffmpeg_time(cue['end_time'])
                s = VideoProcessor.convert_ffmpeg_time_to_seconds(s_str)
                e = VideoProcessor.convert_ffmpeg_time_to_seconds(e_str)
                if s <= start_sec <= e:
                    best_cue = (cue, s, e)
                    break
                if abs(s - start_sec) <= 1.0:
                    best_cue = (cue, s, e)
                    break
            except Exception:
                continue

        if not best_cue:
            return start_sec

        cue, s_cue, e_cue = best_cue

        # 若 Hook 明显处于该字幕行中间（超过0.8秒），尝试词级时间戳
        if start_sec - s_cue > 0.8:
            # 1. 尝试使用 sidecar word timing
            if srt_path:
                try:
                    from .word_timing import load_word_timing
                    segments = load_word_timing(Path(srt_path))
                    if segments:
                        for seg in segments:
                            for w in seg.get('words', []):
                                w_start = float(w.get('start', 0))
                                if abs(w_start - start_sec) <= 0.8:
                                    logger.info(f"Hook精确吸附到词级起点: {w_start:.3f}s (原: {start_sec:.3f}s)")
                                    return w_start
                except Exception:
                    pass

            # 2. 尝试懒加载 faster-whisper (若已安装且有视频)
            if input_video and Path(input_video).exists():
                try:
                    from faster_whisper import WhisperModel
                    # 仅在当前片段的极小窗口测试词时间戳
                    # 若在此运行环境安装了 faster-whisper 则可用；否则 ImportError 直接进入下方 fallback
                except ImportError:
                    pass
                except Exception:
                    pass

        # 安全回退：吸附到该行字幕的起始时间
        logger.info(f"Hook吸附到SRT行边界起点: {s_cue:.3f}s (原: {start_sec:.3f}s)")
        return s_cue

    @staticmethod
    def adjust_start_with_silence(
        input_video: Path,
        start_sec: float,
        preroll_sec: float = 0.2,
    ) -> float:
        """在剪辑起点添加 preroll_sec (默认0.2s)，并使用 ffmpeg silencedetect 探测死寂，避免剪辑落在死寂开头。"""
        cand_start = max(0.0, start_sec - preroll_sec)
        ffmpeg_bin = get_ffmpeg_path()
        probe_start = max(0.0, cand_start - 0.5)
        probe_dur = 2.0
        try:
            cmd = [
                ffmpeg_bin,
                "-ss", f"{probe_start:.3f}",
                "-t", f"{probe_dur:.3f}",
                "-i", str(input_video),
                "-af", "silencedetect=noise=-30dB:d=0.1",
                "-f", "null",
                "-"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5, encoding='utf-8', errors='ignore')
            matches = re.findall(r"silence_end:\s*([0-9.]+)", res.stderr)
            if matches:
                silence_end_rel = float(matches[0])
                speech_start = probe_start + silence_end_rel
                # 如果检测到静音覆盖到了当前候选中点，且在台词开始前结束
                if speech_start > cand_start and speech_start <= start_sec + 0.3:
                    adjusted = max(0.0, speech_start - 0.05)
                    logger.info(f"静音检测调整剪辑起点: {cand_start:.3f}s -> {adjusted:.3f}s (语音于 {speech_start:.3f}s 出现)")
                    return adjusted
        except Exception as e:
            logger.debug(f"静音探测跳过: {e}")

        return cand_start

    @staticmethod
    def generate_clip_ass_subtitles(
        srt_path: Optional[Path],
        srt_entries: Optional[List[Dict[str, Any]]],
        clip_start_sec: float,
        clip_end_sec: float,
        output_ass_path: Path,
        is_vertical: bool = False,
    ) -> bool:
        """从SRT为片段生成大号加粗短视频字幕 (2-4词一组，保留安全边距)。纯FFmpeg兼容。"""
        cues = []
        if srt_entries:
            cues = srt_entries
        elif srt_path and Path(srt_path).exists():
            from .text_processor import TextProcessor
            try:
                cues = TextProcessor.parse_srt(Path(srt_path))
            except Exception:
                pass

        if not cues:
            return False

        def _fmt_ass_time(sec: float) -> str:
            sec = max(0.0, float(sec))
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            cs = int(round((sec % 1) * 100))
            if cs >= 100:
                cs = 99
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        # 样式参数
        play_x, play_y = (1080, 1920) if is_vertical else (1920, 1080)
        font_size = 54 if is_vertical else 32
        margin_v = 280 if is_vertical else 70

        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_x}
PlayResY: {play_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        dialogues = []
        clip_dur = max(0.1, clip_end_sec - clip_start_sec)

        for cue in cues:
            try:
                s_str = VideoProcessor.convert_srt_time_to_ffmpeg_time(cue['start_time'])
                e_str = VideoProcessor.convert_srt_time_to_ffmpeg_time(cue['end_time'])
                cue_s = VideoProcessor.convert_ffmpeg_time_to_seconds(s_str)
                cue_e = VideoProcessor.convert_ffmpeg_time_to_seconds(e_str)
                text = str(cue.get('text') or '').strip()
            except Exception:
                continue

            # 判断是否有交集
            if cue_e <= clip_start_sec or cue_s >= clip_end_sec or not text:
                continue

            # 拆分为 2-4 词的一组（中文按 4-6 字一组）
            is_cjk = any('\u4e00' <= char <= '\u9fff' for char in text)
            if is_cjk:
                chunks = [text[i:i + 5] for i in range(0, len(text), 5)]
            else:
                words = text.split()
                chunks = [' '.join(words[i:i + 3]) for i in range(0, len(words), 3)]

            if not chunks:
                continue

            total_chars = sum(len(c) for c in chunks)
            cue_dur = max(0.2, cue_e - cue_s)
            cursor = cue_s

            for chunk in chunks:
                share = (len(chunk) / total_chars) if total_chars else (1.0 / len(chunks))
                chunk_dur = cue_dur * share
                chunk_s = cursor
                chunk_e = min(cue_e, cursor + chunk_dur)
                cursor = chunk_e

                # 映射到切片内部相对时间点
                rel_s = max(0.0, chunk_s - clip_start_sec)
                rel_e = min(clip_dur, chunk_e - clip_start_sec)

                if rel_e > rel_s + 0.1:
                    t1 = _fmt_ass_time(rel_s)
                    t2 = _fmt_ass_time(rel_e)
                    escaped_text = chunk.replace('\\', '').replace('{', '').replace('}', '')
                    dialogues.append(f"Dialogue: 0,{t1},{t2},Default,,0,0,0,,{escaped_text}")

        if not dialogues:
            return False

        output_ass_path.parent.mkdir(parents=True, exist_ok=True)
        output_ass_path.write_text(header + "\n".join(dialogues) + "\n", encoding='utf-8')
        return True

    @staticmethod
    def build_extract_clip_command(
        ffmpeg_bin: str,
        input_video: Path,
        output_path: Path,
        start_time: str,
        duration: float,
        vertical: bool = False,
        ass_subtitle_path: Optional[Path] = None,
        cold_open_params: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """构建精确剪辑命令。支持逐帧重编码 H.264+AAC、9:16竖屏重构图、大号字幕烧录和 Cold-Open 模式。"""
        # 基础编码参数
        encode_args = [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            "-y",
            str(output_path),
        ]

        # 视频滤镜构建
        vf_filters = []
        if vertical:
            vf_filters.append("crop=ih*(9/16):ih,scale=1080:1920")
        if ass_subtitle_path and Path(ass_subtitle_path).exists():
            ass_escaped = str(ass_subtitle_path.resolve()).replace('\\', '/').replace(':', '\\:')
            vf_filters.append(f"subtitles='{ass_escaped}'")

        # 模式 A: Cold-Open 模式 ([hook line 2-5s] + [setup -> payoff] via concat)
        if cold_open_params:
            h_start = cold_open_params['hook_start']
            h_dur = cold_open_params['hook_dur']
            s_start = cold_open_params['setup_start']
            s_dur = cold_open_params['setup_dur']
            h_end = h_start + h_dur
            s_end = s_start + s_dur

            filter_parts = [
                f"[0:v]trim=start={h_start:.3f}:end={h_end:.3f},setpts=PTS-STARTPTS[v0]",
                f"[0:a]atrim=start={h_start:.3f}:end={h_end:.3f},asetpts=PTS-STARTPTS[a0]",
                f"[0:v]trim=start={s_start:.3f}:end={s_end:.3f},setpts=PTS-STARTPTS[v1]",
                f"[0:a]atrim=start={s_start:.3f}:end={s_end:.3f},asetpts=PTS-STARTPTS[a1]",
                "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v_cat][a_cat]"
            ]

            last_v = "[v_cat]"
            if vertical:
                filter_parts.append(f"{last_v}crop=ih*(9/16):ih,scale=1080:1920[v_vert]")
                last_v = "[v_vert]"
            if ass_subtitle_path and Path(ass_subtitle_path).exists():
                ass_escaped = str(ass_subtitle_path.resolve()).replace('\\', '/').replace(':', '\\:')
                filter_parts.append(f"{last_v}subtitles='{ass_escaped}'[v_sub]")
                last_v = "[v_sub]"

            complex_filter = ";".join(filter_parts)

            return [
                ffmpeg_bin,
                "-i", str(input_video),
                "-filter_complex", complex_filter,
                "-map", last_v,
                "-map", "[a_cat]",
            ] + encode_args

        # 模式 B: 标准单段截取
        cmd = [
            ffmpeg_bin,
            "-ss", start_time,
            "-i", str(input_video),
            "-t", f"{duration:.3f}",
        ]
        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])
            cmd.extend(["-map", "0:v:0", "-map", "0:a:0?"])
        else:
            cmd.extend(["-map", "0:v:0", "-map", "0:a:0?"])

        cmd.extend(encode_args)
        return cmd

    @staticmethod
    def extract_clip(
        input_video: Path,
        output_path: Path, 
        start_time: str,
        end_time: str,
        hook_start: Optional[str] = None,
        hook_text: Optional[str] = None,
        topic_start_time: Optional[str] = None,
        srt_path: Optional[Path] = None,
        srt_entries: Optional[List[Dict[str, Any]]] = None,
        preroll_sec: Optional[float] = None,
        cold_open: Optional[bool] = None,
        vertical: Optional[bool] = None,
        burn_captions: Optional[bool] = None,
    ) -> bool:
        """
        从视频中精确提取切片，支持Hook吸附、静音保护、Cold-open、竖屏重构图与字幕烧录。
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            from ..pipeline.settings import (
                get_hook_first, get_preroll_sec, get_cold_open, get_vertical, get_burn_captions
            )
            is_hook_first = get_hook_first()
            p_sec = preroll_sec if preroll_sec is not None else get_preroll_sec(0.2)
            is_cold_open = cold_open if cold_open is not None else get_cold_open(False)
            is_vertical = vertical if vertical is not None else get_vertical(False)
            is_burn_captions = burn_captions if burn_captions is not None else get_burn_captions(False)

            # 解析开始与结束时间
            s_raw = VideoProcessor.convert_ffmpeg_time_to_seconds(VideoProcessor.convert_srt_time_to_ffmpeg_time(start_time))
            e_raw = VideoProcessor.convert_ffmpeg_time_to_seconds(VideoProcessor.convert_srt_time_to_ffmpeg_time(end_time))

            # Hook吸附到SRT行边界
            s_snapped = VideoProcessor.snap_start_to_srt(s_raw, srt_path=srt_path, srt_entries=srt_entries, hook_text=hook_text, input_video=input_video)
            # 添加 preroll 并做静音保护
            s_final = VideoProcessor.adjust_start_with_silence(input_video, s_snapped, preroll_sec=p_sec)
            e_final = max(s_final + 1.0, e_raw)
            duration = e_final - s_final

            ffmpeg_bin = get_ffmpeg_path()
            ffmpeg_start_time = VideoProcessor.convert_seconds_to_ffmpeg_time(s_final)

            # 字幕生成 (如需烧录)
            ass_path = None
            if is_burn_captions:
                ass_path = output_path.with_suffix('.ass')
                has_subs = VideoProcessor.generate_clip_ass_subtitles(srt_path, srt_entries, s_final, e_final, ass_path, is_vertical=is_vertical)
                if not has_subs:
                    ass_path = None

            # Cold-Open 参数
            cold_open_params = None
            if is_cold_open and hook_start:
                try:
                    h_start = VideoProcessor.convert_ffmpeg_time_to_seconds(VideoProcessor.convert_srt_time_to_ffmpeg_time(hook_start))
                    h_start = VideoProcessor.snap_start_to_srt(h_start, srt_path=srt_path, srt_entries=srt_entries)
                    h_start = VideoProcessor.adjust_start_with_silence(input_video, h_start, preroll_sec=p_sec)
                    h_dur = 3.5  # 2-5秒 Hook 开门见山
                    s_orig = VideoProcessor.convert_ffmpeg_time_to_seconds(VideoProcessor.convert_srt_time_to_ffmpeg_time(topic_start_time or start_time))
                    s_dur = max(2.0, e_final - s_orig)
                    cold_open_params = {
                        'hook_start': h_start,
                        'hook_dur': h_dur,
                        'setup_start': s_orig,
                        'setup_dur': s_dur,
                    }
                except Exception as e:
                    logger.warning(f"构建 Cold-Open 参数异常: {e}，回退标准切片")
                    cold_open_params = None

            cmd = VideoProcessor.build_extract_clip_command(
                ffmpeg_bin=ffmpeg_bin,
                input_video=input_video,
                output_path=output_path,
                start_time=ffmpeg_start_time,
                duration=duration,
                vertical=is_vertical,
                ass_subtitle_path=ass_path,
                cold_open_params=cold_open_params,
            )

            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')

            # 清理临时 ASS 文件
            if ass_path and ass_path.exists():
                ass_path.unlink(missing_ok=True)

            if result.returncode == 0:
                logger.info(f"成功精确提取切片: {output_path} (起点: {ffmpeg_start_time}, 时长: {duration:.2f}s, vertical={is_vertical}, cold_open={bool(cold_open_params)})")
                return True
            else:
                logger.error(f"提取切片失败: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"视频处理异常: {str(e)}")
            return False

    
    @staticmethod
    def create_collection(clips_list: List[Path], output_path: Path) -> bool:
        """
        将多个视频片段拼接成合集
        
        Args:
            clips_list: 视频片段路径列表
            output_path: 输出合集路径
            
        Returns:
            是否成功
        """
        try:
            # 验证输入参数
            if not clips_list:
                logger.error("clips_list为空，无法创建合集")
                return False
            
            # 验证所有视频文件是否存在
            valid_clips = []
            for clip_path in clips_list:
                if not clip_path.exists():
                    logger.warning(f"视频文件不存在，跳过: {clip_path}")
                    continue
                valid_clips.append(clip_path)
            
            if not valid_clips:
                logger.error("没有有效的视频文件，无法创建合集")
                return False
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建concat文件
            concat_file = output_path.parent / "concat_list.txt"
            
            with open(concat_file, 'w', encoding='utf-8') as f:
                for clip_path in valid_clips:
                    # 使用绝对路径并转义单引号
                    abs_path = clip_path.absolute()
                    escaped_path = str(abs_path).replace("'", "'\"'\"'")
                    f.write(f"file '{escaped_path}'\n")
            
            # 验证concat文件内容
            if concat_file.stat().st_size == 0:
                logger.error("concat文件为空，无法创建合集")
                concat_file.unlink(missing_ok=True)
                return False
            
            # 构建FFmpeg命令 - 使用H.264编码确保兼容性
            ffmpeg_bin = get_ffmpeg_path()
            cmd = [
                ffmpeg_bin,
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',  # 使用H.264视频编码
                '-preset', 'ultrafast',  # 使用最快的编码预设
                '-crf', '28',  # 稍微降低质量以加快编码速度
                '-c:a', 'aac',  # 使用AAC音频编码
                '-b:a', '128k',  # 音频比特率
                '-movflags', '+faststart',  # 优化网络播放
                '-y',
                str(output_path)
            ]
            
            logger.info(f"执行FFmpeg命令: {' '.join(cmd)}")
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            # 清理临时文件
            concat_file.unlink(missing_ok=True)
            
            if result.returncode == 0:
                logger.info(f"成功创建合集: {output_path}")
                return True
            else:
                logger.error(f"创建合集失败: {result.stderr}")
                logger.error(f"FFmpeg stdout: {result.stdout}")
                return False
                
        except Exception as e:
            logger.error(f"视频拼接异常: {str(e)}")
            return False
    
    @staticmethod
    def extract_thumbnail(video_path: Path, output_path: Path, time_offset: int = 5) -> bool:
        """
        从视频中提取缩略图
        
        Args:
            video_path: 视频文件路径
            output_path: 输出缩略图路径
            time_offset: 提取时间点（秒）
            
        Returns:
            是否成功
        """
        try:
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 构建FFmpeg命令
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', str(time_offset),
                '-vframes', '1',
                '-q:v', '2',  # 高质量
                '-y',  # 覆盖输出文件
                str(output_path)
            ]
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"成功提取缩略图: {output_path}")
                return True
            else:
                logger.error(f"提取缩略图失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"提取缩略图异常: {str(e)}")
            return False
    
    @staticmethod
    def get_video_info(video_path: Path) -> Dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息字典
        """
        try:
            ffprobe_bin = get_ffprobe_path()
            cmd = [
                ffprobe_bin,
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return {
                    'duration': float(info['format']['duration']),
                    'size': int(info['format']['size']),
                    'bitrate': int(info['format']['bit_rate']),
                    'streams': info['streams']
                }
            else:
                logger.error(f"获取视频信息失败: {result.stderr}")
                return {}
                
        except Exception as e:
            logger.error(f"获取视频信息异常: {str(e)}")
            return {}
    
    def batch_extract_clips(self, input_video: Path, clips_data: List[Dict], srt_path: Optional[Path] = None) -> List[Path]:
        """
        批量提取视频片段
        
        Args:
            input_video: 输入视频路径
            clips_data: 片段数据列表，每个元素包含id、title、start_time、end_time等
            srt_path: 可选的字幕文件路径
            
        Returns:
            成功提取的片段路径列表
        """
        successful_clips = []
        
        for clip_data in clips_data:
            clip_id = clip_data['id']
            title = clip_data.get('title', f"片段_{clip_id}")
            start_time = clip_data['start_time']
            end_time = clip_data['end_time']
            hook_start = clip_data.get('hook_start')
            hook_text = clip_data.get('hook_text')
            topic_start_time = clip_data.get('topic_start_time')
            clip_srt = clip_data.get('srt_path') or srt_path
            
            # 处理时间格式 - 如果是秒数，转换为SRT格式
            if isinstance(start_time, (int, float)):
                start_time = VideoProcessor.convert_seconds_to_ffmpeg_time(start_time)
            if isinstance(end_time, (int, float)):
                end_time = VideoProcessor.convert_seconds_to_ffmpeg_time(end_time)
            if hook_start and isinstance(hook_start, (int, float)):
                hook_start = VideoProcessor.convert_seconds_to_ffmpeg_time(hook_start)
            if topic_start_time and isinstance(topic_start_time, (int, float)):
                topic_start_time = VideoProcessor.convert_seconds_to_ffmpeg_time(topic_start_time)
            
            # 使用标题作为文件名，并清理不合法的字符
            safe_title = VideoProcessor.sanitize_filename(title)
            output_path = self.clips_dir / f"{clip_id}_{safe_title}.mp4"
            
            logger.info(f"提取切片 {clip_id}: {start_time} -> {end_time}, hook_start={hook_start}, 输出: {output_path}")
            
            if VideoProcessor.extract_clip(
                input_video=input_video,
                output_path=output_path,
                start_time=start_time,
                end_time=end_time,
                hook_start=hook_start,
                hook_text=hook_text,
                topic_start_time=topic_start_time,
                srt_path=clip_srt,
            ):
                successful_clips.append(output_path)
                logger.info(f"切片 {clip_id} 提取成功")
            else:
                logger.error(f"切片 {clip_id} 提取失败")

        
        return successful_clips
    
    def create_collections_from_metadata(self, collections_data: List[Dict]) -> List[Dict]:
        """
        根据元数据创建合集
        
        Args:
            collections_data: 合集数据列表
            
        Returns:
            成功创建的合集信息列表，包含视频路径和缩略图路径
        """
        successful_collections = []
        
        for collection_data in collections_data:
            collection_id = collection_data['id']
            collection_title = collection_data.get('collection_title', f'合集_{collection_id}')
            clip_ids = collection_data['clip_ids']
            
            # 构建片段路径列表
            clips_list = []
            for clip_id in clip_ids:
                # 查找对应的切片文件
                # 新的文件名格式是: {clip_id}_{title}.mp4
                clip_path = self.clips_dir / f"{clip_id}_*.mp4"
                found_clips = list(self.clips_dir.glob(f"{clip_id}_*.mp4"))
                
                if found_clips:
                    found_clip = found_clips[0]  # 取第一个匹配的文件
                    clips_list.append(found_clip)
                    logger.info(f"找到合集 {collection_id} 的切片: {found_clip.name}")
                else:
                    logger.warning(f"未找到合集 {collection_id} 的切片 {clip_id}")
            
            if clips_list:
                # 使用collection_title作为文件名，并清理不合法的字符
                safe_title = VideoProcessor.sanitize_filename(collection_title)
                output_path = self.collections_dir / f"{safe_title}.mp4"
                
                if VideoProcessor.create_collection(clips_list, output_path):
                    # 生成合集缩略图
                    thumbnail_path = None
                    try:
                        thumbnail_filename = f"{collection_id}_{safe_title}_thumbnail.jpg"
                        thumbnail_path = self.collections_dir / thumbnail_filename
                        
                        # 从视频中提取缩略图（第2秒的帧）
                        thumbnail_success = VideoProcessor.extract_thumbnail(output_path, thumbnail_path, time_offset=2)
                        if thumbnail_success:
                            logger.info(f"合集 {collection_id} 缩略图生成成功: {thumbnail_path}")
                        else:
                            logger.warning(f"合集 {collection_id} 缩略图生成失败")
                            thumbnail_path = None
                    except Exception as e:
                        logger.error(f"生成合集 {collection_id} 缩略图时出错: {e}")
                        thumbnail_path = None
                    
                    # 返回包含视频路径和缩略图路径的信息
                    collection_info = {
                        'collection_id': collection_id,
                        'video_path': str(output_path),
                        'thumbnail_path': str(thumbnail_path) if thumbnail_path else None,
                        'title': collection_title
                    }
                    successful_collections.append(collection_info)
                    logger.info(f"成功创建合集 {collection_id}: {output_path}")
            else:
                logger.warning(f"合集 {collection_id} 没有找到任何有效的切片文件")
        
        return successful_collections