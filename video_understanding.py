"""
video_understanding.py — دماغ تحليل الفيديو
يستخدم Qwen2-VL-7B-Instruct لفهم الفيديو المرجعي بعمق
Fallback: PySceneDetect بدون AI
"""

import json
import re
import shutil
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

# ── Qwen2-VL (lazy load) ──────────────────────────────────────────────────────
_qwen_model     = None
_qwen_processor = None
_qwen_lock      = threading.Lock()
QWEN_OK         = False

def _load_qwen():
    global _qwen_model, _qwen_processor, QWEN_OK
    with _qwen_lock:
        if _qwen_model is not None:
            return True
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
            )
            _qwen_processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
            )
            QWEN_OK = True
            return True
        except Exception as e:
            print(f"[Qwen2-VL load] {e}")
            return False

# ── Inference ─────────────────────────────────────────────────────────────────

def _qwen_infer(messages, max_new_tokens=1000):
    """يشغّل Qwen2-VL على messages ويرجع النص الناتج."""
    import torch

    # محاولة استخدام qwen_vl_utils إذا متاحة
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
    except ImportError:
        from PIL import Image as _PIL_Image
        image_inputs = []
        for msg in messages:
            for part in msg.get("content", []):
                if part.get("type") == "image":
                    image_inputs.append(_PIL_Image.open(part["image"]).convert("RGB"))
        video_inputs = None

    text = _qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _qwen_processor(
        text=[text],
        images=image_inputs or None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_qwen_model.device)

    with torch.no_grad():
        output_ids = _qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[:, input_len:]
    return _qwen_processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _parse_json(text):
    """استخرج أول JSON object من النص."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None

# ── Frame Extraction ──────────────────────────────────────────────────────────

def _extract_frames(video_path, max_frames=12):
    """يسحب فريمات ذكية — كل ثانية أو بحد أقصى max_frames."""
    cap   = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur   = total / fps

    step    = max(1, int(total / min(max_frames, max(1, int(dur)))))
    frames  = []
    tmp_dir = Path(tempfile.mkdtemp())

    fn = 0
    while fn < total and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, frame = cap.read()
        if not ok:
            break
        out = tmp_dir / f"frame_{len(frames):03d}.jpg"
        cv2.imwrite(str(out), cv2.resize(frame, (640, 360)))
        frames.append({"path": str(out), "ts": round(fn / fps, 2)})
        fn += step

    cap.release()
    return frames, tmp_dir

# ── Reference Video Analysis ──────────────────────────────────────────────────

def analyze_reference_video(video_path):
    """
    يحلل الفيديو المرجعي ويرجع JSON:
    {
      "story": "...",
      "emotional_arc": "...",
      "segments": [
        {
          "index": 0, "start_sec": 0.0, "end_sec": 2.5,
          "type": "تعريف", "purpose": "...",
          "energy": "عالي", "needs_motion": true, "needs_speech": false
        }
      ],
      "success_factors": ["..."]
    }
    """
    if not _load_qwen():
        return _fallback_analysis(video_path)

    frames, tmp_dir = _extract_frames(video_path, max_frames=12)
    if not frames:
        return _fallback_analysis(video_path)

    try:
        content = []
        for f in frames:
            content.append({"type": "text", "text": f"[فريم عند {f['ts']}ث]"})
            content.append({"type": "image", "image": f["path"]})

        content.append({
            "type": "text",
            "text": (
                "حلل هذا الفيديو واستخرج المعلومات التالية بصيغة JSON فقط بدون أي تفسير:\n"
                "{\n"
                '  "story": "القصة العامة والهدف",\n'
                '  "emotional_arc": "كيف يبدأ، يتطور، وينتهي عاطفياً",\n'
                '  "segments": [\n'
                '    {\n'
                '      "index": 0, "start_sec": 0.0, "end_sec": 2.5,\n'
                '      "type": "تعريف", "purpose": "جذب الانتباه",\n'
                '      "energy": "عالي", "needs_motion": true, "needs_speech": false\n'
                '    }\n'
                '  ],\n'
                '  "success_factors": ["..."]\n'
                "}"
            ),
        })

        raw    = _qwen_infer([{"role": "user", "content": content}], max_new_tokens=1200)
        result = _parse_json(raw)
        if result and "segments" in result:
            return result

    except Exception as e:
        print(f"[analyze_reference_video] {e}")
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    return _fallback_analysis(video_path)


def _fallback_analysis(video_path):
    """تحليل بسيط بدون Qwen2-VL — يستخدم PySceneDetect/OpenCV."""
    try:
        from video_maker import detect_scenes, get_video_info
        scenes = detect_scenes(str(video_path), tempfile.mkdtemp())
    except Exception:
        return {"story": "", "emotional_arc": "", "segments": [], "success_factors": []}

    n = len(scenes)
    segments = []
    for i, sc in enumerate(scenes):
        if i == 0:
            seg_type = "تعريف"
        elif i == n - 1:
            seg_type = "ختام"
        else:
            seg_type = "تفصيل"
        segments.append({
            "index":        i,
            "start_sec":    sc["start_sec"],
            "end_sec":      sc["end_sec"],
            "type":         seg_type,
            "purpose":      "",
            "energy":       "متوسط",
            "needs_motion": True,
            "needs_speech": False,
        })

    return {
        "story":           "",
        "emotional_arc":   "",
        "segments":        segments,
        "success_factors": [],
    }

# ── Media Item Analysis ───────────────────────────────────────────────────────

def analyze_media_item(path, video_analysis):
    """
    يحلل صورة أو مقطع ويقرر أين يناسب في الفيديو المرجعي.
    ترجع: {"best_segment_type", "energy_level", "suitable_positions", "confidence"}
    """
    _default = {
        "best_segment_type":  "تفصيل",
        "energy_level":       "متوسط",
        "suitable_positions": ["وسط"],
        "confidence":         0.5,
    }

    if not _load_qwen():
        return _default

    story = video_analysis.get("story", "محتوى عام")
    ext   = Path(path).suffix.lower()
    is_video = ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm')

    tmp = None
    if is_video:
        tmp = Path(tempfile.mktemp(suffix='.jpg'))
        try:
            cap = cv2.VideoCapture(str(path))
            mid_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ok, frame = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(str(tmp), cv2.resize(frame, (640, 360)))
            else:
                tmp = None
        except Exception:
            tmp = None

    img_path = str(tmp) if tmp and tmp.exists() else str(path)

    try:
        content = [
            {"type": "image", "image": img_path},
            {
                "type": "text",
                "text": (
                    f"الفيديو المرجعي يهدف إلى: {story}\n"
                    "هل هذه الصورة/المقطع مناسب لـ: تعريف / تفصيل / حركة / كلام / ختام؟\n"
                    "أجب بـ JSON فقط:\n"
                    "{\n"
                    '  "best_segment_type": "تعريف",\n'
                    '  "energy_level": "عالي",\n'
                    '  "suitable_positions": ["بداية"],\n'
                    '  "confidence": 0.85\n'
                    "}"
                ),
            },
        ]

        raw    = _qwen_infer([{"role": "user", "content": content}], max_new_tokens=200)
        result = _parse_json(raw)
        if result and "best_segment_type" in result:
            return result

    except Exception as e:
        print(f"[analyze_media_item] {e}")
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)

    return _default

# ── Intelligent Matching ──────────────────────────────────────────────────────

def match_media_to_analysis(video_analysis, media_items):
    """
    يربط كل segment بالميديا الأنسب بناءً على تحليل Qwen2-VL.
    media_items : [{"path", "type", "analysis": {...}}, ...]
    ترجع       : [media_index per segment]
    """
    segments = video_analysis.get("segments", [])
    n_segs   = len(segments)
    n_media  = len(media_items)

    if n_media == 0:
        return [None] * n_segs

    # بناء قاموس: نوع الـ segment → [(confidence, media_index)]
    type_to_media: dict = {}
    for mi, m in enumerate(media_items):
        analysis = m.get("analysis", {})
        seg_type = analysis.get("best_segment_type", "تفصيل")
        conf     = float(analysis.get("confidence", 0.5))
        type_to_media.setdefault(seg_type, []).append((conf, mi))

    for k in type_to_media:
        type_to_media[k].sort(reverse=True)  # تنازلي حسب الـ confidence

    used   = set()
    result = []

    for seg in segments:
        seg_type = seg.get("type", "تفصيل")
        chosen   = None

        # ابحث عن أنسب ميديا غير مستخدمة لهذا النوع
        for _, mi in type_to_media.get(seg_type, []):
            if mi not in used:
                chosen = mi
                break

        # fallback: أي ميديا غير مستخدمة بالترتيب
        if chosen is None:
            for mi in range(n_media):
                if mi not in used:
                    chosen = mi
                    break

        # إعادة التدوير لو الكل انحرق
        if chosen is None:
            chosen = len(result) % n_media

        result.append(chosen)
        used.add(chosen)

    return result
