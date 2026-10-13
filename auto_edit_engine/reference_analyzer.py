"""
reference_analyzer.py — المرحلة 1: تحليل الفيديو المرجعي
يعتمد على PySceneDetect + OpenCV فقط (بدون نماذج ثقيلة، يشتغل على CPU).

المخرجات: قاموس (ويُحفظ كـ reference.json) فيه:
  video   : المدة، FPS، الأبعاد، هل فيه صوت
  scenes  : لكل مشهد -> التوقيت، المدة، الألوان، الحركة، الانتقال المقدّر، صورة معاينة
  summary : متوسط المدة، سرعة الإيقاع، متوسط الألوان
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_OK = True
except Exception:
    SCENEDETECT_OK = False

# قيم افتراضية قابلة للتعديل من الواجهة
DEFAULT_SETTINGS = {
    "threshold":       27.0,   # حساسية كشف القصّ (أقل = يكشف قصّات أكثر)
    "min_scene_len":   0.4,    # أقل مدة للمشهد بالثواني
    "max_scenes":      60,     # سقف عدد المشاهد
    "motion_samples":  6,      # عدد الإطارات المقاسة لحساب الحركة بكل مشهد
    "thumb_width":     320,    # عرض صورة المعاينة
}


# ── أدوات مساعدة ──────────────────────────────────────────────────────────────

def _merge_settings(user):
    s = dict(DEFAULT_SETTINGS)
    for k, v in (user or {}).items():
        if k in s and v is not None:
            s[k] = type(s[k])(v)
    return s


def probe_video(path):
    """معلومات الفيديو الأساسية. ffprobe أولًا، وOpenCV احتياطيًا."""
    info = {"duration": 0.0, "fps": 30.0, "width": 0, "height": 0, "has_audio": False}
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=30)
        data = json.loads(r.stdout or "{}")
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and not info["width"]:
                n, _, d = (s.get("r_frame_rate", "30/1")).partition("/")
                info["fps"] = int(n) / max(int(d or 1), 1)
                info["width"] = int(s.get("width", 0))
                info["height"] = int(s.get("height", 0))
                info["duration"] = float(s.get("duration") or 0)
            elif s.get("codec_type") == "audio":
                info["has_audio"] = True
        if not info["duration"]:
            info["duration"] = float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass

    if not info["width"] or not info["duration"]:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        info.update(fps=fps,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    duration=total / fps if fps else 0.0)
        cap.release()

    info["fps"] = round(info["fps"], 3)
    info["duration"] = round(info["duration"], 3)
    info["aspect"] = round(info["width"] / info["height"], 3) if info["height"] else 0
    info["orientation"] = ("vertical" if info["height"] > info["width"]
                           else "horizontal" if info["width"] > info["height"] else "square")
    return info


def _read_frame_at(cap, sec, fps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(sec * fps)))
    ok, frame = cap.read()
    return frame if ok else None


# ── كشف المشاهد ───────────────────────────────────────────────────────────────

def _detect_scenedetect(path, s, fps):
    vid = open_video(str(path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=s["threshold"],
                                    min_scene_len=max(1, int(s["min_scene_len"] * fps))))
    sm.detect_scenes(vid)
    sec = lambda t: t.seconds if hasattr(t, "seconds") else t.get_seconds()  # 0.6.x و0.7.x
    return [(sec(a), sec(b)) for a, b in sm.get_scene_list()]


def _detect_opencv(path, s, fps, duration):
    """احتياطي: فرق الهيستوغرام بين إطارات متتالية (4 إطارات بالثانية)."""
    cap = cv2.VideoCapture(str(path))
    step = max(1, int(fps / 4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cuts, prev, last_cut = [], None, 0.0
    for fn in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (160, 90))
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None,
                            [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev is not None:
            dist = cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA)
            t = fn / fps
            if dist > 0.45 and (t - last_cut) >= s["min_scene_len"]:
                cuts.append(t)
                last_cut = t
        prev = hist
    cap.release()
    bounds = [0.0] + cuts + [duration]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def detect_scene_bounds(path, settings, info):
    scenes, method = [], "none"
    if SCENEDETECT_OK:
        try:
            scenes = _detect_scenedetect(path, settings, info["fps"])
            method = "scenedetect"
        except Exception as e:
            print(f"[reference_analyzer] scenedetect failed: {e}")
    if not scenes:
        scenes = _detect_opencv(path, settings, info["fps"], info["duration"])
        method = "opencv"
    if not scenes:  # فيديو بدون أي قصّة = مشهد واحد
        scenes = [(0.0, info["duration"])]
        method = "single"
    return scenes[: settings["max_scenes"]], method


# ── خصائص كل مشهد ─────────────────────────────────────────────────────────────

def _color_stats(frame):
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    b, g, r = [float(small[:, :, i].mean()) for i in range(3)]
    return {
        "mean_rgb":   [round(r), round(g), round(b)],
        "brightness": round(float(gray.mean()) / 255, 3),
        "contrast":   round(float(gray.std()) / 128, 3),
        "saturation": round(float(hsv[:, :, 1].mean()) / 255, 3),
        "warmth":     round((r - b) / 255, 3),   # موجب = دافئ، سالب = بارد
    }


def _motion_score(cap, start, end, fps, n):
    """متوسط فرق الإطارات المتتالية داخل المشهد (0..1). أعلى = حركة أكثر."""
    dur = end - start
    if dur <= 0.1:
        return 0.0
    gap = min(0.15, dur / (n + 1))
    diffs = []
    for i in range(n):
        t = start + dur * (i + 0.5) / n
        a = _read_frame_at(cap, t, fps)
        b = _read_frame_at(cap, min(t + gap, end - 0.01), fps)
        if a is None or b is None:
            continue
        ga = cv2.cvtColor(cv2.resize(a, (160, 90)), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(b, (160, 90)), cv2.COLOR_BGR2GRAY)
        diffs.append(float(cv2.absdiff(ga, gb).mean()) / 255)
    return round(float(np.mean(diffs)) * 10, 3) if diffs else 0.0  # مضروبة بـ10 لتوضيح الفرق


def _guess_transition(cap, start, fps, prev_end):
    """
    تقدير الانتقال عند بداية المشهد من سطوع أول الإطارات.
    تقدير فقط: الانتقالات المعقدة ما يمكن تحديدها بدقة من MP4 نهائي.
    """
    if start <= 0.05:
        return {"type": "start", "confidence": 1.0}
    a = _read_frame_at(cap, max(0, start - 0.04), fps)
    b = _read_frame_at(cap, start + 0.04, fps)
    c = _read_frame_at(cap, start + 0.3, fps)
    if a is None or b is None:
        return {"type": "cut", "confidence": 0.3}
    br = lambda f: float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean())
    ba, bb = br(a), br(b)
    bc = br(c) if c is not None else bb
    if bb < 25 and bc > bb + 30:
        return {"type": "fade_from_black", "confidence": 0.6}
    if ba < 25 and bb > ba + 30:
        return {"type": "fade_from_black", "confidence": 0.6}
    if bb > 235 or ba > 235:
        return {"type": "flash", "confidence": 0.5}
    return {"type": "cut", "confidence": 0.7}


def _pace_label(avg_dur):
    if avg_dur < 1.2:
        return "very_fast"
    if avg_dur < 2.5:
        return "fast"
    if avg_dur < 4.5:
        return "medium"
    return "slow"


# ── الدالة الرئيسية ───────────────────────────────────────────────────────────

def analyze_reference(video_path, out_dir, settings=None, progress_cb=None):
    """
    video_path  : مسار الفيديو المرجعي
    out_dir     : مجلد التحليل (مثلًا sessions/<id>/analysis)
    settings    : قاموس اختياري يغيّر DEFAULT_SETTINGS
    progress_cb : دالة تستقبل نسبة 0..100
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    thumbs_dir = out_dir / "ref_thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    s = _merge_settings(settings)
    cb = progress_cb or (lambda p: None)

    if not video_path.exists():
        raise FileNotFoundError(f"الفيديو المرجعي غير موجود: {video_path}")

    info = probe_video(video_path)
    if info["duration"] <= 0 or not info["width"]:
        raise ValueError("تعذّر قراءة الفيديو المرجعي (ملف تالف أو صيغة غير مدعومة)")
    cb(5)

    bounds, method = detect_scene_bounds(video_path, s, info)
    cb(35)

    cap = cv2.VideoCapture(str(video_path))
    fps = info["fps"]
    scenes = []
    for i, (a, b) in enumerate(bounds):
        dur = b - a
        mid = a + dur * 0.5
        frame = _read_frame_at(cap, mid, fps)
        thumb_name = f"scene_{i:03d}.jpg"
        stats = {}
        if frame is not None:
            h, w = frame.shape[:2]
            tw = s["thumb_width"]
            cv2.imwrite(str(thumbs_dir / thumb_name),
                        cv2.resize(frame, (tw, max(2, int(h * tw / w)))),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            stats = _color_stats(frame)
        scenes.append({
            "index":      i,
            "start":      round(a, 3),
            "end":        round(b, 3),
            "duration":   round(dur, 3),
            "thumbnail":  thumb_name if frame is not None else None,
            "colors":     stats,
            "motion":     _motion_score(cap, a, b, fps, s["motion_samples"]),
            "transition_in": _guess_transition(cap, a, fps, bounds[i - 1][1] if i else 0),
        })
        cb(35 + int((i + 1) / len(bounds) * 60))
    cap.release()

    durs = [sc["duration"] for sc in scenes]
    avg = float(np.mean(durs))
    color_keys = ("brightness", "contrast", "saturation", "warmth")
    avg_colors = {k: round(float(np.mean([sc["colors"].get(k, 0) for sc in scenes])), 3)
                  for k in color_keys}

    result = {
        "version":  1,
        "source":   str(video_path),
        "settings": s,
        "method":   method,
        "video":    info,
        "scenes":   scenes,
        "summary": {
            "scene_count":   len(scenes),
            "avg_duration":  round(avg, 3),
            "min_duration":  round(min(durs), 3),
            "max_duration":  round(max(durs), 3),
            "cuts_per_min":  round(len(scenes) / info["duration"] * 60, 1),
            "pace":          _pace_label(avg),
            "avg_motion":    round(float(np.mean([sc["motion"] for sc in scenes])), 3),
            "avg_colors":    avg_colors,
        },
    }
    (out_dir / "reference.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    cb(100)
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python reference_analyzer.py <video> <out_dir> [threshold]")
        sys.exit(1)
    st = {"threshold": float(sys.argv[3])} if len(sys.argv) > 3 else None
    res = analyze_reference(sys.argv[1], sys.argv[2], st, lambda p: print(f"{p}%", end="\r"))
    print()
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
    for sc in res["scenes"]:
        print(f"#{sc['index']:02d}  {sc['start']:6.2f}-{sc['end']:6.2f}  "
              f"({sc['duration']:.2f}s)  motion={sc['motion']:.2f}  "
              f"in={sc['transition_in']['type']}")