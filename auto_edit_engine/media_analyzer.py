"""
media_analyzer.py — المرحلة 3 (أ): فحص صور وفيديوهات المستخدم
بدون نماذج ثقيلة: OpenCV + Pillow فقط.

لكل ملف:
  - هل يُقرأ؟ (الملفات التالفة تُستبعد مع سبب واضح)
  - الأبعاد والاتجاه (عمودي/أفقي/مربع) والمدة
  - الجودة (حدّة + إضاءة + دقة) من 0 إلى 1
  - إطارات مختارة (للصور: إطار واحد، للفيديو: عدة إطارات موزعة) تُحفظ كصور صغيرة
    ليستخدمها clip_matcher مع OpenCLIP بدل تحليل كل إطار
  - للفيديو: best_start = أفضل نقطة بداية للقصّ (بدل البدء دائمًا من الثانية 0)
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

DEFAULT_SETTINGS = {
    "frames_per_video": 5,     # عدد الإطارات المستخرجة من كل فيديو
    "frame_max_side":   512,   # أكبر ضلع للإطار المحفوظ (يكفي لـ OpenCLIP)
    "min_side_px":      240,   # أقل دقة مقبولة (أصغر ضلع)
    "min_video_sec":    0.5,   # أقصر فيديو مقبول
    "window_sec":       2.0,   # طول النافذة المستخدمة لاختيار best_start
}


def _merge_settings(user):
    s = dict(DEFAULT_SETTINGS)
    for k, v in (user or {}).items():
        if k in s and v is not None:
            s[k] = type(s[k])(v)
    return s


# ── قياس الجودة ───────────────────────────────────────────────────────────────

def _frame_metrics(bgr):
    """حدّة وسطوع إطار واحد، ودرجة جودة 0..1."""
    h, w = bgr.shape[:2]
    scale = 480 / max(h, w)
    small = cv2.resize(bgr, (max(2, int(w * scale)), max(2, int(h * scale)))) if scale < 1 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean()) / 255

    sharp_score = min(1.0, np.log1p(sharp) / np.log1p(300))        # 300+ = حاد
    expo_score = 1.0 - min(1.0, abs(bright - 0.5) / 0.5) ** 2      # الأفضل قرب 0.5
    quality = 0.65 * sharp_score + 0.35 * expo_score
    return {"sharpness": round(sharp, 2), "brightness": round(bright, 3),
            "quality": round(float(quality), 3)}


def _resolution_score(w, h, min_side):
    return float(min(1.0, min(w, h) / 720)) if min(w, h) >= min_side else 0.0


def _orientation(w, h):
    return "vertical" if h > w else "horizontal" if w > h else "square"


def _save_frame(bgr, path, max_side):
    h, w = bgr.shape[:2]
    sc = max_side / max(h, w)
    if sc < 1:
        bgr = cv2.resize(bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ── فحص عنصر واحد ─────────────────────────────────────────────────────────────

def _analyze_image(path, mid, frames_dir, s):
    # exif_transpose: صور الموبايل المقلوبة تُصحَّح قبل القياس
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    if min(w, h) < s["min_side_px"]:
        return {"status": "rejected", "reason": f"low_resolution_{w}x{h}"}
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    m = _frame_metrics(bgr)
    fp = frames_dir / f"{mid}_0.jpg"
    _save_frame(bgr, fp, s["frame_max_side"])
    q = 0.8 * m["quality"] + 0.2 * _resolution_score(w, h, s["min_side_px"])
    return {"status": "ok", "type": "image", "width": w, "height": h,
            "orientation": _orientation(w, h), "duration": None,
            "quality": round(q, 3),
            "frames": [{"t": 0.0, "path": str(fp), **m}], "best_start": 0.0}


def _analyze_video(path, mid, frames_dir, s):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"status": "rejected", "reason": "cannot_open"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total / fps if fps else 0
    if dur < s["min_video_sec"] or not w or not h:
        cap.release()
        return {"status": "rejected", "reason": "too_short_or_unreadable"}
    if min(w, h) < s["min_side_px"]:
        cap.release()
        return {"status": "rejected", "reason": f"low_resolution_{w}x{h}"}

    n = s["frames_per_video"]
    # نتجنب أول وآخر 5% (غالبًا إطارات سوداء أو غير ثابتة)
    times = [dur * (0.05 + 0.9 * (i + 0.5) / n) for i in range(n)]
    frames = []
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, bgr = cap.read()
        if not ok:
            continue
        m = _frame_metrics(bgr)
        fp = frames_dir / f"{mid}_{i}.jpg"
        _save_frame(bgr, fp, s["frame_max_side"])
        frames.append({"t": round(t, 3), "path": str(fp), **m})
    cap.release()
    if not frames:
        return {"status": "rejected", "reason": "no_readable_frames"}

    # best_start: النافذة التي متوسط جودة إطاراتها الأعلى
    win = min(s["window_sec"], dur)
    best_t, best_q = 0.0, -1.0
    for f in frames:
        start = min(max(0.0, f["t"] - win / 2), max(0.0, dur - win))
        inside = [g["quality"] for g in frames if start <= g["t"] <= start + win]
        q = float(np.mean(inside)) if inside else f["quality"]
        if q > best_q:
            best_q, best_t = q, start

    mean_q = float(np.mean([f["quality"] for f in frames]))
    q = 0.8 * mean_q + 0.2 * _resolution_score(w, h, s["min_side_px"])
    return {"status": "ok", "type": "video", "width": w, "height": h,
            "orientation": _orientation(w, h), "duration": round(dur, 3),
            "quality": round(q, 3), "frames": frames, "best_start": round(best_t, 3)}


# ── الدالة الرئيسية ───────────────────────────────────────────────────────────

def analyze_media(paths, out_dir, settings=None, progress_cb=None):
    """
    paths   : قائمة مسارات ملفات المستخدم (صور وفيديوهات)
    out_dir : مجلد التحليل (يُنشأ فيه media_frames/ و media.json)
    """
    out_dir = Path(out_dir)
    frames_dir = out_dir / "media_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    s = _merge_settings(settings)
    cb = progress_cb or (lambda p: None)

    items, rejected = [], []
    paths = [Path(p) for p in paths]
    for i, p in enumerate(paths):
        mid = f"m{i:03d}"
        ext = p.suffix.lower()
        base = {"id": mid, "name": p.name, "path": str(p)}
        try:
            if not p.exists():
                res = {"status": "rejected", "reason": "file_not_found"}
            elif ext in IMAGE_EXT:
                res = _analyze_image(p, mid, frames_dir, s)
            elif ext in VIDEO_EXT:
                res = _analyze_video(p, mid, frames_dir, s)
            else:
                res = {"status": "rejected", "reason": f"unsupported_{ext or 'noext'}"}
        except Exception as e:
            res = {"status": "rejected", "reason": f"error: {str(e)[:120]}"}

        entry = {**base, **res}
        (items if res["status"] == "ok" else rejected).append(entry)
        cb(int((i + 1) / max(len(paths), 1) * 100))

    n_img = sum(1 for m in items if m["type"] == "image")
    result = {
        "version": 1, "settings": s,
        "items": items, "rejected": rejected,
        "summary": {
            "total": len(paths), "accepted": len(items), "rejected": len(rejected),
            "images": n_img, "videos": len(items) - n_img,
            "vertical":   sum(1 for m in items if m["orientation"] == "vertical"),
            "horizontal": sum(1 for m in items if m["orientation"] == "horizontal"),
            "avg_quality": round(float(np.mean([m["quality"] for m in items])), 3) if items else 0.0,
        },
    }
    (out_dir / "media.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python media_analyzer.py <out_dir> <file_or_folder> [...]")
        sys.exit(1)
    files = []
    for a in sys.argv[2:]:
        p = Path(a)
        files += sorted(x for x in p.iterdir() if x.is_file()) if p.is_dir() else [p]
    res = analyze_media(files, sys.argv[1], progress_cb=lambda p: print(f"{p}%", end="\r"))
    print()
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
    for m in res["items"]:
        print(f"{m['id']} {m['name'][:28]:28} {m['type']:5} {m['orientation']:10} "
              f"q={m['quality']:.2f} best_start={m['best_start']}")
    for m in res["rejected"]:
        print(f"✗ {m['name'][:28]:28} {m['reason']}")