"""
audio_analyzer.py — المرحلة 2: تحليل الموسيقى والإيقاع
يعتمد على librosa + FFmpeg (يشتغل على CPU).

يقبل فيديو أو ملف صوت (الفيديو المرجعي أو موسيقى المستخدم)، وينتج:
  tempo      : BPM مقدّر (مع تنبيه لاحتمال نصف/ضعف السرعة)
  beats      : أوقات النبضات
  onsets     : أوقات الارتفاعات الصوتية القوية (ضربات، تغيّرات)
  energy     : مستوى الطاقة كل ثانية + تصنيف low/mid/high
  cut_points : نقاط القصّ المرشّحة مع قوة كل نقطة (0..1)
ثم الدالة suggest_cuts تحوّل النقاط لتوقيتات لقطات بحسب سرعة إيقاع المرجعي.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

try:
    import librosa
    LIBROSA_OK = True
except Exception:
    LIBROSA_OK = False

# قيم افتراضية قابلة للتعديل من الواجهة
DEFAULT_SETTINGS = {
    "sample_rate":    22050,
    "beat_mode":      "beats",  # beats = كل نبضة | every_2 = كل نبضتين | downbeats = كل 4 نبضات
    "onset_delta":    0.2,      # حساسية الـ onsets (أقل = أكثر نقاط)
    "min_cut_gap":    0.35,     # أقل فاصل بين نقطتي قصّ بالثواني
    "use_onsets":     True,     # إضافة الارتفاعات الصوتية لنقاط القصّ
    "max_duration":   600,      # أقصى مدة تُحلَّل بالثواني (حماية الذاكرة)
}


def _merge_settings(user):
    s = dict(DEFAULT_SETTINGS)
    for k, v in (user or {}).items():
        if k in s and v is not None:
            s[k] = type(s[k])(v)
    return s


def has_audio_stream(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return "audio" in r.stdout
    except Exception:
        return False


def extract_audio(src, out_wav, sr=22050):
    """يستخرج صوت mono WAV مؤقتًا للتحليل (لا يغيّر الملف الأصلي)."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn",
         "-ac", "1", "-ar", str(sr), "-acodec", "pcm_s16le", str(out_wav)],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "ffmpeg error")[-400:])
    return str(out_wav)


def _norm(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)


def _energy_curve(y, sr):
    """طاقة RMS لكل ثانية + تصنيف نسبي."""
    hop = sr  # ثانية واحدة
    rms = librosa.feature.rms(y=y, frame_length=hop, hop_length=hop)[0]
    n = _norm(rms)
    lo, hi = np.percentile(n, 33), np.percentile(n, 66)
    labels = ["low" if v <= lo else "high" if v >= hi else "mid" for v in n]
    return [{"t": i, "level": round(float(v), 3), "label": l}
            for i, (v, l) in enumerate(zip(n, labels))]


def _select_beats(beat_times, mode):
    if mode == "every_2":
        return beat_times[::2]
    if mode == "downbeats":
        return beat_times[::4]
    return beat_times


def _build_cut_points(beats, onsets, onset_strengths, min_gap, use_onsets):
    """يدمج النبضات والـ onsets بقائمة واحدة، ويحذف النقاط المتقاربة (يبقي الأقوى)."""
    pts = [{"t": float(t), "kind": "beat", "strength": 0.6} for t in beats]
    if use_onsets:
        for t, st in zip(onsets, onset_strengths):
            pts.append({"t": float(t), "kind": "onset", "strength": float(0.4 + 0.6 * st)})
    pts.sort(key=lambda p: p["t"])

    kept = []
    for p in pts:
        if kept and p["t"] - kept[-1]["t"] < min_gap:
            if p["strength"] > kept[-1]["strength"]:
                kept[-1] = p
            continue
        kept.append(p)
    for p in kept:
        p["t"] = round(p["t"], 3)
        p["strength"] = round(p["strength"], 3)
    return kept


def analyze_audio(src, out_dir, settings=None, progress_cb=None):
    """
    src         : فيديو أو ملف صوت
    out_dir     : مجلد التحليل (يُحفظ فيه audio.json)
    """
    if not LIBROSA_OK:
        raise RuntimeError("librosa غير مثبّت. نفّذ: pip install librosa soundfile")

    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = _merge_settings(settings)
    cb = progress_cb or (lambda p: None)

    if not src.exists():
        raise FileNotFoundError(f"الملف غير موجود: {src}")

    if not has_audio_stream(src):
        result = {"version": 1, "source": str(src), "available": False,
                  "reason": "no_audio_stream", "cut_points": []}
        (out_dir / "audio.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        cb(100)
        return result
    cb(5)

    wav = out_dir / "_analysis_audio.wav"
    try:
        extract_audio(src, wav, s["sample_rate"])
        cb(20)
        y, sr = librosa.load(str(wav), sr=s["sample_rate"], mono=True,
                             duration=s["max_duration"])
    finally:
        wav.unlink(missing_ok=True)

    duration = float(len(y) / sr)
    if duration < 1.0 or float(np.abs(y).max()) < 1e-4:
        result = {"version": 1, "source": str(src), "available": False,
                  "reason": "silent_or_too_short", "duration": round(duration, 3),
                  "cut_points": []}
        (out_dir / "audio.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        cb(100)
        return result

    # الإيقاع
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    cb(55)

    # الارتفاعات الصوتية
    on_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                           delta=s["onset_delta"], backtrack=False)
    on_times = librosa.frames_to_time(on_frames, sr=sr)
    on_str = _norm(onset_env[on_frames]) if len(on_frames) else np.array([])
    cb(75)

    beats_sel = _select_beats(beat_times, s["beat_mode"])
    cuts = _build_cut_points(beats_sel, on_times, on_str, s["min_cut_gap"], s["use_onsets"])

    result = {
        "version":   1,
        "source":    str(src),
        "available": True,
        "settings":  s,
        "duration":  round(duration, 3),
        "tempo": {
            "bpm":        round(tempo, 1),
            "beat_count": int(len(beat_times)),
            # كشف BPM أحيانًا يطلع نصف أو ضعف الحقيقي، فنعرض البدائل للمراجعة
            "alternatives": [round(tempo / 2, 1), round(tempo * 2, 1)],
        },
        "beats":      [round(float(t), 3) for t in beat_times],
        "onsets":     [round(float(t), 3) for t in on_times],
        "energy":     _energy_curve(y, sr),
        "cut_points": cuts,
    }
    (out_dir / "audio.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    cb(100)
    return result


# ── من نقاط القصّ إلى توقيتات لقطات ───────────────────────────────────────────

def suggest_cuts(audio, target_duration, avg_shot_len, min_len=0.5, max_len=8.0):
    """
    يقترح توقيت كل لقطة بحيث:
      - متوسط مدتها قريب من avg_shot_len (خذه من summary.avg_duration للمرجعي)
      - كل قصّة تقع على نقطة قصّ موسيقية قدر الإمكان

    يرجع قائمة [{start, end, duration, snapped, strength}] تغطي target_duration.
    إذا ما في صوت (available=False)، يوزّع اللقطات بمدد متساوية.
    """
    shots, t = [], 0.0

    if not audio.get("available") or not audio.get("cut_points"):
        n = max(1, round(target_duration / max(avg_shot_len, min_len)))
        step = target_duration / n
        return [{"start": round(i * step, 3), "end": round((i + 1) * step, 3),
                 "duration": round(step, 3), "snapped": False, "strength": 0.0}
                for i in range(n)]

    pts = audio["cut_points"]
    while t < target_duration - 1e-6:
        ideal = t + avg_shot_len
        lo, hi = t + min_len, min(t + max_len, target_duration)
        cands = [p for p in pts if lo <= p["t"] <= hi]
        if cands and ideal < target_duration - min_len:
            # أقرب نقطة للمدة المثالية، مع تفضيل بسيط للقوية
            best = min(cands, key=lambda p: abs(p["t"] - ideal) - 0.3 * p["strength"])
            end, snapped, st = best["t"], True, best["strength"]
        else:
            end, snapped, st = min(ideal, target_duration), False, 0.0
        if target_duration - end < min_len:   # لا تترك لقطة أخيرة قصيرة جدًا
            end, snapped = target_duration, False
        shots.append({"start": round(t, 3), "end": round(end, 3),
                      "duration": round(end - t, 3), "snapped": snapped,
                      "strength": round(st, 3)})
        t = end
    return shots


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python audio_analyzer.py <video_or_audio> <out_dir> [avg_shot_len]")
        sys.exit(1)
    res = analyze_audio(sys.argv[1], sys.argv[2], progress_cb=lambda p: print(f"{p}%", end="\r"))
    print()
    if not res["available"]:
        print("لا يوجد صوت قابل للتحليل:", res["reason"])
        sys.exit(0)
    print(f"BPM: {res['tempo']['bpm']}  (بدائل محتملة: {res['tempo']['alternatives']})")
    print(f"النبضات: {res['tempo']['beat_count']} | الـ onsets: {len(res['onsets'])} "
          f"| نقاط القصّ: {len(res['cut_points'])}")
    avg = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    shots = suggest_cuts(res, res["duration"], avg)
    print(f"\nاقتراح اللقطات (متوسط مستهدف {avg}s): {len(shots)} لقطة، "
          f"{sum(s['snapped'] for s in shots)} منها على نقطة موسيقية")
    for i, sh in enumerate(shots[:15]):
        print(f"#{i:02d} {sh['start']:6.2f}-{sh['end']:6.2f} ({sh['duration']:.2f}s) "
              f"{'♪' if sh['snapped'] else '-'}")