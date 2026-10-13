"""
أداة صنع الفيديو — video_maker.py
Assembly بـ MoviePy (fallback: FFmpeg) + CLIP match + Demucs + MusicGen
"""

import cv2
import json
import shutil
import subprocess
import sys
import threading
import numpy as np
from pathlib import Path
from PIL import Image

# ── PySceneDetect ─────────────────────────────────────────────────────────────
try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_OK = True
except Exception:
    SCENEDETECT_OK = False

# ── MoviePy ───────────────────────────────────────────────────────────────────
try:
    from moviepy.editor import (
        ImageClip, VideoFileClip, ColorClip,
        CompositeVideoClip, concatenate_videoclips,
    )
    MOVIEPY_OK = True
except ImportError:
    try:
        from moviepy import (
            ImageClip, VideoFileClip, ColorClip,
            CompositeVideoClip, concatenate_videoclips,
        )
        MOVIEPY_OK = True
    except ImportError:
        MOVIEPY_OK = False

# ── OpenCLIP (lazy load) ──────────────────────────────────────────────────────
_clip_model = None
_clip_prep  = None
_clip_lock  = threading.Lock()
CLIP_OK     = False

def _load_clip():
    global _clip_model, _clip_prep, CLIP_OK
    with _clip_lock:
        if _clip_model is not None:
            return True
        try:
            import open_clip, torch
            _clip_model, _, _clip_prep = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            _clip_model.eval()
            CLIP_OK = True
            return True
        except Exception as e:
            print(f"[CLIP load] {e}")
            return False

# ── Demucs ────────────────────────────────────────────────────────────────────
try:
    import demucs  # noqa: F401
    DEMUCS_OK = True
except Exception:
    DEMUCS_OK = False

# ── MusicGen (lazy load) ──────────────────────────────────────────────────────
_music_gen_pipe = None
_music_gen_lock = threading.Lock()
MUSICGEN_OK     = False

def _load_musicgen():
    global _music_gen_pipe, MUSICGEN_OK
    with _music_gen_lock:
        if _music_gen_pipe is not None:
            return True
        try:
            from transformers import pipeline
            _music_gen_pipe = pipeline(
                "text-to-audio",
                model="facebook/musicgen-large",
                device="cuda",
            )
            MUSICGEN_OK = True
            return True
        except Exception as e:
            print(f"[MusicGen load] {e}")
            return False

# ── FFmpeg helpers ────────────────────────────────────────────────────────────

def _ff(*args, timeout=300):
    cmd = ['ffmpeg', '-y'] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or 'ffmpeg error')[-600:])

def _ffprobe(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json',
         '-show_streams', '-show_format', str(path)],
        capture_output=True, text=True, timeout=30,
    )
    return json.loads(r.stdout) if r.returncode == 0 else {}

def get_video_info(path):
    info = _ffprobe(path)
    for s in info.get('streams', []):
        if s.get('codec_type') == 'video':
            n, d = (s.get('r_frame_rate', '30/1') + '/1').split('/')[:2]
            fps  = int(n) / max(int(d), 1)
            dur  = float(s.get('duration') or info.get('format', {}).get('duration', 0))
            return {'width': int(s.get('width', 1920)),
                    'height': int(s.get('height', 1080)),
                    'duration': dur, 'fps': fps}
    return {'width': 1920, 'height': 1080, 'duration': 0, 'fps': 30}

def _even(n):
    return n + n % 2

# ── Scene Detection ───────────────────────────────────────────────────────────

def detect_scenes(video_path, output_dir, threshold=27.0, max_scenes=30):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = []

    if SCENEDETECT_OK:
        try:
            vid = open_video(str(video_path))
            sm  = SceneManager()
            sm.add_detector(ContentDetector(threshold=threshold))
            sm.detect_scenes(vid)
            for i, (s, e) in enumerate(sm.get_scene_list()[:max_scenes]):
                scenes.append({
                    'idx':       i,
                    'start_sec': round(s.get_seconds(), 3),
                    'end_sec':   round(e.get_seconds(), 3),
                    'duration':  round(e.get_seconds() - s.get_seconds(), 3),
                })
        except Exception as ex:
            print(f"[scenedetect] {ex}")

    if not scenes:
        scenes = _detect_scenes_cv(video_path, threshold, max_scenes)

    for sc in scenes:
        out = output_dir / f"thumb_{sc['idx']:03d}.jpg"
        try:
            _ff('-ss', sc['start_sec'] + sc['duration'] * 0.3,
                '-i', video_path, '-vframes', '1',
                '-q:v', '3', '-vf', 'scale=320:-2', str(out), timeout=20)
        except Exception:
            pass
        sc['thumbnail'] = out.name

    return scenes


def _detect_scenes_cv(video_path, threshold=30.0, max_scenes=30):
    cap   = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, int(fps / 4))
    scenes, start, prev = [], 0, None

    for fn in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff = float(cv2.absdiff(gray, prev).mean())
            if diff > threshold and (fn - start) > fps * 0.5:
                scenes.append({
                    'idx':       len(scenes),
                    'start_sec': round(start / fps, 3),
                    'end_sec':   round(fn / fps, 3),
                    'duration':  round((fn - start) / fps, 3),
                })
                start = fn
                if len(scenes) >= max_scenes:
                    break
        prev = gray
    cap.release()

    scenes.append({
        'idx':       len(scenes),
        'start_sec': round(start / fps, 3),
        'end_sec':   round(total / fps, 3),
        'duration':  round((total - start) / fps, 3),
    })
    return scenes

# ── CLIP Embedding ────────────────────────────────────────────────────────────

def embed_image(path):
    if not _load_clip():
        return None
    try:
        import torch
        img = _clip_prep(Image.open(str(path)).convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            f = _clip_model.encode_image(img)[0].numpy().astype(np.float32)
        return f / (np.linalg.norm(f) + 1e-8)
    except Exception as e:
        print(f"[embed_image] {e}")
        return None


def embed_video_thumb(video_path, ts=None):
    import tempfile
    tmp = Path(tempfile.mktemp(suffix='.jpg'))
    try:
        _ff('-ss', ts or 0, '-i', video_path,
            '-vframes', '1', '-q:v', '3', str(tmp), timeout=15)
        return embed_image(tmp)
    except Exception:
        return None
    finally:
        tmp.unlink(missing_ok=True)


def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def match_media_to_scenes(scene_embeddings, media_list):
    """
    CLIP متاح → greedy best-match بالـ embeddings
    CLIP مش متاح / كل embeddings None → round-robin مضمون، لا أسود أبداً
    """
    n_scenes = len(scene_embeddings)
    n_media  = len(media_list)

    if n_media == 0:
        return [None] * n_scenes

    def _round_robin():
        return [i % n_media for i in range(n_scenes)]

    if not _load_clip():
        return _round_robin()

    if all(e is None for e in scene_embeddings):
        return _round_robin()

    used = set()
    out  = []
    for idx, s_emb in enumerate(scene_embeddings):
        best_i, best_s = None, -1.0

        if s_emb is not None:
            for mi, m in enumerate(media_list):
                if mi in used:
                    continue
                score = cosine_sim(s_emb, m.get('embedding'))
                if score > best_s:
                    best_s, best_i = score, mi

        if best_i is None:
            for offset in range(n_media):
                candidate = (idx + offset) % n_media
                if candidate not in used:
                    best_i = candidate
                    break
            else:
                best_i = idx % n_media

        out.append(best_i)
        used.add(best_i)
    return out

# ── Audio ─────────────────────────────────────────────────────────────────────

def extract_audio(video_path, out_wav):
    _ff('-i', video_path, '-vn', '-acodec', 'pcm_s16le',
        '-ar', '44100', '-ac', '2', str(out_wav))
    return str(out_wav)


def separate_audio(wav_path, work_dir, stems='vocals'):
    if not DEMUCS_OK:
        raise RuntimeError('Demucs غير متاح')
    work_dir = Path(work_dir)
    norm = work_dir / 'demucs_in.wav'
    _ff('-i', wav_path, '-ar', '44100', '-ac', '2', str(norm), timeout=60)
    r = subprocess.run(
        [sys.executable, '-m', 'demucs', '--two-stems', stems,
         str(norm), '-o', str(work_dir)],
        capture_output=True, text=True, timeout=1200,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:])
    stem_dir = work_dir / 'htdemucs' / 'demucs_in'
    return {p.stem: str(p) for p in stem_dir.glob('*.wav')}


def analyze_video_for_music(scenes, media_list):
    n_scenes = len(scenes)
    avg_dur  = sum(s.get('duration', 2.0) for s in scenes) / max(n_scenes, 1)
    n_videos = sum(1 for m in media_list if m.get('type') == 'video')

    if avg_dur < 1.5:
        energy = "fast-paced, high energy, upbeat"
    elif avg_dur < 3.5:
        energy = "energetic, dynamic, upbeat"
    else:
        energy = "cinematic, smooth, atmospheric"

    base = "arabic" if n_videos else "instrumental"
    return f"{energy} {base} background music, professional production, no lyrics"


def generate_music(prompt, duration_seconds, out_wav):
    if not _load_musicgen():
        raise RuntimeError("MusicGen غير متاح — تحقق من تثبيت transformers وـ CUDA")

    import scipy.io.wavfile as wav_io

    max_tokens = max(50, int(duration_seconds * 51.2))
    result = _music_gen_pipe(
        prompt,
        forward_params={
            "do_sample":      True,
            "guidance_scale": 3.0,
            "max_new_tokens": max_tokens,
        },
    )

    audio = result[0]["audio"]
    sr    = result[0]["sampling_rate"]
    if audio.ndim == 3:
        audio = audio[0]

    audio_int16 = (audio.T * 32767).clip(-32768, 32767).astype(np.int16)
    wav_io.write(out_wav, sr, audio_int16)
    return out_wav


def _add_audio(raw_path, audio_path, out_path):
    """يدمج فيديو مع ملف صوت بـ FFmpeg."""
    _ff('-i', raw_path, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
        '-map', '0:v:0', '-map', '1:a:0', '-shortest', str(out_path))

# ── MoviePy 1.x / 2.x compatibility ─────────────────────────────────────────

def _mv_duration(clip, dur):
    return clip.with_duration(dur) if hasattr(clip, 'with_duration') else clip.set_duration(dur)

def _mv_position(clip, pos):
    return clip.with_position(pos) if hasattr(clip, 'with_position') else clip.set_position(pos)

def _mv_resize(clip, size):
    return clip.resized(size) if hasattr(clip, 'resized') else clip.resize(size)

def _mv_colorclip(w, h, dur):
    bg = ColorClip((w, h), color=(0, 0, 0))
    return _mv_duration(bg, dur)

# ── MoviePy Assembly ──────────────────────────────────────────────────────────

def _fit_clip_mv(clip, w, h, dur):
    """Scale to fit w×h with black letterbox — MoviePy 2.x API."""
    scale   = min(w / clip.w, h / clip.h)
    nw      = max(2, int(clip.w * scale) & ~1)
    nh      = max(2, int(clip.h * scale) & ~1)
    xp      = (w - nw) // 2
    yp      = (h - nh) // 2
    resized = _mv_resize(clip, (nw, nh))
    resized = _mv_duration(resized, dur)
    bg      = _mv_colorclip(w, h, dur)
    return CompositeVideoClip([bg, _mv_position(resized, (xp, yp))]).with_duration(dur) \
           if hasattr(bg, 'with_duration') else \
           CompositeVideoClip([bg, _mv_position(resized, (xp, yp))]).set_duration(dur)


def _build_clip_mv(m, dur, w, h):
    """يبني MoviePy clip من عنصر ميديا واحد — MoviePy 2.x API."""
    path = m['path']
    if m['type'] == 'image':
        base = ImageClip(str(path))
        base = _mv_duration(base, dur)
    else:
        vc   = VideoFileClip(str(path))
        trim = min(dur, vc.duration) if vc.duration > 0 else dur
        # subclipped (2.x) أو subclip (1.x)
        base = vc.subclipped(0, trim) if hasattr(vc, 'subclipped') else vc.subclip(0, trim)
    return _fit_clip_mv(base, w, h, dur)


def _apply_transition(clips, i, trans, t=0.45):
    """يطبّق انتقال قبل الكليب i بتعديل i-1 و i مباشرةً."""
    if not trans or trans == 'cut' or i == 0:
        return
    try:
        from moviepy import vfx as _vfx
        if trans == 'fade_black':
            clips[i - 1] = clips[i - 1].with_effects([_vfx.FadeOut(t)])
            clips[i]     = clips[i].with_effects([_vfx.FadeIn(t)])
        elif trans == 'crossdissolve':
            clips[i - 1] = clips[i - 1].with_effects([_vfx.CrossFadeOut(t)])
            clips[i]     = clips[i].with_effects([_vfx.CrossFadeIn(t)])
        elif trans == 'wipe':
            clips[i] = clips[i].with_effects([_vfx.SlideIn(t, 'right')])
    except Exception as e:
        print(f"[transition {i}] {e}")


def _assemble_moviepy(scenes, assignments, media_list, w, h, fps, raw_path, cb, total,
                      transitions=None, intro_path=None, outro_path=None):
    """بناء الفيديو الخام بـ MoviePy مع دعم الانتقالات والـ intro/outro."""
    clips = []

    # ── Intro ──
    if intro_path and Path(intro_path).exists():
        try:
            vc = VideoFileClip(str(intro_path))
            clips.append(_fit_clip_mv(vc, w, h, vc.duration))
        except Exception as e:
            print(f"[intro] {e}")

    # ── المقاطع الأساسية ──
    for i, (scene, mi) in enumerate(zip(scenes, assignments)):
        dur = max(scene.get('duration', 0.1), 0.1)
        if mi is None or mi >= len(media_list):
            mi = i % len(media_list) if media_list else None

        clip = None
        if mi is not None:
            try:
                clip = _build_clip_mv(media_list[mi], dur, w, h)
            except Exception as e:
                print(f"[clip_mv {i}] {e}")

        clips.append(clip if clip is not None else _mv_colorclip(w, h, dur))
        if cb:
            cb(int((i + 1) / total * 65))

    # ── Outro ──
    if outro_path and Path(outro_path).exists():
        try:
            vc = VideoFileClip(str(outro_path))
            clips.append(_fit_clip_mv(vc, w, h, vc.duration))
        except Exception as e:
            print(f"[outro] {e}")

    # ── تطبيق الانتقالات (بعد بناء كل الكليبات) ──
    # offset = 1 لأن intro (لو موجود) هو الكليب الأول
    intro_offset = 1 if (intro_path and Path(intro_path).exists()) else 0
    if transitions:
        for i, trans in enumerate(transitions):
            clip_idx = intro_offset + i
            if clip_idx < len(clips):
                _apply_transition(clips, clip_idx, trans)

    padding = -0.45 if (transitions and any(t == 'crossdissolve' for t in transitions)) else 0
    final = concatenate_videoclips(clips, method='compose', padding=padding)
    final.write_videofile(
        raw_path, codec='libx264', audio=False,
        fps=fps or 30, preset='fast', threads=4, logger=None,
    )
    final.close()
    for c in clips:
        try: c.close()
        except Exception: pass

# ── FFmpeg Fallback Assembly ──────────────────────────────────────────────────

_SCALE_PAD = ('scale={w}:{h}:force_original_aspect_ratio=decrease,'
              'pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1')


def _build_effect_vf(effect, w, h, dur):
    """بناء ffmpeg vf string للتأثيرات البصرية."""
    base  = _SCALE_PAD.format(w=w, h=h)
    ftype = effect.get('type', 'none') if effect else 'none'
    inten = effect.get('intensity', 50) if effect else 50

    if ftype == 'bw':
        return base + ',hue=s=0'
    elif ftype == 'bright':
        val = 0 + (inten / 100) * 0.6  # 0.0 → 0.6 extra brightness
        return base + f',eq=brightness={val:.2f}'
    elif ftype == 'vignette':
        return base + ',vignette=PI/4'
    elif ftype == 'vintage':
        return base + ',curves=r=\'0/0.1 1/0.9\':g=\'0/0.05 1/0.85\':b=\'0/0 1/0.75\',vignette=PI/5'
    elif ftype == 'mirror':
        return base + ',hflip'
    elif ftype == 'zoomin':
        # Ken Burns — تكبير تدريجي
        zoom_end = 1.0 + (inten / 100) * 0.15
        return base + f',zoompan=z=\'min(zoom+0.0005,{zoom_end:.3f})\':d={int(dur * 30)}:x=\'iw/2-(iw/zoom/2)\':y=\'ih/2-(ih/zoom/2)\':s={w}x{h}'
    elif ftype == 'shake':
        amp = max(2, int(inten / 10))
        return base + f',crop=in_w-{amp*2}:in_h-{amp*2}:{amp}:{amp},scale={w}:{h},' + \
               f'crop=trunc(iw/2)*2:trunc(ih/2)*2'
    else:
        return base


def _image_to_clip_ff(image_path, duration, w, h, out, effect=None):
    w, h = _even(w), _even(h)
    vf   = _build_effect_vf(effect, w, h, duration)
    _ff('-loop', '1', '-i', image_path,
        '-t', duration,
        '-vf', vf,
        '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-an', out)


def _video_to_clip_ff(video_path, start, duration, w, h, out, effect=None):
    w, h = _even(w), _even(h)
    vf   = _build_effect_vf(effect, w, h, duration)
    speed = 0.5 if (effect and effect.get('type') == 'slowmo') else 1.0
    pts_filter = f',setpts={1/speed:.1f}*PTS' if speed != 1.0 else ''
    _ff('-ss', start, '-i', video_path,
        '-t', duration * speed,
        '-vf', vf + pts_filter,
        '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-an', out)


def _make_black_clip(w, h, dur, out):
    w, h = _even(w), _even(h)
    _ff('-f', 'lavfi',
        '-i', f'color=c=black:s={w}x{h}:r=30',
        '-t', dur,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', out)


def _concat_clips_ff(paths, out):
    lst = Path(out).parent / '_list.txt'
    lst.write_text(
        '\n'.join(f"file '{Path(p).resolve()}'" for p in paths),
        encoding='utf-8',
    )
    _ff('-f', 'concat', '-safe', '0', '-i', str(lst),
        '-c', 'copy', out, timeout=300)
    lst.unlink(missing_ok=True)


def _assemble_ffmpeg(scenes, assignments, media_list, w, h, raw_path, work_dir, cb, total):
    """بناء الفيديو الخام بـ FFmpeg (fallback) مع دعم التأثيرات."""
    clips = []
    for i, (scene, mi) in enumerate(zip(scenes, assignments)):
        clip_out = Path(work_dir) / f'c_{i:03d}.mp4'
        dur      = max(scene.get('duration', 0.1), 0.1)
        effect   = scene.get('effect', None)

        if mi is None or mi >= len(media_list):
            mi = i % len(media_list) if media_list else None

        try:
            if mi is None:
                _make_black_clip(w, h, dur, str(clip_out))
            else:
                m = media_list[mi]
                if m['type'] == 'image':
                    _image_to_clip_ff(m['path'], dur, w, h, str(clip_out), effect)
                else:
                    _video_to_clip_ff(m['path'], 0, dur, w, h, str(clip_out), effect)
        except Exception as e:
            print(f"[clip_ff {i}] {e}")
            _make_black_clip(w, h, dur, str(clip_out))

        clips.append(str(clip_out))
        if cb:
            cb(int((i + 1) / total * 70))

    _concat_clips_ff(clips, raw_path)

# ── Template Assembly (entry point) ──────────────────────────────────────────

def assemble_template(scenes, assignments, media_list, audio_track,
                      video_info, work_dir, out_path, ref_video_path=None, cb=None,
                      transitions=None, intro_path=None, outro_path=None):
    """
    assignments  : list of media indices (one per scene)
    audio_track  : 'full' | 'no_vocals' | 'vocals' | 'generate' | 'none'
    transitions  : list of str per scene ('cut'|'fade_black'|'crossdissolve'|'wipe')
    intro_path   : مسار فيديو الـ intro (اختياري)
    outro_path   : مسار فيديو الـ outro (اختياري)
    """
    work_dir = Path(work_dir)
    w        = _even(video_info['width'])
    h        = _even(video_info['height'])
    fps      = video_info.get('fps', 30) or 30
    total    = len(scenes)
    raw_path = str(work_dir / 'raw.mp4')

    # ── بناء الفيديو الخام ──
    if MOVIEPY_OK:
        _assemble_moviepy(scenes, assignments, media_list, w, h, fps, raw_path, cb, total,
                          transitions=transitions, intro_path=intro_path, outro_path=outro_path)
    else:
        _assemble_ffmpeg(scenes, assignments, media_list, w, h, raw_path, str(work_dir), cb, total)

    if cb:
        cb(80)

    # ── الصوت ──
    if audio_track == 'none':
        shutil.copy(raw_path, str(out_path))

    elif audio_track == 'generate':
        try:
            prompt  = analyze_video_for_music(scenes, media_list)
            raw_dur = get_video_info(raw_path).get('duration', 30.0)
            gen_wav = str(work_dir / 'gen_music.wav')
            print(f"[musicgen] prompt: {prompt} | duration: {raw_dur:.1f}s")
            generate_music(prompt, raw_dur, gen_wav)
            _add_audio(raw_path, gen_wav, str(out_path))
        except Exception as e:
            print(f"[musicgen] {e} — saving without audio")
            shutil.copy(raw_path, str(out_path))

    else:
        ref_vid = Path(ref_video_path) if ref_video_path else (work_dir / 'ref.mp4')
        if not ref_vid.exists():
            shutil.copy(raw_path, str(out_path))
        else:
            raw_wav  = str(work_dir / 'audio_full.wav')
            aud_path = raw_wav
            try:
                extract_audio(str(ref_vid), raw_wav)
                if audio_track in ('no_vocals', 'vocals') and DEMUCS_OK:
                    stems    = separate_audio(raw_wav, str(work_dir))
                    aud_path = stems.get(audio_track, raw_wav)
                _add_audio(raw_path, aud_path, str(out_path))
            except Exception as e:
                print(f"[audio] {e} — saving without audio")
                shutil.copy(raw_path, str(out_path))

    if cb:
        cb(100)
    return str(out_path)

# ── AI Edit Mode ──────────────────────────────────────────────────────────────

def assemble_ai_edit(ref_video_path, media_list, work_dir, out_path,
                     audio_track='full', cb=None, use_understanding=True):
    """
    يحلل الفيديو المرجعي، يختار الميديا المناسبة، يجمع، يضيف الصوت.
    audio_track    : 'full' | 'no_vocals' | 'vocals' | 'generate' | 'none'
    use_understanding : استخدم Qwen2-VL لفهم أعمق (يحتاج vram)
    """
    work_dir = Path(work_dir)
    ref_path = Path(ref_video_path)
    info     = get_video_info(str(ref_path))

    if cb:
        cb(5)

    # ── محاولة استخدام video_understanding (Qwen2-VL) ──
    if use_understanding:
        try:
            from video_understanding import (
                analyze_reference_video,
                analyze_media_item,
                match_media_to_analysis,
            )

            if cb: cb(10)
            video_analysis = analyze_reference_video(str(ref_path))
            if cb: cb(30)

            for m in media_list:
                if 'analysis' not in m:
                    m['analysis'] = analyze_media_item(m['path'], video_analysis)
            if cb: cb(45)

            assignments = match_media_to_analysis(video_analysis, media_list)
            scenes = [
                {
                    'idx':       s['index'],
                    'start_sec': s['start_sec'],
                    'end_sec':   s['end_sec'],
                    'duration':  s['end_sec'] - s['start_sec'],
                    'thumbnail': '',
                }
                for s in video_analysis.get('segments', [])
            ]

            if scenes:
                if cb: cb(50)
                return assemble_template(
                    scenes, assignments, media_list,
                    audio_track, info,
                    str(work_dir), str(out_path),
                    ref_video_path=str(ref_path),
                    cb=lambda p: cb(50 + int(p * 0.5)) if cb else None,
                )
        except Exception as e:
            print(f"[assemble_ai_edit] video_understanding failed: {e} — fallback to CLIP")

    # ── Fallback: detect_scenes + CLIP ──
    if cb: cb(20)
    scenes = detect_scenes(str(ref_path), str(work_dir / 'ref_thumbs'))

    if cb: cb(35)
    scene_embs = []
    for sc in scenes:
        th = work_dir / 'ref_thumbs' / sc.get('thumbnail', '')
        scene_embs.append(embed_image(str(th)) if th.exists() else None)

    if cb: cb(45)
    assignments = match_media_to_scenes(scene_embs, media_list)

    if cb: cb(50)
    return assemble_template(
        scenes, assignments, media_list,
        audio_track, info,
        str(work_dir), str(out_path),
        ref_video_path=str(ref_path),
        cb=lambda p: cb(50 + int(p * 0.5)) if cb else None,
    )