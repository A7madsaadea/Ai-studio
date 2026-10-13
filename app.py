import os
import shutil
import zipfile
import json
import base64
import threading
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    DRIVE_AVAILABLE = True
except ImportError:
    DRIVE_AVAILABLE = False

from filter_module import ImageFilter
from enhance_module import ImageEnhancer
from watermark_module import Watermarker
from image_processor import ImageProcessor
from video_maker import (
    detect_scenes, embed_image, embed_video_thumb,
    match_media_to_scenes, assemble_template, assemble_ai_edit,
    get_video_info, CLIP_OK, DEMUCS_OK,
)
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

BASE_DIR     = Path(__file__).parent
LOGO_PATH    = str(BASE_DIR / "logo.png")
SESSIONS_DIR = BASE_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {'.jpg', '.jpeg', '.png'}
MAX_WORKERS  = 4

JOBS = {}
JOBS_LOCK = threading.Lock()

PROC_JOBS = {}
PROC_JOBS_LOCK = threading.Lock()

PROC_ALLOWED_IMG = {'.jpg', '.jpeg', '.png'}
PROC_ALLOWED_VID = {'.mp4', '.mov'}

def proc_allowed(filename):
    return Path(filename).suffix.lower() in PROC_ALLOWED_IMG | PROC_ALLOWED_VID

def is_video(filename):
    return Path(filename).suffix.lower() in PROC_ALLOWED_VID

def get_proc_session_dir(session_id):
    d = SESSIONS_DIR / ('proc_' + session_id)
    for sub in ['INPUT', 'REELS', 'PHOTOS']:
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d

def run_proc_job(job_id, session_id, saved, options):
    session_dir = get_proc_session_dir(session_id)
    results = {'reels': [], 'photos': [], 'videos_reels': [], 'videos_photos': []}
    lock = threading.Lock()

    def process_one_file(name):
        src = str(session_dir / 'INPUT' / name)
        try:
            if is_video(name):
                cat = ImageProcessor.classify_video(src)
                shutil.copy(src, str(session_dir / cat / name))
                with lock:
                    results['videos_reels' if cat == 'REELS' else 'videos_photos'].append(name)
            else:
                img, cat = ImageProcessor.process(
                    src,
                    auto_lighting=options.get('auto_lighting', True),
                    auto_wb=options.get('auto_wb', True),
                    flip_h=options.get('flip_h', False),
                    flip_v=options.get('flip_v', False),
                    face_crop=options.get('face_crop', False),
                    denoise=options.get('denoise', False),
                    straighten=options.get('straighten', False),
                )
                if options.get('watermark', False) and Path(LOGO_PATH).exists():
                    try:
                        _wm = Watermarker(LOGO_PATH, scale=0.15, opacity=200)
                        img = _wm.apply(img)
                    except Exception as we:
                        print(f"Watermark error {name}: {we}")
                out = str(session_dir / cat / name)
                img.convert('RGB').save(out, quality=95)
                with lock:
                    results['reels' if cat == 'REELS' else 'photos'].append(name)
        except Exception as e:
            print(f"Processor error {name}: {e}")
            with lock:
                results['photos'].append(name)
        with PROC_JOBS_LOCK:
            PROC_JOBS[job_id]['progress'] += 1

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one_file, name): name for name in saved}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Job error: {e}")
        with PROC_JOBS_LOCK:
            PROC_JOBS[job_id].update({
                'status': 'done',
                'results': results,
                'stats': {
                    'total': len(saved),
                    'reels': len(results['reels']),
                    'photos': len(results['photos']),
                    'videos_reels': len(results['videos_reels']),
                    'videos_photos': len(results['videos_photos']),
                },
            })
    except Exception as e:
        with PROC_JOBS_LOCK:
            PROC_JOBS[job_id]['status'] = 'error'
            PROC_JOBS[job_id]['error'] = str(e)

def allowed(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT

def get_session_dir(session_id):
    d = SESSIONS_DIR / session_id
    for sub in ['INPUT', 'FINAL', 'REJECTED', 'DUPLICATES']:
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d

def process_one(name, session_dir, img_filter, results, lock, job_id):
    src = str(session_dir / 'INPUT' / name)
    _enhancer    = ImageEnhancer()
    _watermarker = Watermarker(LOGO_PATH)

    if img_filter.is_blurry(src):
        shutil.copy(src, str(session_dir / 'REJECTED' / name))
        with lock:
            results['rejected'].append(name)
    else:
        enhanced = _enhancer.process(src)
        if enhanced:
            final = _watermarker.apply(enhanced)
            out_path = str(session_dir / 'FINAL' / name)
            final.save(out_path, quality=95)
            with lock:
                results['final'].append(name)
        else:
            shutil.copy(src, str(session_dir / 'REJECTED' / name))
            with lock:
                results['rejected'].append(name)

    with JOBS_LOCK:
        JOBS[job_id]['progress'] += 1

def run_job(job_id, session_id, saved):
    session_dir = get_session_dir(session_id)
    img_filter  = ImageFilter(blur_threshold=100, hash_cutoff=5)
    results     = {'final': [], 'rejected': [], 'duplicates': []}
    lock        = threading.Lock()

    try:
        non_dupes = []
        for name in saved:
            src = str(session_dir / 'INPUT' / name)
            if img_filter.is_duplicate(src):
                shutil.copy(src, str(session_dir / 'DUPLICATES' / name))
                results['duplicates'].append(name)
                with JOBS_LOCK:
                    JOBS[job_id]['progress'] += 1
            else:
                non_dupes.append(name)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_one, name, session_dir,
                                img_filter, results, lock, job_id): name
                for name in non_dupes
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    name = futures[future]
                    print(f"Error {name}: {e}")
                    with lock:
                        results['rejected'].append(name)
                    with JOBS_LOCK:
                        JOBS[job_id]['progress'] += 1

        with JOBS_LOCK:
            JOBS[job_id].update({
                'status':  'done',
                'results': results,
                'stats': {
                    'total':      len(saved),
                    'final':      len(results['final']),
                    'rejected':   len(results['rejected']),
                    'duplicates': len(results['duplicates']),
                }
            })

    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error']  = str(e)


# ── ROUTES ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/script')
def script_page():
    return render_template('script.html')

@app.route('/brain')
def brain_page():
    return render_template('brain.html')

@app.route('/ideas')
def ideas_page():
    return render_template('ideas.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    session_id  = uuid.uuid4().hex
    job_id      = uuid.uuid4().hex
    session_dir = get_session_dir(session_id)

    saved = []
    for f in files:
        if f and allowed(f.filename):
            name = secure_filename(f.filename)
            dest = session_dir / 'INPUT' / name
            f.save(str(dest))
            saved.append(name)

    if not saved:
        return jsonify({'error': 'No valid image files'}), 400

    with JOBS_LOCK:
        JOBS[job_id] = {
            'status':     'processing',
            'session_id': session_id,
            'total':      len(saved),
            'progress':   0,
            'results':    None,
            'stats':      None,
        }

    t = threading.Thread(target=run_job, args=(job_id, session_id, saved), daemon=True)
    t.start()
    return jsonify({'job_id': job_id, 'total': len(saved)})

@app.route('/status/<job_id>')
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)

@app.route('/preview/<session_id>/<category>/<filename>')
def preview(session_id, category, filename):
    category = category.upper()
    if category not in ('FINAL', 'REJECTED', 'DUPLICATES'):
        return 'Invalid category', 400
    path = SESSIONS_DIR / session_id / category / secure_filename(filename)
    if not path.exists():
        return 'Not found', 404
    return send_file(str(path))

@app.route('/download/<session_id>/<filename>')
def download_one(session_id, filename):
    path = SESSIONS_DIR / session_id / 'FINAL' / secure_filename(filename)
    if not path.exists():
        return 'Not found', 404
    return send_file(str(path), as_attachment=True)

@app.route('/download-all/<session_id>')
def download_all(session_id):
    final_dir = SESSIONS_DIR / session_id / 'FINAL'
    if not final_dir.exists():
        return 'Session not found', 404
    zip_path = SESSIONS_DIR / session_id / 'final_images.zip'
    with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
        for img in final_dir.iterdir():
            zf.write(str(img), img.name)
    return send_file(str(zip_path), as_attachment=True,
                     download_name='TechArabi_Processed.zip')

@app.route('/upload-drive/<session_id>', methods=['POST'])
def upload_drive(session_id):
    if not DRIVE_AVAILABLE:
        return jsonify({'error': 'Google Drive library not installed'}), 500
    creds_b64 = os.environ.get('GOOGLE_CREDENTIALS_B64')
    if not creds_b64:
        return jsonify({'error': 'Google Drive credentials not configured'}), 500
    try:
        creds_json = json.loads(base64.b64decode(creds_b64))
        creds = service_account.Credentials.from_service_account_info(
            creds_json, scopes=['https://www.googleapis.com/auth/drive.file'])
        service   = build('drive', 'v3', credentials=creds)
        from datetime import date
        folder_name = f"TechArabi-Processed-{date.today()}"
        folder    = service.files().create(
            body={'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'},
            fields='id').execute()
        folder_id = folder['id']
        final_dir = SESSIONS_DIR / session_id / 'FINAL'
        uploaded  = []
        for img_path in final_dir.iterdir():
            media = MediaFileUpload(str(img_path), mimetype='image/jpeg')
            service.files().create(
                body={'name': img_path.name, 'parents': [folder_id]},
                media_body=media, fields='id').execute()
            uploaded.append(img_path.name)
        service.permissions().create(
            fileId=folder_id, body={'type': 'anyone', 'role': 'reader'}).execute()
        return jsonify({
            'url': f"https://drive.google.com/drive/folders/{folder_id}",
            'uploaded': len(uploaded)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/approve', methods=['POST'])
def approve():
    data       = request.get_json() or {}
    session_id = data.get('session_id', '').strip()
    files      = data.get('files', [])
    event_name = data.get('event_name', '').strip()

    if not session_id or not files or not event_name:
        return jsonify({'error': 'بيانات ناقصة'}), 400

    safe_event = secure_filename(event_name)
    event_dir  = BASE_DIR / 'events' / safe_event
    event_dir.mkdir(parents=True, exist_ok=True)

    session_final = SESSIONS_DIR / session_id / 'FINAL'
    saved = 0
    for fname in files:
        safe_fname = secure_filename(fname)
        src = session_final / safe_fname
        if src.exists():
            shutil.copy(str(src), str(event_dir / safe_fname))
            saved += 1

    return jsonify({'ok': True, 'saved': saved, 'event': safe_event})



@app.route('/processor')
def processor_page():
    return render_template('processor.html')

@app.route('/processor/upload', methods=['POST'])
def processor_upload():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'لا يوجد ملفات'}), 400

    options = {}
    for key in ['auto_lighting', 'auto_wb', 'flip_h', 'flip_v', 'face_crop', 'denoise', 'straighten', 'watermark']:
        options[key] = request.form.get(key, 'false').lower() == 'true'

    session_id = uuid.uuid4().hex
    job_id = uuid.uuid4().hex
    session_dir = get_proc_session_dir(session_id)

    saved = []
    for f in files:
        if f and proc_allowed(f.filename):
            name = secure_filename(f.filename)
            base = session_dir / 'INPUT' / name
            if base.exists():
                stem, suffix = Path(name).stem, Path(name).suffix
                i = 1
                while (session_dir / 'INPUT' / f'{stem}_{i}{suffix}').exists():
                    i += 1
                name = f'{stem}_{i}{suffix}'
            f.save(str(session_dir / 'INPUT' / name))
            saved.append(name)

    if not saved:
        return jsonify({'error': 'لا يوجد ملفات صالحة (JPG/PNG/MP4/MOV)'}), 400

    with PROC_JOBS_LOCK:
        PROC_JOBS[job_id] = {
            'status': 'processing',
            'session_id': session_id,
            'total': len(saved),
            'progress': 0,
            'results': None,
            'stats': None,
        }

    t = threading.Thread(target=run_proc_job, args=(job_id, session_id, saved, options), daemon=True)
    t.start()
    return jsonify({'job_id': job_id, 'total': len(saved)})

@app.route('/processor/status/<job_id>')
def processor_status(job_id):
    with PROC_JOBS_LOCK:
        job = PROC_JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)

@app.route('/processor/original/<session_id>/<filename>')
def processor_original(session_id, filename):
    path = SESSIONS_DIR / ('proc_' + session_id) / 'INPUT' / secure_filename(filename)
    if not path.exists():
        return 'Not found', 404
    return send_file(str(path))

@app.route('/processor/approve', methods=['POST'])
def processor_approve():
    data = request.get_json() or {}
    session_id = data.get('session_id', '').strip()
    approved   = data.get('approved', [])   # [{name, category}, ...]
    event_name = data.get('event_name', '').strip()

    if not session_id or not approved or not event_name:
        return jsonify({'error': 'بيانات ناقصة'}), 400

    safe_event = secure_filename(event_name)
    proc_dir = SESSIONS_DIR / ('proc_' + session_id)
    saved = 0
    for item in approved:
        name = secure_filename(item.get('name', ''))
        cat  = item.get('category', 'PHOTOS').upper()
        if cat not in ('REELS', 'PHOTOS'):
            cat = 'PHOTOS'
        src = proc_dir / cat / name
        if src.exists():
            dest = BASE_DIR / 'events' / safe_event / cat
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src), str(dest / name))
            saved += 1

    return jsonify({'ok': True, 'saved': saved, 'event': safe_event})

@app.route('/processor/preview/<session_id>/<category>/<filename>')
def processor_preview(session_id, category, filename):
    category = category.upper()
    if category not in ('REELS', 'PHOTOS'):
        return 'Invalid category', 400
    path = SESSIONS_DIR / ('proc_' + session_id) / category / secure_filename(filename)
    if not path.exists():
        return 'Not found', 404
    return send_file(str(path))

@app.route('/processor/download/<session_id>')
def processor_download(session_id):
    proc_dir = SESSIONS_DIR / ('proc_' + session_id)
    if not proc_dir.exists():
        return 'Session not found', 404
    zip_path = proc_dir / 'sorted.zip'
    with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
        for cat in ['REELS', 'PHOTOS']:
            cat_dir = proc_dir / cat
            if cat_dir.exists():
                for f in cat_dir.iterdir():
                    zf.write(str(f), f'{cat}/{f.name}')
    return send_file(str(zip_path), as_attachment=True, download_name='TechArabi_Sorted.zip')

@app.route('/analyze')
def analyze_page():
    return render_template('analyze.html')

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    try:
        import requests as req
        import time
    except ImportError:
        return jsonify({'error': 'requests غير مثبت'}), 500

    f = request.files.get('video')
    if not f:
        return jsonify({'error': 'ما في فيديو'}), 400

    assemblyai_key = os.environ.get('ASSEMBLYAI_KEY')
    if not assemblyai_key:
        return jsonify({'error': 'ASSEMBLYAI_KEY غير مضبوط'}), 503
    headers = {'authorization': assemblyai_key}

    tmp_dir = BASE_DIR / 'sessions' / ('analyze_' + uuid.uuid4().hex)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_dir / secure_filename(f.filename)
    f.save(str(video_path))

    try:
        # 1. رفع الفيديو لـ AssemblyAI
        with open(str(video_path), 'rb') as vf:
            upload_res = req.post(
                'https://api.assemblyai.com/v2/upload',
                headers=headers,
                data=vf
            )
        upload_url = upload_res.json().get('upload_url')
        if not upload_url:
            return jsonify({'error': 'فشل رفع الفيديو'}), 500

        # 2. إرسال طلب التحويل
        transcript_res = req.post(
            'https://api.assemblyai.com/v2/transcript',
            headers={**headers, 'content-type': 'application/json'},
            json={
                'audio_url': upload_url,
                'speech_models': ['universal-3-pro', 'universal-2'],
                'language_code': 'ar'
            }
        )
        transcript_id = transcript_res.json().get('id')
        if not transcript_id:
            return jsonify({'error': 'فشل إنشاء الطلب'}), 500

        # 3. انتظار النتيجة
        for _ in range(60):
            poll = req.get(
                f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                headers=headers
            ).json()
            if poll['status'] == 'completed':
                return jsonify({'transcript': poll['text']})
            if poll['status'] == 'error':
                return jsonify({'error': poll.get('error', 'خطأ غير معروف')}), 500
            time.sleep(3)

        return jsonify({'error': 'انتهت مهلة الانتظار'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        import shutil as _shutil
        _shutil.rmtree(str(tmp_dir), ignore_errors=True)

# ══════════════════════════════════════════════════════════════════════════════
# المرحلة الثانية — أداة صنع الفيديو
# ══════════════════════════════════════════════════════════════════════════════

VIDEO_JOBS      = {}
VIDEO_JOBS_LOCK = threading.Lock()
VIDEO_DIR       = SESSIONS_DIR / 'video'
VIDEO_DIR.mkdir(exist_ok=True)

VIDEO_IMG = {'.jpg', '.jpeg', '.png'}
VIDEO_VID = {'.mp4', '.mov', '.avi', '.mkv'}
VIDEO_ALL = VIDEO_IMG | VIDEO_VID | {'.webm'}

def _vext(name):  return Path(name).suffix.lower()
def _is_img(name): return _vext(name) in VIDEO_IMG
def _is_vid(name): return _vext(name) in VIDEO_VID

def _vsession(sid):
    d = VIDEO_DIR / sid
    for sub in ('INPUT', 'MEDIA', 'THUMBS', 'MEDIA_THUMBS', 'OUTPUT', 'WORK'):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d

def _vjob_set(jid, **kw):
    with VIDEO_JOBS_LOCK:
        VIDEO_JOBS.setdefault(jid, {}).update(kw)

def _vjob_progress(jid, pct):
    with VIDEO_JOBS_LOCK:
        VIDEO_JOBS[jid]['progress'] = pct


# ── GET /video ────────────────────────────────────────────────────────────────
@app.route('/video')
def video_page():
    return render_template('video.html',
                           clip_ok=CLIP_OK, demucs_ok=DEMUCS_OK)


# ── POST /video/analyze ───────────────────────────────────────────────────────
@app.route('/video/analyze', methods=['POST'])
def video_analyze():
    f = request.files.get('reference')
    if not f or _vext(f.filename) not in VIDEO_VID:
        return jsonify({'error': 'أرفع فيديو مرجعي (mp4/mov)'}), 400

    sid = uuid.uuid4().hex
    jid = uuid.uuid4().hex
    sd  = _vsession(sid)
    ref = sd / 'INPUT' / secure_filename(f.filename)
    f.save(str(ref))
    shutil.copy(str(ref), str(sd / 'WORK' / 'ref.mp4'))

    _vjob_set(jid, status='processing', type='analyze',
              session_id=sid, progress=0)

    def _run():
        try:
            _vjob_progress(jid, 10)
            scenes = detect_scenes(str(ref), str(sd / 'THUMBS'))
            info   = get_video_info(str(ref))
            _vjob_set(jid, status='done', progress=100,
                      result={'scenes': scenes,
                              'video_info': info,
                              'session_id': sid,
                              'ref_name': ref.name})
        except Exception as e:
            _vjob_set(jid, status='error', error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'job_id': jid})


# ── POST /video/upload-media ──────────────────────────────────────────────────
@app.route('/video/upload-media', methods=['POST'])
def video_upload_media():
    sid   = request.form.get('session_id', '').strip()
    files = request.files.getlist('media')
    if not sid or not files:
        return jsonify({'error': 'بيانات ناقصة'}), 400

    sd = _vsession(sid)
    saved = []
    for f in files:
        if not f or _vext(f.filename) not in VIDEO_ALL:
            continue
        name = secure_filename(f.filename)
        dest = sd / 'MEDIA' / name
        f.save(str(dest))

        thumb_out = sd / 'MEDIA_THUMBS' / (Path(name).stem + '.jpg')
        if _is_img(name):
            try:
                img = Image.open(str(dest)).convert('RGB')
                img.thumbnail((320, 320))
                img.save(str(thumb_out))
            except Exception:
                pass
        elif _is_vid(name):
            try:
                import subprocess as _sp
                _sp.run(['ffmpeg', '-y', '-ss', '0.5', '-i', str(dest),
                         '-vframes', '1', '-q:v', '3', '-vf', 'scale=320:-2',
                         str(thumb_out)], capture_output=True, timeout=15)
            except Exception:
                pass

        saved.append({'name': name,
                      'type': 'image' if _is_img(name) else 'video',
                      'thumb': thumb_out.name if thumb_out.exists() else None})
    return jsonify({'saved': saved})


# ── POST /video/match ─────────────────────────────────────────────────────────
@app.route('/video/match', methods=['POST'])
def video_match():
    data     = request.get_json() or {}
    sid      = data.get('session_id', '')
    scenes   = data.get('scenes', [])
    media_names = data.get('media', [])
    if not sid or not scenes or not media_names:
        return jsonify({'error': 'بيانات ناقصة'}), 400

    sd     = _vsession(sid)
    jid    = uuid.uuid4().hex
    _vjob_set(jid, status='processing', type='match',
              session_id=sid, progress=0)

    def _run():
        try:
            media_list = []
            for m in media_names:
                p = sd / 'MEDIA' / m['name']
                emb = (embed_image(str(p)) if m['type'] == 'image'
                       else embed_video_thumb(str(p), 1.0))
                media_list.append({'name': m['name'], 'path': str(p),
                                   'type': m['type'], 'embedding': emb})
                _vjob_progress(jid, int(len(media_list) / len(media_names) * 50))

            sc_embs = []
            for sc in scenes:
                th = sd / 'THUMBS' / sc.get('thumbnail', '')
                sc_embs.append(embed_image(str(th)) if th.exists() else None)
                _vjob_progress(jid, 50 + int(len(sc_embs) / len(scenes) * 40))

            assignments = match_media_to_scenes(sc_embs, media_list)
            result = []
            for sc, mi in zip(scenes, assignments):
                entry = {'scene_idx': sc['idx'], 'media': None}
                if mi is not None:
                    m = media_list[mi]
                    th_name = Path(m['name']).stem + '.jpg'
                    entry['media'] = {'name': m['name'], 'type': m['type'],
                                      'thumb': th_name}
                result.append(entry)

            _vjob_set(jid, status='done', progress=100, result=result)
        except Exception as e:
            _vjob_set(jid, status='error', error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'job_id': jid})


# ── POST /video/upload-intro-outro ───────────────────────────────────────────
@app.route('/video/upload-intro-outro', methods=['POST'])
def video_upload_intro_outro():
    sid  = request.form.get('session_id', '').strip()
    role = request.form.get('role', 'intro')   # intro | outro | hook
    f    = request.files.get('file')
    if not sid or not f or _vext(f.filename) not in VIDEO_VID:
        return jsonify({'error': 'بيانات ناقصة أو ملف غير صالح'}), 400

    sd    = _vsession(sid)
    fname = f'_{role}.mp4'
    dest  = sd / 'MEDIA' / fname
    f.save(str(dest))

    thumb = sd / 'MEDIA_THUMBS' / f'_{role}.jpg'
    try:
        import subprocess as _sp
        _sp.run(['ffmpeg', '-y', '-ss', '0.5', '-i', str(dest),
                 '-vframes', '1', '-q:v', '3', '-vf', 'scale=320:-2', str(thumb)],
                capture_output=True, timeout=15)
    except Exception:
        pass

    return jsonify({'name': fname,
                    'thumb': thumb.name if thumb.exists() else None})


# ── POST /video/assemble ──────────────────────────────────────────────────────
@app.route('/video/assemble', methods=['POST'])
def video_assemble():
    data        = request.get_json() or {}
    sid         = data.get('session_id', '')
    scenes      = data.get('scenes', [])
    assignments = data.get('assignments', [])   # [{scene_idx, media_name or null}]
    audio_track = data.get('audio_track', 'full')
    transitions = data.get('transitions', [])   # list of str — one per scene
    intro_name  = data.get('intro')             # filename in MEDIA or null
    outro_name  = data.get('outro')             # filename in MEDIA or null
    if not sid or not scenes:
        return jsonify({'error': 'بيانات ناقصة'}), 400

    sd  = _vsession(sid)
    jid = uuid.uuid4().hex
    _vjob_set(jid, status='processing', type='assemble',
              session_id=sid, progress=0)

    def _run():
        try:
            # ── تأكد من وجود ref.mp4 في WORK ──
            ref_vid = sd / 'WORK' / 'ref.mp4'
            if not ref_vid.exists():
                for p in sorted((sd / 'INPUT').iterdir()):
                    if _vext(p.name) in VIDEO_VID:
                        shutil.copy(str(p), str(ref_vid))
                        break

            info = get_video_info(str(ref_vid)) if ref_vid.exists() else \
                   {'width': 1080, 'height': 1920, 'fps': 30}

            # ── بناء media_list من assignments ──
            assign_map  = {a['scene_idx']: a.get('media_name') for a in assignments}
            media_list  = []
            name_to_idx = {}
            for sc in scenes:
                mn = assign_map.get(sc['idx'])
                if mn and mn not in name_to_idx:
                    p = sd / 'MEDIA' / mn
                    if p.exists():
                        name_to_idx[mn] = len(media_list)
                        media_list.append({'name': mn, 'path': str(p),
                                           'type': 'image' if _is_img(mn) else 'video'})

            # ── Fallback: ما في assignments صالحة → خذ كل ملفات MEDIA ──
            if not media_list:
                for p in sorted((sd / 'MEDIA').iterdir()):
                    if p.is_file() and _vext(p.name) in VIDEO_ALL:
                        media_list.append({'name': p.name, 'path': str(p),
                                           'type': 'image' if _is_img(p.name) else 'video'})
                sc_assignments = [None] * len(scenes)  # round-robin in assemble_template
            else:
                sc_assignments = [name_to_idx.get(assign_map.get(sc['idx')))
                                  for sc in scenes]

            if not media_list:
                _vjob_set(jid, status='error',
                          error='ما في ميديا — ارفع صور أو فيديوهات أولاً')
                return

            intro_path = str(sd / 'MEDIA' / intro_name) if intro_name else None
            outro_path = str(sd / 'MEDIA' / outro_name) if outro_name else None

            out_path = sd / 'OUTPUT' / 'final.mp4'
            assemble_template(
                scenes, sc_assignments, media_list,
                audio_track, info,
                str(sd / 'WORK'), str(out_path),
                ref_video_path=str(ref_vid) if ref_vid.exists() else None,
                cb=lambda p: _vjob_progress(jid, p),
                transitions=transitions or [],
                intro_path=intro_path,
                outro_path=outro_path,
            )
            _vjob_set(jid, status='done', progress=100,
                      result={'output': str(out_path)})
        except Exception as e:
            _vjob_set(jid, status='error', error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'job_id': jid})


# ── POST /video/assemble-review ──────────────────────────────────────────────
@app.route('/video/assemble-review', methods=['POST'])
def video_assemble_review():
    """
    تجميع نهائي من واجهة المراجعة — يدعم:
    - ترتيب مخصص للمقاطع
    - مدد مخصصة لكل مقطع
    - انتقالات (cut / fade / dissolve / wipe) بين المقاطع
    """
    data        = request.get_json() or {}
    sid         = data.get('session_id', '')
    scenes      = data.get('scenes', [])
    assignments = data.get('assignments', [])
    trans_map   = data.get('transitions', {})
    effects_map = data.get('effects', {})   # {clipId: {type, intensity}}
    scales_map  = data.get('scales', {})    # {clipId: number}
    audio_track = data.get('audio_track', 'full')

    if not sid or not scenes:
        return jsonify({'error': 'بيانات ناقصة'}), 400

    sd  = _vsession(sid)
    jid = uuid.uuid4().hex
    _vjob_set(jid, status='processing', type='assemble_review',
              session_id=sid, progress=0)

    def _run():
        try:
            ref_vid = sd / 'WORK' / 'ref.mp4'
            if not ref_vid.exists():
                for p in sorted((sd / 'INPUT').iterdir()):
                    if _vext(p.name) in VIDEO_VID:
                        shutil.copy(str(p), str(ref_vid))
                        break

            info = get_video_info(str(ref_vid)) if ref_vid.exists() else \
                   {'width': 1080, 'height': 1920, 'fps': 30}

            assign_map  = {a['scene_idx']: a.get('media_name') for a in assignments}
            media_list  = []
            name_to_idx = {}
            for sc in scenes:
                mn = assign_map.get(sc['idx'])
                if mn and mn not in name_to_idx:
                    p = sd / 'MEDIA' / mn
                    if p.exists():
                        name_to_idx[mn] = len(media_list)
                        media_list.append({'name': mn, 'path': str(p),
                                           'type': 'image' if _is_img(mn) else 'video'})

            if not media_list:
                for p in sorted((sd / 'MEDIA').iterdir()):
                    if p.is_file() and _vext(p.name) in VIDEO_ALL:
                        media_list.append({'name': p.name, 'path': str(p),
                                           'type': 'image' if _is_img(p.name) else 'video'})
                sc_assignments = [None] * len(scenes)  # round-robin in assemble_template
            else:
                sc_assignments = [name_to_idx.get(assign_map.get(sc['idx')))
                                  for sc in scenes]

            if not media_list:
                _vjob_set(jid, status='error',
                          error='ما في ميديا — أضف مقاطع أولاً')
                return

            # Inject transitions + effects into scenes list
            clip_ids = [f'c{i}' for i in range(len(scenes))]
            for i, sc in enumerate(scenes):
                sc['transition'] = trans_map.get(str(i), 'cut') if i > 0 else 'cut'
                cid = sc.get('clip_id', clip_ids[i] if i < len(clip_ids) else str(i))
                fx = effects_map.get(cid, sc.get('effect', {'type':'none','intensity':50}))
                sc['effect']     = fx
                sc['scale']      = scales_map.get(cid, sc.get('scale', 100))

            out_path = sd / 'OUTPUT' / 'reviewed.mp4'
            assemble_template(
                scenes, sc_assignments, media_list,
                audio_track, info,
                str(sd / 'WORK'), str(out_path),
                ref_video_path=str(ref_vid) if ref_vid.exists() else None,
                cb=lambda p: _vjob_progress(jid, p),
            )
            _vjob_set(jid, status='done', progress=100,
                      result={'output': str(out_path)})
        except Exception as e:
            _vjob_set(jid, status='error', error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'job_id': jid})



@app.route('/video/ai-edit', methods=['POST'])
def video_ai_edit():
    """
    وضع AI Edit الجديد:
    - ترفع فيديو مرجعي (reference) + صور وفيديوهات (media)
    - النظام يحلل المرجعي ويعمل فيديو زيه من المحتوى المرفوع
    """
    from video_maker import assemble_ai_edit

    ref_file    = request.files.get('reference')
    media_files = request.files.getlist('media')
    audio_track = request.form.get('audio_track', 'full')

    if not ref_file or _vext(ref_file.filename) not in VIDEO_VID:
        return jsonify({'error': 'أرفع فيديو مرجعي (mp4/mov)'}), 400
    if not media_files:
        return jsonify({'error': 'أرفع صور أو فيديوهات'}), 400

    sid = uuid.uuid4().hex
    jid = uuid.uuid4().hex
    sd  = _vsession(sid)

    # حفظ المرجعي
    ref_name = secure_filename(ref_file.filename)
    ref_path = sd / 'INPUT' / ref_name
    ref_file.save(str(ref_path))
    shutil.copy(str(ref_path), str(sd / 'WORK' / 'ref.mp4'))

    # حفظ الميديا
    saved_media = []
    for f in media_files:
        if not f or _vext(f.filename) not in VIDEO_ALL:
            continue
        name = secure_filename(f.filename)
        dest = sd / 'MEDIA' / name
        f.save(str(dest))
        saved_media.append({'name': name, 'path': str(dest),
                            'type': 'image' if _is_img(name) else 'video'})

    if not saved_media:
        return jsonify({'error': 'ما في ميديا صالحة'}), 400

    _vjob_set(jid, status='processing', type='ai_edit',
              session_id=sid, progress=0)

    def _run():
        try:
            # embeddings للميديا
            for i, m in enumerate(saved_media):
                emb = (embed_image(m['path']) if m['type'] == 'image'
                       else embed_video_thumb(m['path'], 1.0))
                m['embedding'] = emb
                _vjob_progress(jid, int((i + 1) / len(saved_media) * 20))

            out_path = sd / 'OUTPUT' / 'ai_edit.mp4'
            (sd / 'OUTPUT').mkdir(exist_ok=True)

            assemble_ai_edit(
                ref_video_path=str(sd / 'WORK' / 'ref.mp4'),
                media_list=saved_media,
                work_dir=str(sd / 'WORK'),
                out_path=str(out_path),
                audio_track=audio_track,
                use_understanding=True,
                cb=lambda p: _vjob_progress(jid, 20 + int(p * 0.8)),
            )
            _vjob_set(jid, status='done', progress=100,
                      result={'output': str(out_path)})
        except Exception as e:
            _vjob_set(jid, status='error', error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'job_id': jid, 'session_id': sid})


# ── GET /video/status/<jid> ───────────────────────────────────────────────────
@app.route('/video/status/<jid>')
def video_status(jid):
    with VIDEO_JOBS_LOCK:
        job = dict(VIDEO_JOBS.get(jid, {}))
    # don't serialize numpy arrays
    if 'result' in job and job['result']:
        r = job['result']
        if 'scenes' in r:
            for sc in r['scenes']:
                sc.pop('embedding', None)
    return jsonify(job) if job else (jsonify({'error': 'not found'}), 404)


# ── Static file servers ───────────────────────────────────────────────────────
@app.route('/video/thumb/<sid>/<fname>')
def video_thumb(sid, fname):
    p = VIDEO_DIR / sid / 'THUMBS' / secure_filename(fname)
    return send_file(str(p)) if p.exists() else ('Not found', 404)

@app.route('/video/media-thumb/<sid>/<fname>')
def video_media_thumb(sid, fname):
    p = VIDEO_DIR / sid / 'MEDIA_THUMBS' / secure_filename(fname)
    return send_file(str(p)) if p.exists() else ('Not found', 404)

@app.route('/video/stream/<jid>')
def video_stream(jid):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(jid, {})
    out = job.get('result', {}).get('output')
    if not out or not Path(out).exists():
        return 'Not found', 404
    return send_file(str(out), mimetype='video/mp4',
                     conditional=True)

@app.route('/video/download/<jid>')
def video_download(jid):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(jid, {})
    out = job.get('result', {}).get('output')
    if not out or not Path(out).exists():
        return 'Not found', 404
    fname = 'TechArabi_Final.mp4' if 'reviewed' in str(out) else 'TechArabi_Video.mp4'
    return send_file(str(out), as_attachment=True,
                     download_name=fname)


# ── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)