"""
clip_matcher.py — المرحلة 3 (ب): مطابقة وسائط المستخدم مع مشاهد المرجعي بـ OpenCLIP

الفصل مهم: الجزء الذي يحمّل النموذج (ClipEmbedder) مستقل عن منطق الاختيار (match_media)،
فنقدر نختبر قرارات الاختيار بدون تحميل أي نموذج.

لكل مشهد مرجعي يرجّع عدة خيارات مرتبة (top_k) مع تفصيل الدرجة، والخيار المختار يراعي:
  - التشابه البصري (OpenCLIP)
  - جودة الملف
  - ملاءمة الاتجاه (عمودي/أفقي) لفيديو المرجعي
  - كفاية مدة الفيديو للقطة
  - التنوع: عقوبة على تكرار نفس الملف
"""

import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_SETTINGS = {
    "model_name":     "ViT-B-32",   # صغير ومناسب لـ 4GB VRAM
    "pretrained":     "openai",
    "batch_size":     16,
    "w_similarity":   0.60,
    "w_quality":      0.25,
    "w_orientation":  0.10,
    "w_duration":     0.05,
    "reuse_penalty":  0.35,         # عقوبة لكل مرة سبق فيها استخدام نفس الملف
    "top_k":          5,
}


def _merge_settings(user):
    s = dict(DEFAULT_SETTINGS)
    for k, v in (user or {}).items():
        if k in s and v is not None:
            s[k] = type(s[k])(v)
    return s


# ── تحميل النموذج وحساب الـ embeddings ────────────────────────────────────────

class ClipEmbedder:
    """يحمّل OpenCLIP عند أول استخدام. GPU إذا متاح، وإلا CPU."""

    def __init__(self, model_name="ViT-B-32", pretrained="openai", batch_size=16):
        self.model_name, self.pretrained, self.batch_size = model_name, pretrained, batch_size
        self._model = self._prep = self._torch = None
        self.device = "cpu"

    def _load(self):
        if self._model is not None:
            return
        import torch
        import open_clip
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, prep = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained)
        self._model = model.to(self.device).eval()
        self._prep = prep

    def embed_images(self, paths):
        """يرجع مصفوفة (N, D) مطبَّعة. الصورة التي تفشل قراءتها تأخذ صفًّا من الأصفار."""
        from PIL import Image
        self._load()
        torch = self._torch
        out = [None] * len(paths)
        for i in range(0, len(paths), self.batch_size):
            chunk, idxs = [], []
            for j, p in enumerate(paths[i:i + self.batch_size]):
                try:
                    chunk.append(self._prep(Image.open(p).convert("RGB")))
                    idxs.append(i + j)
                except Exception as e:
                    print(f"[clip] skip {p}: {e}")
            if not chunk:
                continue
            x = torch.stack(chunk).to(self.device)
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        f = self._model.encode_image(x)
                else:
                    f = self._model.encode_image(x)
            f = f.float()
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
            for k, vec in zip(idxs, f.cpu().numpy()):
                out[k] = vec
        dim = next((v.shape[0] for v in out if v is not None), 512)
        return np.stack([v if v is not None else np.zeros(dim, np.float32) for v in out])


def _file_key(path):
    p = Path(path)
    st = p.stat()
    return hashlib.sha1(f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()


def embed_with_cache(embedder, paths, cache_file):
    """يحسب embeddings للمسارات الجديدة فقط ويحفظ الباقي بالكاش (إعادة التوليد تصير سريعة)."""
    cache_file = Path(cache_file)
    cache = {}
    if cache_file.exists():
        try:
            z = np.load(cache_file)
            cache = {k: z[k] for k in z.files}
        except Exception:
            cache = {}
    keys = [_file_key(p) for p in paths]
    missing = [(k, p) for k, p in zip(keys, paths) if k not in cache]
    if missing:
        vecs = embedder.embed_images([p for _, p in missing])
        for (k, _), v in zip(missing, vecs):
            cache[k] = v
        np.savez(cache_file, **cache)
    return np.stack([cache[k] for k in keys])


# ── منطق الاختيار (مستقل عن النموذج) ──────────────────────────────────────────

def _norm_minmax(x):
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-6 else np.full_like(x, 0.5)


def match_media(reference, media, scene_emb, frame_emb, settings=None):
    """
    reference : ناتج reference_analyzer (reference.json)
    media     : ناتج media_analyzer (media.json)
    scene_emb : (S, D) embedding لصورة كل مشهد مرجعي، بنفس ترتيب reference["scenes"]
    frame_emb : dict {media_id: (F, D)} embeddings لإطارات كل ملف، بنفس ترتيب item["frames"]

    يرجّع قائمة، عنصر لكل مشهد:
      {scene_index, duration, chosen: {...}, options: [ ... top_k ... ]}
    """
    s = _merge_settings(settings)
    items = media["items"]
    if not items:
        raise ValueError("لا توجد وسائط صالحة للمطابقة")

    ref_orient = reference["video"].get("orientation", "vertical")
    used = {m["id"]: 0 for m in items}
    results = []

    for si, scene in enumerate(reference["scenes"]):
        # أفضل إطار لكل ملف + تشابهه مع هذا المشهد
        raw_sim, best_frame = [], []
        for m in items:
            sims = frame_emb[m["id"]] @ scene_emb[si]
            bi = int(np.argmax(sims))
            raw_sim.append(float(sims[bi]))
            best_frame.append(bi)
        sim_n = _norm_minmax(raw_sim)   # CLIP يعطي أرقامًا متقاربة، فنفرد الفروق داخل كل مشهد

        cands = []
        for k, m in enumerate(items):
            orient = 1.0 if m["orientation"] == ref_orient else (0.6 if m["orientation"] == "square" else 0.2)
            if m["type"] == "video":
                usable = max(0.0, m["duration"] - m["best_start"])
                dur_fit = min(1.0, usable / max(scene["duration"], 0.1))
            else:
                dur_fit = 1.0
            base = (s["w_similarity"] * sim_n[k] + s["w_quality"] * m["quality"] +
                    s["w_orientation"] * orient + s["w_duration"] * dur_fit)
            final = base - s["reuse_penalty"] * used[m["id"]]

            frame = m["frames"][best_frame[k]]
            source_in = 0.0
            if m["type"] == "video":
                # نبدأ من أفضل نافذة، لكن نضمن أن اللقطة لا تتجاوز نهاية الفيديو
                source_in = min(m["best_start"], max(0.0, m["duration"] - scene["duration"]))
            cands.append({
                "media_id": m["id"], "name": m["name"], "type": m["type"],
                "score": round(final, 3),
                "similarity": round(raw_sim[k], 3),
                "quality": m["quality"],
                "orientation_ok": m["orientation"] == ref_orient,
                "reused_before": used[m["id"]],
                "source_in": round(source_in, 3),
                "match_frame_t": frame["t"],
            })

        cands.sort(key=lambda c: c["score"], reverse=True)
        chosen = cands[0]
        used[chosen["media_id"]] += 1
        results.append({
            "scene_index": scene["index"],
            "duration": scene["duration"],
            "chosen": chosen,
            "options": cands[: s["top_k"]],
        })
    return results


# ── الدالة الرئيسية ───────────────────────────────────────────────────────────

def run_matching(reference, media, out_dir, settings=None, embedder=None, progress_cb=None):
    """
    يحسب الـ embeddings (مع كاش) ثم يطابق ويحفظ matches.json.
    embedder : اختياري، للاختبار أو لإعادة استخدام نموذج محمّل.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = _merge_settings(settings)
    cb = progress_cb or (lambda p: None)
    embedder = embedder or ClipEmbedder(s["model_name"], s["pretrained"], s["batch_size"])

    thumbs_dir = out_dir / "ref_thumbs"
    scene_paths = [str(thumbs_dir / sc["thumbnail"]) for sc in reference["scenes"]]
    cache = out_dir / "embed_cache.npz"

    scene_emb = embed_with_cache(embedder, scene_paths, cache)
    cb(30)

    all_paths, owners = [], []
    for m in media["items"]:
        for f in m["frames"]:
            all_paths.append(f["path"])
            owners.append(m["id"])
    vecs = embed_with_cache(embedder, all_paths, cache)
    cb(80)

    frame_emb = {}
    for mid, v in zip(owners, vecs):
        frame_emb.setdefault(mid, []).append(v)
    frame_emb = {k: np.stack(v) for k, v in frame_emb.items()}

    matches = match_media(reference, media, scene_emb, frame_emb, s)
    (out_dir / "matches.json").write_text(
        json.dumps({"version": 1, "settings": s, "matches": matches},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    cb(100)
    return matches


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python clip_matcher.py <analysis_dir>   (فيه reference.json و media.json)")
        sys.exit(1)
    d = Path(sys.argv[1])
    ref = json.loads((d / "reference.json").read_text(encoding="utf-8"))
    med = json.loads((d / "media.json").read_text(encoding="utf-8"))
    ms = run_matching(ref, med, d, progress_cb=lambda p: print(f"{p}%", end="\r"))
    print()
    for m in ms:
        c = m["chosen"]
        alt = ", ".join(f"{o['name'][:14]}({o['score']:.2f})" for o in m["options"][1:3])
        print(f"مشهد {m['scene_index']:02d} ({m['duration']:.2f}s) → {c['name'][:24]:24} "
              f"score={c['score']:.2f} sim={c['similarity']:.2f} | بدائل: {alt}")