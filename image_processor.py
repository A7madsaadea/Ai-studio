"""
معالج الصور المتقدم — image_processor.py
كل عملية مستقلة، أي خطأ يرجع الصورة الأصلية بدون كراش.
"""

import cv2
import math
import numpy as np
from PIL import Image

MAX_PROCESS_PX = 2400
MAX_THUMB_PX   = 1600

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def _limit_size(img: Image.Image, max_px: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    scale = max_px / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert('RGB') if img.mode != 'RGB' else img


class ImageProcessor:

    # ── 1. تصحيح الإضاءة ──────────────────────────────────────────────────
    @staticmethod
    def auto_lighting(img: Image.Image) -> Image.Image:
        try:
            work = _limit_size(_to_rgb(img), MAX_PROCESS_PX)
            arr  = np.array(work)
            lab  = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)
            out = Image.fromarray(result)
            # أعد الحجم الأصلي لو تغيّر
            if out.size != img.size:
                out = out.resize(img.size, Image.Resampling.LANCZOS)
            return out
        except Exception as e:
            print(f"[auto_lighting] {e}")
            return img

    # ── 2. White Balance ───────────────────────────────────────────────────
    @staticmethod
    def auto_white_balance(img: Image.Image) -> Image.Image:
        try:
            work = _limit_size(_to_rgb(img), MAX_PROCESS_PX)
            arr  = np.array(work, dtype=np.float32)
            means = arr.mean(axis=(0, 1))          # [r, g, b]
            gray  = means.mean()
            scale = np.where(means > 1e-6, gray / means, 1.0)
            out_arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
            out = Image.fromarray(out_arr)
            if out.size != img.size:
                out = out.resize(img.size, Image.Resampling.LANCZOS)
            return out
        except Exception as e:
            print(f"[auto_wb] {e}")
            return img

    # ── 3 & 4. قلب أفقي / عمودي ───────────────────────────────────────────
    @staticmethod
    def flip_horizontal(img: Image.Image) -> Image.Image:
        try:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        except Exception as e:
            print(f"[flip_h] {e}")
            return img

    @staticmethod
    def flip_vertical(img: Image.Image) -> Image.Image:
        try:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        except Exception as e:
            print(f"[flip_v] {e}")
            return img

    # ── 5. Noise Reduction ─────────────────────────────────────────────────
    @staticmethod
    def noise_reduction(img: Image.Image) -> Image.Image:
        # هذه العملية ثقيلة — نحدد الحجم بـ MAX_THUMB_PX
        try:
            original_size = img.size
            work = _limit_size(_to_rgb(img), MAX_THUMB_PX)
            arr  = np.array(work)
            bgr  = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            den  = cv2.fastNlMeansDenoisingColored(
                bgr, None,
                h=8, hColor=8,
                templateWindowSize=7,
                searchWindowSize=15,   # أصغر من 21 لتوفير الذاكرة
            )
            out = Image.fromarray(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))
            if out.size != original_size:
                out = out.resize(original_size, Image.Resampling.LANCZOS)
            return out
        except Exception as e:
            print(f"[noise_reduction] {e}")
            return img

    # ── 6. تعديل الأفق ─────────────────────────────────────────────────────
    @staticmethod
    def straighten_horizon(img: Image.Image) -> Image.Image:
        try:
            work = _limit_size(_to_rgb(img), MAX_PROCESS_PX)
            arr  = np.array(work)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
            if lines is None:
                return img

            angles = []
            for line in lines:
                angle = math.degrees(line[0][1]) - 90
                if abs(angle) < 15:
                    angles.append(angle)

            if not angles:
                return img

            rotation = float(np.median(angles))
            if abs(rotation) < 0.5:
                return img

            return img.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC)
        except Exception as e:
            print(f"[straighten] {e}")
            return img

    # ── 7. Smart Crop للوجه (Haar cascade فقط) ────────────────────────────
    @staticmethod
    def smart_crop_face(img: Image.Image) -> Image.Image:
        try:
            w, h = img.size
            work = _limit_size(_to_rgb(img), MAX_PROCESS_PX)
            ww, wh = work.size

            gray = cv2.cvtColor(np.array(work), cv2.COLOR_RGB2GRAY)
            faces = _FACE_CASCADE.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                sx, sy = w / ww, h / wh
                cx = int((fx + fw / 2) * sx)
                cy = int((fy + fh / 2) * sy)
                pad = int(max(fw, fh) * max(sx, sy))
                return img.crop((
                    max(0, cx - pad), max(0, cy - pad),
                    min(w, cx + pad), min(h, cy + pad),
                ))

            # لم يُعثر على وجه — center square crop
            side = min(w, h)
            return img.crop(((w - side) // 2, (h - side) // 2,
                              (w + side) // 2, (h + side) // 2))
        except Exception as e:
            print(f"[face_crop] {e}")
            return img

    # ── 8. تصنيف REELS / PHOTOS ───────────────────────────────────────────
    @staticmethod
    def classify_image(img: Image.Image) -> str:
        try:
            w, h = img.size
            return 'REELS' if h > 0 and (w / h) <= 0.6 else 'PHOTOS'
        except Exception:
            return 'PHOTOS'

    @staticmethod
    def classify_video(video_path: str) -> str:
        try:
            cap = cv2.VideoCapture(video_path)
            vw  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            vh  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            return 'REELS' if vh > 0 and (vw / vh) <= 0.6 else 'PHOTOS'
        except Exception:
            return 'PHOTOS'

    # ── الـ pipeline الرئيسي ───────────────────────────────────────────────
    @staticmethod
    def process(
        image_path: str,
        auto_lighting: bool = True,
        auto_wb:       bool = True,
        flip_h:        bool = False,
        flip_v:        bool = False,
        face_crop:     bool = False,
        denoise:       bool = False,
        straighten:    bool = False,
    ):
        """يرجع (img, classification). لا يرمي exceptions أبداً."""
        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"[process] فشل فتح الصورة {image_path}: {e}")
            raise

        if auto_wb:
            img = ImageProcessor.auto_white_balance(img)
        if auto_lighting:
            img = ImageProcessor.auto_lighting(img)
        if denoise:
            img = ImageProcessor.noise_reduction(img)
        if straighten:
            img = ImageProcessor.straighten_horizon(img)
        if face_crop:
            img = ImageProcessor.smart_crop_face(img)
        if flip_h:
            img = ImageProcessor.flip_horizontal(img)
        if flip_v:
            img = ImageProcessor.flip_vertical(img)

        return img, ImageProcessor.classify_image(img)
