from PIL import Image

class Watermarker:
    def __init__(self, watermark_path, scale=0.15, opacity=200, padding=20):
        self.watermark_path = watermark_path
        # تم تغيير الحجم من 8% إلى 12% (0.12)
        self.scale = scale       
        self.opacity = opacity   # Visibility: 100% (255)
        self.padding = padding   # Padding: 20 pixels

    def apply(self, background_img):
        """
        يضع الشعار في الزاوية العلوية اليمنى (Top-Right)
        """
        try:
            # نفتح الشعار بصيغة RGBA
            watermark = Image.open(self.watermark_path).convert("RGBA")

            # طبّق الشفافية دائماً — يعمل سواء كان الشعار RGBA أو RGB
            r, g, b, a = watermark.split()
            a = a.point(lambda p: int(p * (self.opacity / 255.0)))
            watermark = Image.merge('RGBA', (r, g, b, a))

            # حساب الحجم الجديد بناءً على مقياس الـ 12% الجديد من عرض الصورة
            bg_w, bg_h = background_img.size
            new_w = int(bg_w * self.scale)
            new_h = int(watermark.size[1] * (new_w / watermark.size[0]))
            watermark = watermark.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # تحديد الإحداثيات (أعلى اليمين Top-Right)
            x = bg_w - new_w - self.padding
            y = self.padding

            # دمج الشعار
            transparent = Image.new('RGBA', background_img.size, (0,0,0,0))
            transparent.paste(watermark, (x, y), mask=watermark)
            final_img = Image.alpha_composite(background_img.convert("RGBA"), transparent)
            
            return final_img.convert("RGB") 

        except Exception as e:
            print(f"Error applying watermark: {e}")
            return background_img