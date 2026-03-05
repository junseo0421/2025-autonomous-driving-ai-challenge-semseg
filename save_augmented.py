# save_augmented.py
import os, math, random, argparse
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageOps
import re
import cv2
import pywt

# ---------- 공용 유틸 ----------
def _split_list(s):
    return [x.strip().lower() for x in re.split(r'[+,]', s) if x.strip()]

def parse_aug_argument(s, default_sev=1):
    s = s.strip().lower()

    def parse_token(tok):
        m = re.match(r'^([a-z_]+)\s*(?:[:@]\s*(\d+))?$', tok.strip())
        if not m:
            raise ValueError(f"잘못된 aug 토큰: {tok}")
        name = m.group(1)
        sev  = int(m.group(2)) if m.group(2) else default_sev
        return (name, max(1, min(5, sev)))  # 1~5로 클램프

    # all 또는 all:K → 각각 따로 저장
    m_all = re.match(r'^all(?:[:@]\s*(\d+))?$', s)
    if m_all:
        sev_all = int(m_all.group(1)) if m_all.group(1) else default_sev
        sev_all = max(1, min(5, sev_all))
        return [
            [("haze", sev_all)],
            [("rain", sev_all)],
            [("raindrop", sev_all)],
            [("lowlight", sev_all)],
            [("overbright", sev_all)],  # ← 추가
        ]

    # 연속 적용(체인)
    if "+" in s:
        return [[parse_token(t) for t in s.split("+")]]

    # 각각 저장
    if "," in s:
        return [[parse_token(t)] for t in s.split(",")]

    # 단일
    return [[parse_token(s)]]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def pil_gamma(img, gamma):
    # Pillow 버전 상관없이 동작하는 감마
    lut = [min(255, int((i/255.0) ** gamma * 255 + 0.5)) for i in range(256)]
    if img.mode == "RGB":
        return img.point(lut * 3)
    elif img.mode == "L":
        return img.point(lut)
    else:
        return img.convert("RGB").point(lut * 3)

# ---------- 1) Haze/Fog ----------
class AddHazeTV:
    def __init__(self, beta=(0.6, 1.6), A=(0.85, 0.98), blur_ratio=0.05):
        self.beta, self.A, self.blur_ratio = beta, A, blur_ratio
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        beta = np.random.uniform(*self.beta)
        A = np.random.uniform(*self.A)
        gray = img.convert("L")
        depth = ImageOps.invert(gray).filter(
            ImageFilter.GaussianBlur(radius=max(1, int(min(w, h) * self.blur_ratio))))
        depth = np.asarray(depth, np.float32) / 255.0
        t = np.exp(-beta * depth)[..., None]
        I = np.asarray(img, np.float32) / 255.0
        out = I * t + A * (1.0 - t)
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))

# ---------- 2) Rain Streaks ----------
class AddRainStreaksTV:
    def __init__(self, density=(200, 600), length=(10, 22), angle=(-15, 15),
                 alpha=(0.15, 0.35), blur=1.2, width=(1, 2), color=(225, 225, 225)):
        self.density, self.length, self.angle = density, length, angle
        self.alpha, self.blur = alpha, blur
        self.width = width
        self.color = color  # RGB 값

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        den = np.random.randint(*self.density)
        L   = np.random.randint(*self.length)
        ang = np.random.uniform(*self.angle)
        a   = np.random.uniform(*self.alpha)
        thick = np.random.randint(self.width[0], self.width[1] + 1)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        dx = int(L * math.cos(math.radians(ang)))
        dy = int(L * math.sin(math.radians(ang)))

        col = (*self.color, 255)  # 불투명 라인 후 전체 알파로 블렌드
        for _ in range(den):
            x, y = random.randrange(w), random.randrange(h)
            draw.line([(x, y), (x + dx, y + dy)], fill=col, width=thick)

        overlay = overlay.filter(ImageFilter.GaussianBlur(self.blur))
        out = Image.alpha_composite(
            img.convert("RGBA"),
            Image.blend(Image.new("RGBA", (w, h), (0,0,0,0)), overlay, a)
        )
        return out.convert("RGB")

# ---------- 3) Raindrops(근사) ----------
class AddRaindropsTV:
    def __init__(self, num=(15, 60), radius=(5, 22), alpha=(0.25, 0.55), blur=(1.5, 3.5)):
        self.num, self.radius, self.alpha, self.blur = num, radius, alpha, blur
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        n = np.random.randint(*self.num)
        a = np.random.uniform(*self.alpha)
        br = np.random.uniform(*self.blur)
        drop = Image.new("L", (w, h), 0)
        rim  = Image.new("L", (w, h), 0)
        d1, d2 = ImageDraw.Draw(drop), ImageDraw.Draw(rim)
        for _ in range(n):
            r = np.random.randint(*self.radius)
            x, y = np.random.randint(-r, w + r), np.random.randint(-r, h + r)
            box = (x - r, y - r, x + r, y + r)
            d1.ellipse(box, fill=200)                         # 내부 디밍
            d2.ellipse(box, outline=255, width=max(1, r // 6))# 림 하이라이트
        drop = drop.filter(ImageFilter.GaussianBlur(br))
        rim  = rim.filter(ImageFilter.GaussianBlur(max(0.5, br / 2)))
        base = img.convert("RGBA")
        dark = ImageEnhance.Brightness(img).enhance(0.8)
        bright = ImageEnhance.Brightness(img).enhance(1.2)
        base = Image.composite(dark, base, drop)
        base = Image.composite(bright, base, rim)
        return Image.blend(img, base.convert("RGB"), a)

# ---------- 4) Low-light ----------
class FastRetinexLowLightTV:
    """
    PIL.Image -> PIL.Image
    HSV의 V 채널만 대상으로 DWT(LL) + MSR + 역DWT.
    """
    def __init__(
        self,
        sigmas=(8, 40, 80),      # LL 기준 가우시안 시그마들 (원본 환산은 대략 ×2)
        weights=None,            # None이면 균등
        levels=1,                # DWT 레벨(1 또는 2 추천)
        alpha=0.6,               # LL과 MSR(LL)의 블렌딩 계수
        clip_percentiles=(1,99), # MSR 결과의 퍼센타일 클리핑
        wavelet='haar',
    ):
        self.sigmas = tuple(sigmas)
        self.weights = (
            np.ones(len(sigmas), dtype=np.float32) / len(sigmas)
            if weights is None else np.array(weights, dtype=np.float32)
        )
        assert len(self.weights) == len(self.sigmas)
        self.levels = int(levels)
        self.alpha = float(alpha)
        self.clip = clip_percentiles
        self.wavelet = wavelet

    # ---------- 내부 유틸 ----------
    def _dwt_levels(self, channel, levels):
        coeffs_stack = []
        LL = channel
        for _ in range(levels):
            LL, (LH, HL, HH) = pywt.dwt2(LL, self.wavelet, mode='symmetric')
            coeffs_stack.append((LH, HL, HH))
        return LL, coeffs_stack

    def _idwt_levels(self, LL, coeffs_stack):
        for (LH, HL, HH) in reversed(coeffs_stack):
            LL = pywt.idwt2((LL, (LH, HL, HH)), self.wavelet, mode='symmetric')
        return LL

    def _msr(self, img_u8, sigmas, weights):
        # 입력은 uint8 (LL 채널)
        x = img_u8.astype(np.float32) / 255.0
        logx = np.log1p(x)
        out = np.zeros_like(x, dtype=np.float32)
        for w, s in zip(weights, sigmas):
            blur = cv2.GaussianBlur(x, (0,0), float(s))
            out += w * (logx - np.log1p(blur))
        # 퍼센타일 기반 정규화 (깜빡임/들쭉날쭉 완화)
        lo, hi = np.percentile(out, self.clip)
        if hi <= lo:  # 드문 예외 대비
            lo, hi = out.min(), out.max()
        out = (out - lo) / max(1e-6, (hi - lo))
        out = np.clip(out, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8)

    # ---------- 호출 ----------
    def __call__(self, img: Image.Image) -> Image.Image:
        assert isinstance(img, Image.Image), "PIL.Image 입력을 기대합니다."
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        # RGB -> HSV (OpenCV는 BGR 기본이므로 주의)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # DWT 레벨 분해 (V 채널)
        LL, coeffs_stack = self._dwt_levels(v, self.levels)

        # MSR on LL
        LL_msr = self._msr(LL, self.sigmas, self.weights)

        # 블렌딩으로 과도한 대비 억제
        LL_blend = ((1.0 - self.alpha) * LL.astype(np.float32) +
                    self.alpha * LL_msr.astype(np.float32))
        LL_blend = np.clip(LL_blend, 0, 255).astype(np.uint8)

        # 역DWT로 V 복원
        v_enh = self._idwt_levels(LL_blend, coeffs_stack)
        v_enh = np.clip(v_enh, 0, 255).astype(np.uint8)

        hsv_enh = cv2.merge([h, s, v_enh])
        rgb_enh = cv2.cvtColor(hsv_enh, cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb_enh, mode="RGB")


# ---------- 5) Over-brighten (lowlight → normal/bright) ----------
class OverBrightTV:
    """
    HSV의 V 채널 증폭 + 하이라이트 롤오프 + 감마(<1)로 자연스러운 과다노출(밝기 상승).
    sat>1로 약간의 채도 복원, contrast는 필요 시 소폭 조정.
    """
    def __init__(self, gain=1.4, gamma=0.85, sat=1.06, rolloff=0.6, contrast=1.0):
        self.gain = float(gain)      # 전체 밝기 승수 (V 스케일)
        self.gamma = float(gamma)    # 감마 (<1이면 더 밝게)
        self.sat = float(sat)        # 채도 배율
        self.rolloff = float(rolloff) # 하이라이트 압축 강도 (클수록 더 눌러짐)
        self.contrast = float(contrast)

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

        # RGB → HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        # [0,1] 정규화
        v01 = v / 255.0

        # 1) 밝기 승수 적용
        v_boost = v01 * self.gain

        # 2) 하이라이트 롤오프 (Reinhard 스타일)
        #    v' = v / (1 + rolloff * v)
        v_roll = v_boost / (1.0 + self.rolloff * v_boost)

        # 3) 감마(<1)로 추가 밝기
        v_gamma = np.power(np.clip(v_roll, 0.0, 1.0), self.gamma)

        # 4) 채도 살짝 올리기
        s_out = np.clip(s * self.sat, 0, 255).astype(np.uint8)

        v_out = np.clip(v_gamma * 255.0, 0, 255).astype(np.uint8)
        h_out = np.clip(h, 0, 255).astype(np.uint8)

        hsv_out = cv2.merge([h_out, s_out, v_out])
        rgb_out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)
        out = Image.fromarray(rgb_out, mode="RGB")

        # 5) 필요하면 약간의 콘트라스트 조정
        if abs(self.contrast - 1.0) > 1e-3:
            from PIL import ImageEnhance
            out = ImageEnhance.Contrast(out).enhance(self.contrast)

        return out


class OverBrightExposureTV:
    """
    Y(휘도)를 bilateral로 base/detail 분해 → base에 노출(ev) 상승 →
    하이라이트 소프트 숄더 압축 → 감마(<1) → 블랙 리프트 → detail 가중 재합성.
    채도는 약간만 올려 색 빠짐 방지.
    """
    def __init__(self,
                 ev=1.2,            # 노출 스톱(2^ev 배)
                 shoulder=1.5,      # 하이라이트 압축 강도 (↑ 더 눌림)
                 gamma=0.85,        # <1 이면 더 밝게
                 lift=0.07,         # 블랙 리프트(어두운 곳 살짝 들어올림)
                 bilateral_sc=35,   # bilateral sigmaColor
                 bilateral_ss=7,    # bilateral sigmaSpace
                 detail_gain=0.90,  # 디테일 비율(1이면 원본)
                 chroma_gain=1.06   # 색 보정 (1.0~1.1 권장)
                 ):
        self.ev = float(ev)
        self.shoulder = float(shoulder)
        self.gamma = float(gamma)
        self.lift = float(lift)
        self.bilateral_sc = float(bilateral_sc)
        self.bilateral_ss = float(bilateral_ss)
        self.detail_gain = float(detail_gain)
        self.chroma_gain = float(chroma_gain)

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

        # RGB → YCrCb (Y = 휘도)
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        Y, Cr, Cb = cv2.split(ycrcb)

        # base/detail 분해 (edge-preserving)
        base = cv2.bilateralFilter(Y, d=0,
                                   sigmaColor=self.bilateral_sc,
                                   sigmaSpace=self.bilateral_ss)
        detail = Y - base

        # [0,1] 정규화
        base01 = np.clip(base / 255.0, 0.0, 1.0)

        # 1) 노출 상승 (2^ev)
        exposed = base01 * (2.0 ** self.ev)

        # 2) 하이라이트 숄더 (Reinhard 스타일)
        comp = 1.0 - np.exp(-self.shoulder * exposed)

        # 3) 감마(<1)로 중간톤 추가 상승
        comp = np.clip(comp, 0.0, 1.0) ** self.gamma

        # 4) 블랙 리프트로 극저조도 영역도 살짝 들어올리기
        comp = comp * (1.0 - self.lift) + self.lift

        # 5) 디테일 재합성
        Y_new = np.clip(comp + self.detail_gain * (detail / 255.0), 0.0, 1.0) * 255.0

        # 6) 색 채널 살짝 증폭
        Cr_new = 128.0 + self.chroma_gain * (Cr - 128.0)
        Cb_new = 128.0 + self.chroma_gain * (Cb - 128.0)

        out_ycrcb = cv2.merge([Y_new.astype(np.float32),
                               Cr_new.astype(np.float32),
                               Cb_new.astype(np.float32)])
        out_rgb = cv2.cvtColor(out_ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        return Image.fromarray(out_rgb, mode="RGB")


# ---------- 5-c) Over-brighten (Curve + CLAHE, non-hazy) ----------
class OverBrightCurveTV:
    """
    Lab 공간 L 채널에 CLAHE 후, 섀도/미드톤만 올리는 톤 커브.
    y = x + s*(1-x)^2 + m*x*(1-x)  (x,y ∈ [0,1])
    - s: shadows lift, m: midtone lift
    뿌연 느낌 없이 어두운 영역 위주로 밝힘.
    """
    def __init__(self,
                 clip=2.0, tile=8,      # CLAHE 파라미터
                 s=0.25, m=0.12,        # 톤 커브 강도
                 sat=1.04,              # 채도 약간 보존
                 sharpen=0.0):          # 0이면 샤픈 끔 (권장: 0~0.4)
        self.clip = float(clip)
        self.tile = int(tile)
        self.s = float(s)
        self.m = float(m)
        self.sat = float(sat)
        self.sharpen = float(sharpen)

    def _tone_curve(self, L01):
        s, m = self.s, self.m
        y = L01 + s * (1.0 - L01) ** 2 + m * L01 * (1.0 - L01)
        return np.clip(y, 0.0, 1.0)

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(lab)

        # 1) CLAHE (로컬 대비 확보)
        clahe = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=(self.tile, self.tile))
        Lc = clahe.apply(L)

        # 2) 톤 커브 (섀도/미드톤 위주 리프트)
        L01 = np.clip(Lc.astype(np.float32) / 255.0, 0.0, 1.0)
        Lt  = (self._tone_curve(L01) * 255.0).astype(np.uint8)

        # 3) 색 보정 (채도 약간 복원)
        a = 128 + (a.astype(np.float32) - 128) * self.sat
        b = 128 + (b.astype(np.float32) - 128) * self.sat
        lab_out = cv2.merge([Lt, np.clip(a, 0, 255).astype(np.uint8), np.clip(b, 0, 255).astype(np.uint8)])
        out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

        # 4) (선택) 언샤프마스크
        if self.sharpen > 1e-6:
            blur = cv2.GaussianBlur(out, (0, 0), 1.2)
            out  = cv2.addWeighted(out, 1 + self.sharpen, blur, -self.sharpen, 0)

        return Image.fromarray(out, mode="RGB")

# ---------- 5-d) Over-brighten (Exposure Fusion, anti-crush) ----------
class OverBrightFusionTV:
    """
    3장의 가상 노출을 만든 뒤(cv2.createMergeMertens) 합성.
    - 극저조도에서도 중간톤 확보를 위해 큰 EV + 밝기 바이어스 적용
    - 합성 후 감마/퍼센타일 스트레치 + 바닥 리프트로 블랙 크러시 방지
    """
    def __init__(self, evs=(0.0, 3.5, 5.5), gammas=(1.0, 0.75, 0.60),
                 sat=1.06, contrast_w=0.50, satur_w=0.30, expose_w=1.00,
                 bright_bias=0.10,     # ← 합성 전 전체를 위로 올리는 바이어스(0~0.2)
                 post_gamma=0.80,      # ← 0.7~0.9 권장 (작을수록 밝아짐)
                 stretch=(0.1, 99.9),  # ← 퍼센타일 스트레치
                 floor=0.04,           # ← 최저 밝기 리프트(0~0.08)
                 unsharp=0.08):
        self.evs = tuple(evs)
        self.gammas = tuple(gammas)
        self.sat = float(sat)
        self.cw = float(contrast_w)
        self.sw = float(satur_w)
        self.ew = float(expose_w)
        self.bright_bias = float(bright_bias)
        self.post_gamma = float(post_gamma)
        self.stretch = stretch
        self.floor = float(floor)
        self.unsharp = float(unsharp)

    def _wb_grayworld(self, imgf):
        mean = imgf.reshape(-1, 3).mean(axis=0) + 1e-6
        gain = mean.mean() / mean
        return np.clip(imgf * gain, 0.0, 1.0)

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        im = rgb.astype(np.float32) / 255.0

        # 0) 간단 WB + 밝기 바이어스
        im = self._wb_grayworld(im)
        bb = self.bright_bias
        im = np.clip(im * (1.0 - bb) + bb, 0.0, 1.0)

        # 1) 가상 노출 3장 만들기 (크게 밝힘)
        views = []
        for ev, g in zip(self.evs, self.gammas):
            expo = np.clip(im * (2.0 ** float(ev)), 0.0, 1.0)
            expo = np.power(expo, float(g))  # g<1 → 더 밝게
            hsv = cv2.cvtColor((expo * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * self.sat, 0, 255)
            img_expo = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            views.append(cv2.cvtColor((img_expo * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0)

        merge = cv2.createMergeMertens(self.cw, self.sw, self.ew)
        fused = merge.process(views)  # BGR float [0,1]

        # 2) 후처리: 감마 + 퍼센타일 스트레치 + 바닥 리프트
        fused_rgb = cv2.cvtColor((fused * 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        fused_rgb = np.power(np.clip(fused_rgb, 0, 1), self.post_gamma)

        lo, hi = np.percentile(fused_rgb, self.stretch[0]), np.percentile(fused_rgb, self.stretch[1])
        fused_rgb = np.clip((fused_rgb - lo) / max(1e-6, (hi - lo)), 0, 1)

        # 바닥 리프트: 매우 어두운 구간 올리기
        f = self.floor
        fused_rgb = fused_rgb * (1.0 - f) + f

        out = (fused_rgb * 255).astype(np.uint8)

        if self.unsharp > 1e-6:
            blur = cv2.GaussianBlur(out, (0, 0), 1.2)
            out = cv2.addWeighted(out, 1 + self.unsharp, blur, -self.unsharp, 0)

        return Image.fromarray(out, mode="RGB")


# ---------- 5-e) Over-brighten (LIME-style illumination map) ----------
class OverBrightLIMETV:
    """
    L(x) = max(R,G,B)로 조명맵 추정 → 가우시안으로 부드럽게 → I/L^gamma 로 밝기 보정
    + 퍼센타일 스트레치 + 바닥 리프트 + 채도/대비 보정
    야간/극저조도에서도 블랙 크러시/뿌연 베일을 억제.
    """
    def __init__(self,
                 gamma_t=0.75,           # 조명맵 감마(↓ 더 밝게)
                 ksize=15, sigma=3.0,    # 조명맵 스무딩 커널/시그마
                 gain=1.00,              # 전체 밝기 추가 배율
                 floor=0.04,             # 최저 밝기 리프트(0~0.08)
                 stretch=(0.2, 99.8),    # 퍼센타일 스트레치 구간
                 sat=1.05, contrast=1.00,# 채도/대비
                 unsharp=0.06):          # 언샤프 마스크 강도(0이면 끔)
        self.gamma_t  = float(gamma_t)
        self.ksize    = int(ksize)
        self.sigma    = float(sigma)
        self.gain     = float(gain)
        self.floor    = float(floor)
        self.stretch  = stretch
        self.sat      = float(sat)
        self.contrast = float(contrast)
        self.unsharp  = float(unsharp)

    def __call__(self, img: Image.Image) -> Image.Image:
        import cv2, numpy as np
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8).astype(np.float32) / 255.0

        # 1) 조명맵 추정 & 스무딩
        L = np.max(rgb, axis=2)
        k = max(3, self.ksize | 1)  # 홀수 보장
        Ls = cv2.GaussianBlur(L, (k, k), self.sigma)
        Ls = np.clip(Ls, 1e-4, 1.0)

        # 2) 조명맵 감마 & 보정
        T = np.power(Ls, self.gamma_t)
        J = np.clip((rgb / T[..., None]) * self.gain, 0.0, 1.0)

        # 3) 퍼센타일 스트레치 + 바닥 리프트
        lo, hi = np.percentile(J, self.stretch[0]), np.percentile(J, self.stretch[1])
        J = np.clip((J - lo) / max(1e-6, (hi - lo)), 0, 1)
        if self.floor > 0:
            J = J * (1.0 - self.floor) + self.floor

        out = (J * 255).astype(np.uint8)

        # 4) 채도/대비 보정
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[...,1] = np.clip(hsv[...,1] * self.sat, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if abs(self.contrast - 1.0) > 1e-3:
            out = cv2.convertScaleAbs(out, alpha=self.contrast, beta=0)

        # 5) (선택) 언샤프
        if self.unsharp > 1e-6:
            blur = cv2.GaussianBlur(out, (0,0), 1.2)
            out  = cv2.addWeighted(out, 1 + self.unsharp, blur, -self.unsharp, 0)

        return Image.fromarray(out, mode="RGB")




# --- 추가: 비오는 감성 파이프라인 ---
class RainyLookTV:
    def __init__(self, severity=3, with_drops=True):
        s = max(1, min(5, severity))
        # severity에 따른 톤/헤이즈/빗줄기 강도 프리셋
        self.sat = [0.95, 0.9, 0.85, 0.8, 0.75][s-1]   # 채도 배율(<1이면 desat)
        self.brt = [0.95, 0.9, 0.88, 0.85, 0.82][s-1] # 밝기 배율
        self.cts = [0.95, 0.9, 0.85, 0.8, 0.75][s-1]  # 대비 배율
        self.cool = [0.02, 0.03, 0.04, 0.05, 0.06][s-1]  # 블루 톤 가중
        self.haze = [
            dict(beta=(0.4,0.8), A=(0.88,0.94), blur_ratio=0.03),
            dict(beta=(0.5,1.0), A=(0.90,0.95), blur_ratio=0.04),
            dict(beta=(0.6,1.2), A=(0.92,0.96), blur_ratio=0.05),
            dict(beta=(0.8,1.4), A=(0.93,0.98), blur_ratio=0.06),
            dict(beta=(1.0,1.6), A=(0.94,0.99), blur_ratio=0.07),
        ][s-1]
        self.rain_layers = [
            # (density, length, angle, alpha, blur)
            dict(density=(180,320), length=(8,14),  angle=(-12,12), alpha=(0.14,0.22), blur=1.0),
            dict(density=(280,520), length=(12,18), angle=(-16,16), alpha=(0.16,0.28), blur=1.2),
            dict(density=(380,700), length=(14,24), angle=(-18,18), alpha=(0.18,0.32), blur=1.4),
            dict(density=(480,900), length=(16,26), angle=(-20,20), alpha=(0.20,0.36), blur=1.6),
            dict(density=(600,1100),length=(18,30), angle=(-22,22), alpha=(0.22,0.40), blur=1.8),
        ][s-1]
        self.with_drops = with_drops

    def _color_grade(self, img):
        from PIL import ImageEnhance
        # 채도/대비/밝기
        img = ImageEnhance.Color(img).enhance(self.sat)
        img = ImageEnhance.Contrast(img).enhance(self.cts)
        img = ImageEnhance.Brightness(img).enhance(self.brt)
        # 쿨톤(Blue↑, Red↓)
        x = np.asarray(img, np.float32) / 255.0
        x[..., 0] *= (1.0 - self.cool)     # R down
        x[..., 2] *= (1.0 + self.cool)     # B up
        return Image.fromarray((np.clip(x,0,1)*255).astype(np.uint8))

    def __call__(self, img):
        # 1) 색보정
        img = self._color_grade(img)
        # 2) 약한 헤이즈로 원경/하늘 톤 죽이기
        img = AddHazeTV(**self.haze)(img)
        # 3) 빗줄기 2층(강층+약층) 합성
        strong = AddRainStreaksTV(**self.rain_layers)
        soft   = AddRainStreaksTV(
            density=(self.rain_layers["density"][0]//2, self.rain_layers["density"][1]//2),
            length=(self.rain_layers["length"][0]-2,  self.rain_layers["length"][1]-2),
            angle=self.rain_layers["angle"], alpha=(0.08, 0.16), blur=self.rain_layers["blur"]*0.9
        )
        img = strong(img)
        img = soft(img)
        # 4) 선택: 상단 위주 작은 방울
        if self.with_drops:
            h = img.size[1]
            top = img.crop((0, 0, img.size[0], int(h*0.6)))
            top = AddRaindropsTV(num=(10,30), radius=(5,18), alpha=(0.22,0.38), blur=(1.5,2.5))(top)
            img.paste(top, (0,0))
        return img


# ---------- severity → 파라미터 매핑 ----------
def build_transform(aug: str, severity: int):
    s = max(1, min(5, severity))
    if aug == "haze":
        presets = [
            dict(beta=(0.4,0.8),  A=(0.85,0.92), blur_ratio=0.03),
            dict(beta=(0.6,1.2),  A=(0.86,0.94), blur_ratio=0.04),
            dict(beta=(0.8,1.6),  A=(0.88,0.96), blur_ratio=0.05),
            dict(beta=(1.1,2.0),  A=(0.90,0.98), blur_ratio=0.06),
            dict(beta=(1.4,2.4),  A=(0.92,0.99), blur_ratio=0.07),
        ][s-1]
        return AddHazeTV(**presets)
    elif aug == "rain":
        presets = [
            # s=1 (기존 s=5 보다 살짝 ↑)
            dict(density=(700, 1100), length=(18, 30), angle=(-20, 20),
                 alpha=(0.28, 0.40), blur=1.6, width=(2, 3), color=(235, 235, 235)),
            # s=2
            dict(density=(1000, 1500), length=(20, 34), angle=(-22, 22),
                 alpha=(0.30, 0.45), blur=1.7, width=(2, 3), color=(235, 235, 235)),
            # s=3
            dict(density=(1300, 1900), length=(22, 36), angle=(-24, 24),
                 alpha=(0.32, 0.48), blur=1.8, width=(2, 4), color=(238, 238, 238)),
            # s=4
            dict(density=(1600, 2300), length=(24, 38), angle=(-26, 26),
                 alpha=(0.34, 0.50), blur=1.9, width=(3, 4), color=(240, 240, 240)),
            # s=5 (매우 강함)
            dict(density=(2000, 3000), length=(26, 42), angle=(-28, 28),
                 alpha=(0.36, 0.55), blur=2.0, width=(3, 5), color=(242, 242, 242)),
        ][s - 1]
        return AddRainStreaksTV(**presets)
    elif aug == "raindrop":
        presets = [
            dict(num=(10,25), radius=(5,12),  alpha=(0.20,0.35), blur=(1.0,2.0)),
            dict(num=(15,35), radius=(6,16),  alpha=(0.25,0.40), blur=(1.5,2.5)),
            dict(num=(20,50), radius=(7,20),  alpha=(0.28,0.45), blur=(1.8,3.0)),
            dict(num=(30,80), radius=(8,24),  alpha=(0.32,0.50), blur=(2.0,3.2)),
            dict(num=(50,120),radius=(10,30), alpha=(0.35,0.55), blur=(2.2,3.5)),
        ][s-1]
        return AddRaindropsTV(**presets)
    elif aug == "lowlight":
        presets = [
            dict(sigmas=(6, 24, 48), weights=None, levels=1, alpha=0.45, clip_percentiles=(2, 98)),
            dict(sigmas=(8, 32, 64), weights=None, levels=1, alpha=0.55, clip_percentiles=(2, 98)),
            dict(sigmas=(10, 40, 80), weights=None, levels=1, alpha=0.60, clip_percentiles=(1, 99)),
            dict(sigmas=(12, 48, 96), weights=None, levels=2, alpha=0.65, clip_percentiles=(1, 99)),
            dict(sigmas=(14, 56, 112), weights=None, levels=2, alpha=0.70, clip_percentiles=(1, 99)),
        ][s - 1]
        return FastRetinexLowLightTV(**presets)
    elif aug == "rainy":
        return RainyLookTV(severity=s, with_drops=True)
    elif aug == "overbright":  # ← 추가
        presets = [
            dict(gain=1.15, gamma=0.95, sat=1.02, rolloff=0.40, contrast=1.00),
            dict(gain=1.25, gamma=0.90, sat=1.04, rolloff=0.50, contrast=1.00),
            dict(gain=1.40, gamma=0.85, sat=1.06, rolloff=0.60, contrast=1.00),
            dict(gain=1.60, gamma=0.80, sat=1.08, rolloff=0.80, contrast=0.98),
            dict(gain=1.85, gamma=0.75, sat=1.10, rolloff=1.00, contrast=0.96),
        ][s - 1]
        return OverBrightTV(**presets)
    elif aug == "overbright_ev":  # ← 새 클래스 추가
        presets = [
            dict(ev=0.6, shoulder=1.0, gamma=0.92, lift=0.03,
                 bilateral_sc=25, bilateral_ss=5, detail_gain=0.95, chroma_gain=1.03),
            dict(ev=0.9, shoulder=1.2, gamma=0.88, lift=0.05,
                 bilateral_sc=30, bilateral_ss=6, detail_gain=0.92, chroma_gain=1.05),
            dict(ev=1.2, shoulder=1.5, gamma=0.85, lift=0.07,
                 bilateral_sc=35, bilateral_ss=7, detail_gain=0.90, chroma_gain=1.06),
            dict(ev=1.6, shoulder=1.8, gamma=0.82, lift=0.09,
                 bilateral_sc=40, bilateral_ss=8, detail_gain=0.86, chroma_gain=1.08),
            dict(ev=2.0, shoulder=2.2, gamma=0.80, lift=0.12,
                 bilateral_sc=45, bilateral_ss=9, detail_gain=0.82, chroma_gain=1.10),
        ][s - 1]
        return OverBrightExposureTV(**presets)
    elif aug == "overbright_curve":  # 새 방식
        presets = [
            dict(clip=1.6, tile=8, s=0.18, m=0.08, sat=1.02, sharpen=0.00),
            dict(clip=1.8, tile=8, s=0.22, m=0.10, sat=1.03, sharpen=0.05),
            dict(clip=2.0, tile=8, s=0.26, m=0.12, sat=1.04, sharpen=0.08),
            dict(clip=2.2, tile=8, s=0.32, m=0.14, sat=1.05, sharpen=0.10),
            dict(clip=2.4, tile=10, s=0.38, m=0.16, sat=1.06, sharpen=0.12),
        ][s - 1]
        return OverBrightCurveTV(**presets)
    elif aug == "overbright_fusion":
        # 새 s=1~5 (야간처럼 극저조도도 커버)
        presets = [
            dict(evs=(0.0, 2.5, 4.0), gammas=(1.00, 0.85, 0.70),
                 sat=1.03, contrast_w=0.45, satur_w=0.25, expose_w=0.9,
                 bright_bias=0.06, post_gamma=0.90, stretch=(0.3, 99.7), floor=0.02, unsharp=0.04),
            dict(evs=(0.0, 3.0, 5.0), gammas=(1.00, 0.80, 0.66),
                 sat=1.04, contrast_w=0.50, satur_w=0.30, expose_w=1.0,
                 bright_bias=0.08, post_gamma=0.85, stretch=(0.2, 99.8), floor=0.03, unsharp=0.06),
            dict(evs=(0.0, 3.5, 5.5), gammas=(1.00, 0.75, 0.60),
                 sat=1.06, contrast_w=0.55, satur_w=0.35, expose_w=1.0,
                 bright_bias=0.10, post_gamma=0.80, stretch=(0.1, 99.9), floor=0.04, unsharp=0.08),
            dict(evs=(0.0, 4.0, 6.0), gammas=(1.00, 0.72, 0.58),
                 sat=1.08, contrast_w=0.60, satur_w=0.40, expose_w=1.0,
                 bright_bias=0.12, post_gamma=0.78, stretch=(0.1, 99.9), floor=0.05, unsharp=0.09),
            dict(evs=(0.0, 4.5, 7.0), gammas=(1.00, 0.70, 0.55),
                 sat=1.10, contrast_w=0.60, satur_w=0.45, expose_w=1.0,
                 bright_bias=0.16, post_gamma=0.75, stretch=(0.1, 99.9), floor=0.06, unsharp=0.10),
        ][s - 1]
        return OverBrightFusionTV(**presets)
    elif aug == "overbright_lime":
        presets = [
            dict(gamma_t=0.90, ksize=11, sigma=2.0, gain=1.00, floor=0.02, stretch=(0.3, 99.7), sat=1.03, contrast=1.00,
                 unsharp=0.00),
            dict(gamma_t=0.82, ksize=13, sigma=2.5, gain=1.05, floor=0.03, stretch=(0.25, 99.75), sat=1.04,
                 contrast=1.00, unsharp=0.03),
            dict(gamma_t=0.75, ksize=15, sigma=3.0, gain=1.10, floor=0.04, stretch=(0.2, 99.8), sat=1.05, contrast=1.00,
                 unsharp=0.06),
            dict(gamma_t=0.70, ksize=17, sigma=3.5, gain=1.15, floor=0.05, stretch=(0.15, 99.85), sat=1.06,
                 contrast=0.98, unsharp=0.08),
            dict(gamma_t=0.65, ksize=19, sigma=4.0, gain=1.20, floor=0.06, stretch=(0.10, 99.90), sat=1.06,
                 contrast=0.96, unsharp=0.10),
        ][s - 1]
        return OverBrightLIMETV(**presets)
    else:
        raise ValueError(f"Unknown aug: {aug}")

# ---------- 실행부 ----------
def main():
    parser = argparse.ArgumentParser(description="Single-image weather/illumination augmentation saver")
    parser.add_argument("--input", default="./dataset/aug_test/clear.jpg", help="입력 이미지 경로")
    parser.add_argument("--output_dir", default="./aug_out/clear1", help="결과 저장 폴더")
    parser.add_argument("--aug", default="raindrop:5+rain:2+haze:2")  # "haze","rain","raindrop","rainy","lowlight","all"
    parser.add_argument("--severity", type=int, default=2, help="default 강도 1~5")
    parser.add_argument("--num", type=int, default=1, help="각 증강당 생성 개수")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    img = Image.open(args.input).convert("RGB")
    stem = os.path.splitext(os.path.basename(args.input))[0]
    ensure_dir(args.output_dir)

    aug_plans = parse_aug_argument(args.aug, args.severity)  # ← 변경됨

    for plan in aug_plans:
        for i in range(args.num):
            out_img = img.copy()
            # plan: [("haze",2), ("rain",5)] 같은 튜플 리스트
            for (name, sev) in plan:
                t = build_transform(name, sev)
                out_img = t(out_img)

            suffix = "_".join(f"{name}s{sev}" for (name, sev) in plan)
            out_path = os.path.join(
                args.output_dir, f"{stem}_{suffix}_{i + 1}.jpg"
            )
            out_img.save(out_path, quality=95)
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
