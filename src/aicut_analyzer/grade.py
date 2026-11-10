"""Colour grades made as 3D LUTs, from looks described in numbers, and applied with ffmpeg.

A grade that reads as "expensive" does what a single curve cannot: it rolls highlights off instead of clipping them,
keeps skin where it is while moving other colours, and moves one hue (greens toward olive) without the rest. So a
look here works on each colour in OKLab, a space where hue and lightness are separate and even:
- a tone curve (`tone`): a gentle S around mid-grey, lifted blacks, and a soft shoulder for highlights;
- hue moves (`shifts`): for a band of hues, turn them and scale their chroma, with skin (`SKIN`) protected;
- split toning: a tint added to shadows and another to highlights, weighted by lightness;
- overall chroma, skin kept at its own.
`portra400` is a warm, soft day look after Kodak Portra 400; `vision500t` a night look after Kodak Vision3 500T
(teal shadows, warm practical lights). `finish` adds what film does on top: halation round bright lights, and grain.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- colour spaces

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


_BT2020_TO_709 = np.array([[1.6605, -0.5876, -0.0728],
                           [-0.1246, 1.1329, -0.0083],
                           [-0.0182, -0.1006, 1.1187]])


def hlg_to_sdr(rgb: np.ndarray, white: float = 0.9) -> np.ndarray:
    """HLG BT.2100 R'G'B' (0–1) to SDR sRGB values: inverse OETF, the system gamma (1.2), HLG reference white
    (signal 0.75, 203 nits) put at SDR white, highlights above it rolled off instead of clipped, BT.2020 → BT.709."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    e = np.clip(rgb, 0, 1)
    scene = np.where(e <= 0.5, e * e / 3, (np.exp((e - c) / a) + b) / 12)
    y = scene @ np.array([0.2627, 0.6780, 0.0593])
    display = scene * np.clip(y, 1e-6, None)[..., None] ** 0.2             # OOTF, gamma 1.2
    ref = (np.exp((white - c) / a) + b) / 12
    ref = ref ** 1.2
    x = display / ref                                                    # 1 = reference white
    knee = 0.65
    x = np.where(x > knee, knee + (1 - knee) * np.tanh((x - knee) / (1 - knee)), x)   # shoulder, never above 1
    x = np.clip(x @ _BT2020_TO_709.T, 0, 1)
    y709 = (x @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
    x = y709 + 0.8 * (x - y709)          # the glasses' HLG comes out hotter than the phone's SDR: bring colour in line
    return linear_to_srgb(np.clip(x, 0, 1))


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = srgb_to_linear(rgb) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lms = (lab @ np.linalg.inv(_M2).T) ** 3
    return linear_to_srgb(lms @ np.linalg.inv(_M1).T)


# ---------------------------------------------------------------- looks

@dataclass(frozen=True)
class Shift:
    hue: float          # centre of the band, degrees (OKLab hue)
    width: float        # degrees either side where it fades out
    turn: float = 0.0   # degrees to turn those hues
    chroma: float = 1.0


SKIN = Shift(hue=55.0, width=28.0)      # OKLab hue of skin, light to dark, across complexions


@dataclass(frozen=True)
class Look:
    name: str
    contrast: float = 1.0           # S-curve strength around mid-grey (1: none)
    lift: float = 1.0               # midtone brightness as a gamma (>1 brighter, ends stay put): the airy look
    black: float = 0.0              # lifted black level
    shoulder: float = 1.0           # where highlights start to roll off (1: they never do)
    chroma: float = 1.0             # overall saturation, skin excepted
    skin_chroma: float = 1.0
    shifts: tuple[Shift, ...] = ()
    tint: tuple[float, float] = (0.0, 0.0)              # added to OKLab (a, b) everywhere: the white balance
    shadow_tint: tuple[float, float] = (0.0, 0.0)       # added to OKLab (a, b) in the shadows
    highlight_tint: tuple[float, float] = (0.0, 0.0)    # and in the highlights
    halation: float = 0.0           # strength of the glow round bright lights
    grain: float = 0.0              # grain strength (ffmpeg noise, 0–20)
    sharpen: float = 0.0            # luma unsharp mask amount (0: none, ~0.6 crisp)
    vignette: float = 0.0           # edge darkening as ffmpeg vignette angle (0: none, ~0.5 subtle)


def portra400() -> Look:
    """Kodak Portra 400: warm clean skin, soft contrast, creamy highlights, olive greens, slightly muted colour."""
    return Look("portra400", contrast=0.86, black=0.05, shoulder=0.66, chroma=0.88, skin_chroma=1.04,
                shifts=(Shift(140, 50, turn=-28, chroma=0.68),       # greens toward olive, quieter
                        Shift(235, 45, turn=-18, chroma=0.8),        # blues toward cyan
                        Shift(25, 18, turn=8, chroma=0.95)),         # reds toward orange
                tint=(0.003, 0.012), shadow_tint=(0.004, 0.008), highlight_tint=(0.006, 0.028),
                halation=0.32, grain=7)


def vision500t() -> Look:
    """Kodak Vision3 500T at night: teal shadows, warm orange practical lights, firm but not crushed blacks."""
    return Look("vision500t", contrast=1.1, black=0.035, shoulder=0.66, chroma=0.82, skin_chroma=0.98,
                shifts=(Shift(60, 30, turn=-8, chroma=1.25),         # sodium and tungsten lights, warmer
                        Shift(140, 50, turn=40, chroma=0.75),        # greens toward teal
                        Shift(235, 50, turn=-25, chroma=1.05)),      # blues toward teal
                shadow_tint=(-0.032, -0.036), highlight_tint=(0.02, 0.05), halation=0.5, grain=10)


def classic_chrome() -> Look:
    """Fujifilm Classic Chrome: muted documentary colour, deep but soft shadows, reds toward brick, blues toward
    slate, skin kept natural."""
    return Look("classic_chrome", contrast=1.12, black=0.03, shoulder=0.7, chroma=0.64, skin_chroma=0.86,
                shifts=(Shift(25, 20, turn=-6, chroma=0.85),         # reds deeper, toward brick
                        Shift(95, 30, turn=6, chroma=0.75),          # yellows quieter
                        Shift(140, 50, turn=-12, chroma=0.7),        # greens toward olive
                        Shift(235, 45, turn=12, chroma=0.75)),       # blues toward slate
                shadow_tint=(-0.018, -0.014), highlight_tint=(0.006, 0.016), halation=0.12, grain=6)


def classic_neg() -> Look:
    """Fujifilm Classic Negative (Superia): hard shadows with a cyan-green cast, warm magenta-leaning highlights,
    yellowish greens, reds rich, overall colour held back. The street-snap look."""
    return Look("classic_neg", contrast=1.2, black=0.025, shoulder=0.68, chroma=0.7, skin_chroma=0.88,
                shifts=(Shift(25, 20, turn=-4, chroma=1.08),         # reds rich
                        Shift(140, 50, turn=-32, chroma=0.7),        # greens toward yellow-olive
                        Shift(235, 45, turn=-20, chroma=0.85)),      # blues toward cyan
                shadow_tint=(-0.034, -0.008), highlight_tint=(0.022, 0.022), halation=0.18, grain=8)


def japan_airy() -> Look:
    """Japanese airy look (Fuji Pro 400H, 日系空气感): bright, low contrast, lifted milky blacks, minty greens,
    cyan sky, cool shadows, clean pink-leaning skin, little saturation."""
    return Look("japan_airy", lift=1.22, contrast=0.8, black=0.07, shoulder=0.72, chroma=0.7, skin_chroma=0.92,
                shifts=(Shift(140, 50, turn=22, chroma=0.75),        # greens toward mint
                        Shift(235, 45, turn=-15, chroma=0.85),       # blues toward cyan
                        Shift(95, 30, turn=-10, chroma=0.7),         # yellows tamed
                        Shift(25, 18, turn=-6, chroma=0.8)),         # reds toward pink
                tint=(-0.004, -0.006), shadow_tint=(-0.012, -0.012), highlight_tint=(0.002, 0.004),
                halation=0.15, grain=5)


def classic_neg_cinema(brightness: float) -> Look:
    """Classic Negative made filmic, adapted to one recording's brightness (its mean sRGB luma, 0–1), so a whole
    film comes out alike: bright stretches are brought down and given harder contrast, dark ones left nearly alone.
    Highlights roll off early, edges darken a little, the picture is sharpened, halation and grain on top."""
    base = classic_neg()
    target = 0.40
    raw = np.log(max(brightness, 0.05)) / np.log(target)              # the gamma that would hit the target
    lift = float(np.clip(1 + 0.65 * (raw - 1), 0.74, 1.1))
    hard = float(np.clip((brightness - 0.35) / 0.25, 0, 1))           # 0 for dim scenes, 1 for bright ones
    from dataclasses import replace
    return replace(base, name=f"classic_neg_cinema_{brightness:.2f}", lift=lift, contrast=1.2 + 0.28 * hard,
                   shoulder=0.62, black=0.02, sharpen=0.7, vignette=0.45, grain=0)


def tone(x: np.ndarray, look: Look) -> np.ndarray:
    """The look's curve on values 0–1: an S of strength `contrast` round 0.45, a shoulder from `shoulder` up that
    reaches 1 smoothly, and blacks lifted to `black`."""
    p = 0.45
    x = x ** (1 / look.lift)
    y = np.where(x < p, p * (x / p) ** look.contrast, 1 - (1 - p) * ((1 - x) / (1 - p)) ** look.contrast)
    h = look.shoulder
    if h < 1:
        y = np.where(y > h, h + (1 - h) * np.tanh((y - h) / (1 - h) * 1.25) / np.tanh(1.25), y)
    return look.black + (1 - look.black) * y


def band(hue: np.ndarray, s: Shift) -> np.ndarray:
    """How much each hue is in a band: 1 at its centre, fading (smoothly) to 0 at `width` away."""
    d = np.abs((hue - s.hue + 180) % 360 - 180)
    w = np.clip(1 - d / s.width, 0, 1)
    return w * w * (3 - 2 * w)


def grade(rgb: np.ndarray, look: Look) -> np.ndarray:
    """sRGB values 0–1 (…, 3) graded by the look."""
    rgb = tone(np.clip(rgb, 0, 1), look)
    lab = srgb_to_oklab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360
    skin = band(h, SKIN) * np.clip(C / 0.03, 0, 1)        # greys are not skin
    turn = np.zeros_like(h)
    scale = np.ones_like(h)
    for s in look.shifts:
        w = band(h, s) * (1 - skin)
        turn += w * s.turn
        scale *= 1 + w * (s.chroma - 1)
    scale *= look.chroma + skin * (look.skin_chroma - look.chroma)
    h, C = h + turn, C * scale
    a, b = C * np.cos(np.radians(h)), C * np.sin(np.radians(h))
    shadow = np.clip(1 - L / 0.55, 0, 1) ** 1.5
    high = np.clip((L - 0.55) / 0.45, 0, 1) ** 1.5
    a = a + look.tint[0] + shadow * look.shadow_tint[0] + high * look.highlight_tint[0]
    b = b + look.tint[1] + shadow * look.shadow_tint[1] + high * look.highlight_tint[1]
    return np.clip(oklab_to_srgb(np.stack([L, a, b], axis=-1)), 0, 1)


# ---------------------------------------------------------------- LUTs and ffmpeg

def lut(look: Look, size: int = 33, source: str = "sdr") -> np.ndarray:
    """The look as a 3D table (size³ × 3), red changing fastest, as .cube files order it. `source` "hlg" takes HLG
    HDR input (e.g. Ray-Ban Meta glasses) and converts it to SDR first."""
    x = np.linspace(0, 1, size)
    b, g, r = np.meshgrid(x, x, x, indexing="ij")
    rgb = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    if source == "hlg":
        rgb = hlg_to_sdr(rgb)
    return grade(rgb, look)


def write_cube(path: Path, look: Look, size: int = 33, source: str = "sdr") -> Path:
    table = lut(look, size, source)
    lines = [f'TITLE "{look.name}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    lines += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in table]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def finish(cube: Path, look: Look, height: int, matrix: str | None = None) -> str:
    """An ffmpeg filter: the LUT, then halation (bright parts blurred, tinted red-orange, screened back on), a
    vignette, sharpening and luma grain, scaled to the picture's long side (`height`) so they look alike at 1080p
    and 4K. `matrix` names the source's YUV matrix (bt709, bt2020) so it is read into RGB exactly; the result is
    BT.709."""
    k = height / 1920
    chain = (f"scale=in_color_matrix={matrix}:in_range=tv,format=gbrp," if matrix else "")
    chain += f"lut3d=file='{cube}':interp=tetrahedral,format=gbrp"
    if look.halation:
        chain += (f",split[base][hl];[hl]colorlevels=rimin=0.78:gimin=0.78:bimin=0.78,gblur=sigma={18 * k:.1f},"
                  f"colorchannelmixer=rr=1:gg=0.38:bb=0.12[glow];[base][glow]blend=all_mode=screen:"
                  f"all_opacity={look.halation}")
    if look.vignette:
        chain += f",vignette=angle={look.vignette}"
    chain += (",scale=out_color_matrix=bt709:out_range=tv,format=yuv420p" if matrix else ",format=yuv420p")
    if look.sharpen:
        chain += f",unsharp=5:5:{look.sharpen}:5:5:0"
    if look.grain:
        chain += f",noise=c0s={look.grain}:c0f=t+u"
    return chain


def apply(src: Path, dst: Path, look: Look, cube: Path, start: float | None = None,
          length: float | None = None) -> None:
    """The recording (or a stretch of it) graded into a new file, sound copied."""
    size = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                           "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.split()[0]
    height = max(int(v) for v in size.split(",") if v)     # the long side: a phone's portrait clip is stored on its side
    cut = (["-ss", str(start)] if start is not None else []) + (["-t", str(length)] if length is not None else [])
    subprocess.run(["ffmpeg", "-v", "error", "-y", *cut, "-i", str(src), "-filter_complex", finish(cube, look, height),
                    "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-c:a", "copy", str(dst)], check=True)
