"""Picture cards for the deck: each picture cropped to its slot, rounded corners baked onto the
slide's background colour (JPEG, so the deck stays small), plus the cover and closing backgrounds.
Sizes are in inches at 200 dpi and match the slots in build_deck.js.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
IMG, CARDS = ROOT / "img", ROOT / "img" / "cards"
DPI = 200
WHITE, NAVY, LIGHT = (255, 255, 255), (11, 34, 48), (238, 243, 246)


def cover_crop(im, w, h):
    """Crop to the w:h aspect around the centre, then resize to w x h pixels."""
    src = im.width / im.height
    dst = w / h
    if src > dst:
        nw = int(im.height * dst)
        im = im.crop(((im.width - nw) // 2, 0, (im.width + nw) // 2, im.height))
    else:
        nh = int(im.width / dst)
        im = im.crop((0, (im.height - nh) // 2, im.width, (im.height + nh) // 2))
    return im.resize((w, h), Image.LANCZOS)


def card(src, out, w_in, h_in, bg, radius_in=0.12, dpi=DPI):
    w, h = int(w_in * dpi), int(h_in * dpi)
    im = cover_crop(Image.open(IMG / src).convert("RGB"), w, h)
    scale = 4  # draw the mask large and shrink it for smooth corners
    mask = Image.new("L", (w * scale, h * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w * scale - 1, h * scale - 1),
                                           radius=int(radius_in * dpi * scale), fill=255)
    mask = mask.resize((w, h), Image.LANCZOS)
    Image.composite(im, Image.new("RGB", (w, h), bg), mask).save(CARDS / out, quality=90)


def fade_left(im, end=0.80, strength=0.94, power=1.0):
    w, h = im.size
    ramp = np.clip((end - np.linspace(0, 1, w)) / end, 0, 1) ** power * strength
    alpha = Image.fromarray((np.tile(ramp, (h, 1)) * 255).astype("uint8"))
    return Image.composite(Image.new("RGB", im.size, NAVY), im, alpha)


def main():
    CARDS.mkdir(parents=True, exist_ok=True)
    hero = Image.open(IMG / "hero.jpg").convert("RGB")
    fade_left(hero).save(CARDS / "cover.jpg", quality=88)
    Image.blend(hero, Image.new("RGB", hero.size, NAVY), 0.82).save(CARDS / "closing.jpg", quality=85)
    card("problem.jpg", "problem.jpg", 4.8, 5.3, WHITE)
    card("swot_satellite.jpg", "swot_satellite.jpg", 6.9, 3.88, NAVY)
    for name in ("hurulu_full", "hurulu_cloud", "hurulu_radar"):
        card(f"{name}.jpg", f"{name}_w.jpg", 3.85, 4.06, WHITE)
    for name in ("hurulu_low", "hurulu_mid", "hurulu_full"):
        card(f"{name}.jpg", f"{name}_n.jpg", 3.85, 4.06, NAVY)
    for name, src in (("thumb_s2", "hurulu_full.jpg"), ("thumb_s1", "hurulu_radar.jpg"),
                      ("thumb_water", "hurulu_water.png")):
        card(src, f"{name}.jpg", 1.08, 1.14, LIGHT, radius_in=0.07, dpi=240)
    print(sorted(p.name for p in CARDS.iterdir()))


if __name__ == "__main__":
    main()
