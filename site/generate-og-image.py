#!/usr/bin/env python3
"""Regenerate brand social cards from the synthpanel brand palette.

Run from repo root: ``python3 site/generate-og-image.py``.

Produces two dark-slate PNGs with an emerald radial glow and the wordmark +
tagline, matching the synthpanel.dev visual identity:

* ``site/og-image.png`` (1200x630) — embedded in landing/mcp/blog pages via
  ``og:image`` / ``twitter:image`` meta tags.
* ``site/github-social-preview.png`` (1280x640) — dimensioned for GitHub's
  repo Settings → Social preview (upload is manual, no API exists).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

BG = (15, 23, 42)  # slate-900  #0f172a
FG_HI = (248, 250, 252)  # slate-50
FG_LO = (148, 163, 184)  # slate-400
EMERALD = (52, 211, 153)  # emerald-400  #34d399
EMERALD_DIM = (16, 185, 129)  # emerald-500

MONO_CANDIDATES = [
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
]
SANS_CANDIDATES = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def load_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def radial_glow(size: tuple[int, int], color: tuple[int, int, int], strength: float) -> Image.Image:
    w, h = size
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    cx, cy = w // 2, -int(h * 0.15)
    max_r = int(h * 1.1)
    steps = 60
    for i in range(steps, 0, -1):
        t = i / steps
        alpha = int(strength * 255 * (t**2.2))
        r = int(max_r * (1 - t) + 40)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, alpha))
    return glow.filter(ImageFilter.GaussianBlur(60))


def draw_pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    x, y = xy
    pad_x, pad_y = 20, 10
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    w = tw + pad_x * 2
    h = th + pad_y * 2 + 4
    r = h // 2
    box = (x, y, x + w, y + h)
    draw.rounded_rectangle(box, radius=r, outline=(*fill, 120), width=2, fill=(*fill, 25))
    draw.ellipse((x + pad_x - 4, y + h // 2 - 3, x + pad_x + 2, y + h // 2 + 3), fill=fill)
    draw.text((x + pad_x + 12, y + pad_y - bbox[1]), text, font=font, fill=fill)
    return box


def render(width: int, height: int, out: Path) -> None:
    img = Image.new("RGB", (width, height), BG)

    glow = radial_glow((width, height), EMERALD, strength=0.18)
    img.paste(glow, (0, 0), glow)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    draw.rectangle((0, height - 4, width, height), fill=(*EMERALD_DIM, 160))

    font_badge = load_font(SANS_CANDIDATES, 22)
    font_title = load_font(MONO_CANDIDATES, 148)
    font_tag = load_font(SANS_CANDIDATES, 44)
    font_foot = load_font(MONO_CANDIDATES, 28)
    font_meta = load_font(SANS_CANDIDATES, 24)

    margin_x = 88
    y = 92
    draw_pill(draw, (margin_x, y), "v0.9.1  public beta", font_badge, EMERALD)

    title = "synthpanel"
    title_y = 176
    tb = draw.textbbox((0, 0), title, font=font_title)
    draw.text((margin_x - tb[0], title_y - tb[1]), title, font=font_title, fill=FG_HI)
    title_h = tb[3] - tb[1]

    tagline = "Run synthetic focus groups with any LLM."
    tag_y = title_y + title_h + 36
    draw.text((margin_x, tag_y), tagline, font=font_tag, fill=FG_LO)

    meta = "Personas in YAML  ·  Claude · GPT · Gemini · Grok  ·  CLI + MCP server"
    meta_y = tag_y + 72
    draw.text((margin_x, meta_y), meta, font=font_meta, fill=(100, 116, 139))

    foot = "$ pip install synthpanel"
    foot_bbox = draw.textbbox((0, 0), foot, font=font_foot)
    foot_w = foot_bbox[2] - foot_bbox[0]
    foot_h = foot_bbox[3] - foot_bbox[1]
    foot_pad_x, foot_pad_y = 22, 14
    fx = margin_x
    fy = height - 88 - foot_h - foot_pad_y * 2
    draw.rounded_rectangle(
        (fx, fy, fx + foot_w + foot_pad_x * 2, fy + foot_h + foot_pad_y * 2 + 6),
        radius=10,
        outline=(51, 65, 85, 255),
        width=2,
        fill=(15, 23, 42, 230),
    )
    draw.text((fx + foot_pad_x, fy + foot_pad_y - foot_bbox[1]), "$", font=font_foot, fill=(100, 116, 139))
    prefix_bbox = draw.textbbox((0, 0), "$ ", font=font_foot)
    prefix_w = prefix_bbox[2] - prefix_bbox[0]
    draw.text(
        (fx + foot_pad_x + prefix_w, fy + foot_pad_y - foot_bbox[1]),
        "pip install synthpanel",
        font=font_foot,
        fill=EMERALD,
    )

    brand = "synthpanel.dev"
    brand_bbox = draw.textbbox((0, 0), brand, font=font_meta)
    brand_w = brand_bbox[2] - brand_bbox[0]
    draw.text(
        (width - margin_x - brand_w, height - 88 - (brand_bbox[3] - brand_bbox[1])),
        brand,
        font=font_meta,
        fill=FG_LO,
    )

    img.paste(overlay, (0, 0), overlay)
    img.save(out, "PNG", optimize=True)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


def main() -> int:
    site_dir = Path(__file__).resolve().parent
    render(1200, 630, site_dir / "og-image.png")
    render(1280, 640, site_dir / "github-social-preview.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
