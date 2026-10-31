"""Draw the link-preview image for the results page (assets/og.png, 1200 × 630).

    python scripts/make_og_image.py

The page title above the test-setup figure (assets/test_setup.png).
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H, PAD = 1200, 630, 48
TITLE = ["What can a generative model recover about a", "person's immune cells without paired samples?"]
SUB = "Predicting between PBMC single-cell (pseudobulk) and whole-blood transcriptomes · Arushi Sahai"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF"):
        if (Path(d) / name).exists():
            return ImageFont.truetype(str(Path(d) / name), size)
    return ImageFont.load_default()


img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)
y = PAD - 8
for line in TITLE:
    draw.text((PAD, y), line, font=font(40, bold=True), fill="#1a1a19")
    y += 50
draw.text((PAD, y + 6), SUB, font=font(20), fill="#5f5e5a")
top = y + 50
fig = Image.open(ROOT / "assets/test_setup.png").convert("RGB")
fig.thumbnail((W - 2 * PAD, H - top - PAD // 2), Image.LANCZOS)
img.paste(fig, ((W - fig.width) // 2, top))
img.save(ROOT / "assets/og.png", optimize=True)
print("assets/og.png", img.size)
