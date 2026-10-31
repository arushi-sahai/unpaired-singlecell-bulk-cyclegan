"""Prepare Phase 0's two descriptive figures for the results page, from their saved vector PDFs.

    python scripts/make_public_gap_figure.py

- assets/gap.png (Phase 0 figure 0.11): panels (b) and (c) recolored in grays, and the legend entry "domain" relabeled "assay".
- assets/granulocytes.png (Phase 0 figure 0.10): the "lineage controls" label moved down off the divider line.

Phase 0 saved no table behind these figures, and recomputing them would read every sample, including the test donors.
This script edits only color operators, one text string, and one text position in the saved PDFs and renders the
result, so the geometry and every number are exactly Phase 0's. results/phase0 is not modified.
"""
import re
import subprocess
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "phase0" / "figures"

INK_2 = r"0\.3215686275\s+0\.3176470588\s+0\.3058823529"      # dark gray in the source
MUTED = r"0\.5411764706\s+0\.537254902\s+0\.5176470588"         # mid gray in the source
DARK, LIGHT = "0.204 0.200 0.184", "0.541 0.537 0.518"          # dark gray: own pair and assay; mid gray: same donor
GRAY, LIGHT_GRAY = "0.765 0.761 0.741", "0.784 0.780 0.761"      # light gray: other donors; age group
# Grays, not hues: on the page, purple means composition labels and blue/orange mean the assays.

EDITS = [  # (pattern, replacement, expected count): panel (b) bars and legend, then panel (c) curves and legend
    (rf"/A1\s+gs\s+{INK_2}\s+rg\s+0\s+j\s+0\s+w\s+0\s+G\s+{INK_2}\s+rg", f"/A1 gs {DARK} rg 0 j 0 w 0 G {DARK} rg", 11),
    (rf"/A1\s+gs\s+{MUTED}\s+rg\s+0\s+j\s+0\s+w\s+0\s+G\s+{MUTED}\s+rg", f"/A1 gs {LIGHT_GRAY} rg 0 j 0 w 0 G {LIGHT_GRAY} rg", 11),
    (rf"0\s+j\s+1\.2\s+w\s+{MUTED}\s+RG", f"0 j 1.2 w {GRAY} RG", 2),
    (rf"0\s+j\s+1\.2\s+w\s+\[\s*4\.44\s+1\.92\s*\]\s+0\s+d\s+{INK_2}\s+RG", f"0 j 1.2 w [ 4.44 1.92 ] 0 d {LIGHT} RG", 2),
    (r"0\s+j\s+1\.2\s+w\s+/DeviceRGB\s+cs", f"0 j 1.2 w {DARK} RG /DeviceRGB cs", 2),
    (r"\(\x00d\x00o\x00m\x00a\x00i\x00n\)", "(\x00a\x00s\x00s\x00a\x00y)", 1),
]
GRAN_EDITS = [  # move the "lineage controls" label (baseline y 72.07) below the divider line at y 74.84
    (r"1 0 -0 1 174\.9416 72\.0697263991 cm", "1 0 -0 1 174.9416 67.6 cm", 1),
]


def render(src: Path, edits, out: Path):
    raw = src.read_bytes()
    target = None
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
        try:
            text = zlib.decompress(m.group(1)).decode("latin1")
        except zlib.error:
            continue
        if "BT" in text and (target is None or len(text) > len(target[1])):
            target = (m, text)
    m, text = target
    for pat, rep, n in edits:
        text, k = re.subn(pat, rep, text)
        assert k == n, (src.name, pat[:40], k, n)
    body = zlib.compress(text.encode("latin1"))
    new = raw[:m.start(1)] + body + raw[m.end(1):]
    head = new[:m.start(1)]
    lm = list(re.finditer(rb"/Length (\d+)", head))[-1]   # the stream's /Length; ghostscript rebuilds the xref table
    new = new[:lm.start(1)] + str(len(body)).encode() + new[lm.end(1):]
    with tempfile.TemporaryDirectory() as t:
        pdf = Path(t) / "fig.pdf"
        pdf.write_bytes(new)
        subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r300",
                        "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", f"-sOutputFile={out}", str(pdf)], check=True)
    print(out)


def main():
    render(FIGS / "0.11_domain_gap.pdf", EDITS, ROOT / "assets" / "gap.png")
    render(FIGS / "0.10_granulocyte_genes.pdf", GRAN_EDITS, ROOT / "assets" / "granulocytes.png")


if __name__ == "__main__":
    main()
