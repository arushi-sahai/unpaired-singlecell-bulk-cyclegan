"""Round the results page's numbers once, from full precision (run after Phase 8 writes assets/data.js).

    python scripts/fix_page_rounding.py          # report what would change
    python scripts/fix_page_rounding.py --apply  # rewrite assets/data.js

Phase 8 writes the page's numbers at 3 decimals (the report's precision), and the page shows 2. Rounding twice is
wrong only when the third decimal is 5 (0.105 can come from 0.1049 or 0.1052). Each such value is replaced by the
2-decimal rounding of its full-precision source in results/phase7/: table-2 cells from the matching row of
models_test.csv, and named numbers and table-1 estimates from the result column they report. Interval ends are
stored at 3 decimals in the result files, so they keep the page's own rounding. The Table 1 footnote is reworded
for the page (FOOT). Reads saved result files only.
"""
import argparse, json, re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JS, R7 = ROOT / "assets/data.js", ROOT / "results/phase7"
FIVE = re.compile(r"\d\.\d\d5$")
SB, BS = "single-cell → bulk", "bulk → single-cell"


FOOT = [("independent of the composition networks; accuracy is centered R²", "independent of the composition networks. Accuracy is centered R²"),
        ("margins were +0.257 (single-cell → bulk) and −0.476 (bulk → single-cell) (point estimates).",
         "margins were +0.26 when predicting bulk and −0.48 when predicting single-cell profiles, as point estimates."),
        ("would not have been supported (realism not distinguishable; accuracy worse) in both directions.",
         "would not have been supported when predicting either assay, with no clear difference in realism and worse accuracy."),
        ("Settings compared on validation: unpaired CycleGAN: 12 settings × 3 seeds on validation; CycleGAN + composition labels: 9 settings × 3 seeds on validation.",
         "On validation, 12 settings × 3 seeds were compared for the unpaired CycleGAN and 9 × 3 for CycleGAN + composition labels. \"No claim\" marks a descriptive comparison that was not one of the planned questions."),
        ("partly fits the network it trained against, in both directions.", "partly fits the network it trained against, when predicting either assay.")]


def fmt(x: float, signed: bool) -> str:
    s = str(Decimal(repr(abs(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return ("+" if x >= 0 else "−") + s if signed else ("−" if x < 0 else "") + s


def num(s: str) -> float:
    return float(s.replace("−", "-").replace("+", "").rstrip("*†"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    text = JS.read_text(); D = json.loads(text[text.index("{"):text.rindex("}") + 1])
    h1, h3, bcf, models = (pd.read_csv(R7 / f) for f in ("h1_test.csv", "h3_test.csv", "bcf_test.csv", "models_test.csv"))
    h3v = lambda d, m, metric: h3[(h3.direction == d) & (h3["ridge + spread minus"] == m) & (h3.metric == metric)].difference.iloc[0]
    # named numbers whose third decimal is 5, each with the result value it reports
    SOURCE = {"HOneSBMargin": h1[h1.direction == SB].margin.iloc[0],
              "MOneBSRealism": h1[h1.direction == BS]["M1 realism"].iloc[0],
              "HThreeSBMOneAcc": h3v(SB, "M1", "centered R²"), "HThreeSBMTwoAcc": h3v(SB, "M2", "centered R²"),
              "HThreeBSMOneAcc": h3v(BS, "M1", "centered R²"), "HThreeBSMTwoComp": h3v(BS, "M2", "marker composition R²"),
              "BcfBSComp": bcf[(bcf.direction == BS) & (bcf.role == "primary") & (bcf.metric == "marker composition R²")].difference.iloc[0]}
    changes, left = [], []
    for k, v in D["numbers"].items():
        if FIVE.search(v.rstrip("*†")):
            if k in SOURCE:
                new = fmt(SOURCE[k], v.startswith(("+", "−")) and not v.startswith("−") or v.startswith("+"))
                if v.startswith("−"): new = fmt(SOURCE[k], False)
                D["numbers"][k] = new; changes.append((k, v, new))
            else:
                left.append((k, v))
    by_value = {round(x, 3): x for x in pd.concat([h1.margin, h3.difference, bcf.difference])}
    for r in D["table1"]:
        e = r["estimate"]
        if FIVE.search(e):
            x = by_value[round(num(e), 3)]; new = fmt(x, True); changes.append(("table1 " + r["comparison"][:40], e, new)); r["estimate"] = new
        if re.search(r"\d\.\d\d5\b", r["ci"]): left.append(("table1 CI " + r["comparison"][:40], r["ci"]))
    cols = {"r2": "r2_centered", "realism": "realism", "margin": "margin", "composition": "marker composition R²", "within": "within_top1", "nn": "nn_ratio"}
    direction = None
    for r in D["table2"]:
        direction = r["direction"] or direction
        cand = models[models.direction == direction]
        dist = (cand.r2_centered - num(r["r2"])).abs() + (cand.realism - num(r["realism"])).abs()
        m = cand.loc[[dist.idxmin()]]                                 # the row these rounded values came from (rerunnable)
        assert dist.min() < 0.011, (direction, r["model"], dist.min())
        for c, col in cols.items():
            v = r[c]
            if FIVE.search(v.rstrip("*†")):
                tail = v[len(v.rstrip("*†")):]
                new = fmt(m[col].iloc[0], v.startswith(("+", "−")) and c == "margin") + tail
                if c != "margin" and v.startswith("−"): new = fmt(m[col].iloc[0], False) + tail
                changes.append((f"table2 {direction} {r['model'][:30]} {c}", v, new)); r[c] = new
    for old, new in FOOT:
        if old in D["foot"]:
            D["foot"] = D["foot"].replace(old, new); changes.append(("foot", old[:30], new[:30]))
    for c in changes: print(f"{c[0]}: {c[1]} -> {c[2]}")
    print(f"{len(changes)} values rounded from full precision. Left as is (no full-precision source): {left}")
    if a.apply:
        out = json.dumps(D, ensure_ascii=False, indent=1).replace("What pairs buy:", "What pairs add:")   # page wording
        JS.write_text("window.DATA = " + out + ";\n")


if __name__ == "__main__":
    main()
