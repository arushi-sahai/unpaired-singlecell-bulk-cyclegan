"""Write the results page's Phase 9 numbers (assets/data_bio.js); its Phase 9 figures come from make_public_figures.py.

    python scripts/make_bio_page_data.py

Every value is read from results/phase9/ (validation, post-hoc); nothing is recomputed.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / "results" / "phase9"
ASSETS = ROOT / "assets"
SB, BS = "single-cell → bulk", "bulk → single-cell"
GAN, RIDGE, AVG = "unpaired CycleGAN", "paired ridge regression", "unpaired CycleGAN (averaged)"


def f(x, nd=2):
    if 0 < abs(x) < 0.005:   # the page shows at most two decimals; a nonzero value that rounds to 0.00 says so
        return "<0.01" if x > 0 else ">−0.01"
    return f"{x:.{nd}f}".replace("-", "−")


def ci(s):
    lo, hi = (float(v) for v in s.split(" to "))
    return f"{f(lo)} to {f(hi)}"


def main():
    t = pd.read_csv(R9 / "program_tracking_val.csv")
    sh = pd.read_csv(R9 / "program_between_share.csv")
    vg = pd.read_csv(R9 / "vaccine_gate_train.csv")
    vr = pd.read_csv(R9 / "vaccine_response_val.csv")
    pt = pd.read_csv(R9 / "person_traits_val.csv")
    mv = pd.read_csv(R9 / "moves_all_genes_val.csv")
    row = lambda d, p, tr: t[(t.direction == d) & (t.program == p) & (t.translator == tr)].iloc[0]
    n = {}
    for key, d in (("SB", SB), ("BS", BS)):
        s, g = t[(t.direction == d) & (t.translator == "start")], t[(t.direction == d) & (t.translator == GAN)]
        n[f"StartTracked{key}"] = f"{int(s.tracked.sum())} of {len(s)}"
        n[f"Inherit{key}"] = f"{int((g.reading == 'inherited from the alignment').sum())} of {len(g)}"
        b = sh[sh.direction == d]
        n[f"PersonLevel{key}"] = f"{int((b.level == 'person').sum())} of {len(b)}"
    for key, d, p in (("IfnSB", SB, "interferon response"), ("PltSB", SB, "platelet"), ("MonoSB", SB, "Mono markers"),
                      ("PlasmaBS", BS, "plasmablast and immunoglobulin"), ("TBS", BS, "T markers")):
        r = row(d, p, GAN)
        n[key], n[key + "CI"] = f(r["Δr vs start"]), ci(r["Δr vs start CI"])
    n["IfnStartWithinSB"] = f(row(SB, "interferon response", "start")["within r"], 2)
    for key, p in (("EryGANSB", "erythroid"), ("RiboGANSB", "ribosomal")):
        r = row(SB, p, GAN)
        n[key], n[key + "CI"] = f(r["Δr vs start"]), ci(r["Δr vs start CI"])
    for key, tr in (("Start", "start"), ("GAN", GAN), ("Ridge", RIDGE)):
        r = row(SB, "granulocyte", tr)
        n[f"Gran{key}"], n[f"Gran{key}CI"] = f(r["r"]), ci(r["r CI"])
    r = row(SB, "granulocyte", RIDGE)
    n["GranNull"], n["GranRidgeDelta"], n["GranRidgeDeltaCI"] = f(r["null r hi"]), f(r["Δr vs start"]), ci(r["Δr vs start CI"])
    pb = "plasmablast and immunoglobulin"
    for key, a in (("Bulk", "bulk"), ("SC", "single-cell")):
        r = vg[(vg["target assay"] == a) & (vg.program == pb)].iloc[0]
        n[f"VacGate{key}"], n[f"VacGate{key}CI"] = f(r["training change"]), f"{f(r['CI low'])} to {f(r['CI high'])}"
    vrow = lambda d, tr: vr[(vr.direction == d) & (vr.program == pb) & (vr.translator == tr)].iloc[0]
    for key, d in (("SB", SB), ("BS", BS)):
        r = vrow(d, "start")
        n[f"VacTrue{key}"], n[f"VacTrue{key}CI"], n[f"VacNull{key}"] = f(r["true change"]), ci(r["true CI"]), f(r["null r 97.5%"])
        for k2, tr in (("Start", "start"), ("GAN", GAN), ("Ridge", RIDGE), ("Avg", AVG)):
            r = vrow(d, tr)
            n[f"Vac{k2}{key}"], n[f"Vac{k2}{key}CI"], n[f"Vac{k2}{key}Lo"] = f(r["r with true change"]), ci(r["r CI"]), f(r["r lo"])
    rs = vr[(vr.program == pb) & vr.translator.isin(["start", GAN, RIDGE])]["r with true change"]
    n["VacRLo"], n["VacRHi"] = f(rs.min(), 2), f(rs.max(), 2)
    lab = vr[(vr.program == pb) & (vr.direction == SB) & vr.translator.str.contains("labels")]["r with true change"]
    n["VacLabelRLo"], n["VacLabelRHi"] = f(lab.min(), 2), f(lab.max(), 2)
    auc = pt[(pt.direction == BS) & pt.translator.str.contains("CycleGAN")]["XIST donor AUROC"]
    n["XistAucGANLo"], n["XistAucGANHi"] = f(auc.min(), 2), f(auc.max(), 2)
    ratio = "XIST translated / true difference"
    for key, d in (("SB", SB), ("BS", BS)):
        for k2, tr in (("Start", "start"), ("GAN", GAN), ("Ridge", RIDGE)):
            n[f"Xist{k2}{key}"] = f(pt[(pt.direction == d) & (pt.translator == tr)][ratio].iloc[0], 2)
    hla = pt[pt.translator != "true target"][[c for c in pt.columns if c.startswith("HLA-")]].to_numpy()
    n["HlaLo"], n["HlaHi"] = f(hla.min(), 2), f(hla.max(), 2)
    for key, d in (("SB", SB), ("BS", BS)):
        m = mv[mv.direction == d].set_index("comparison")
        for k2, c in (("GAN", GAN), ("Coral", "CORAL (unpaired reference)"), ("Ridge", RIDGE),
                      ("GANvCoral", f"{GAN} − CORAL (unpaired reference)"), ("RidgevGAN", f"{RIDGE} − {GAN}")):
            n[f"Move{k2}{key}"], n[f"Move{k2}{key}CI"] = f(m.loc[c, "median partial r"]), ci(m.loc[c, "95% CI"])
    n["GanGran2"], n["GranNull2"], n["VacNull2"] = f(float(n["GranGAN"]), 2), f(float(n["GranNull"]), 2), f(float(n["VacNullBS"]), 2)
    (ASSETS / "data_bio.js").write_text("window.DATA_BIO = " + json.dumps(n, ensure_ascii=False, indent=1) + ";\n")
    for k, v in n.items():
        print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
