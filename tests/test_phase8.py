"""Known-answer checks for the report formatting (notes/phase8_spec.md): python -m tests.test_phase8"""
from dgm import report as rp


def test_rounding_matches_the_notebooks():
    for x in (-0.0895, 0.1052362, -0.372471, 0.0004, -0.00049, 0.5):
        assert rp.num(x, latex=False) == f"{x:.3f}".replace("-0.000", "0.000")
    assert rp.num(-0.089636, latex=False) == "-0.090" and rp.num(-0.0895, latex=False) == f"{-0.0895:.3f}"   # binary -0.0895 prints -0.089, as in the notebooks
    assert rp.num(0.105236, sign=True, latex=False) == "+0.105"
    assert rp.num(-0.0001, sign=True, latex=False) == "+0.000"
    assert rp.num(-0.372471) == "$-$0.372"


def test_ci_and_macros():
    assert rp.ci(-0.2593, 0.29, latex=False) == "-0.259 to 0.290"
    assert rp.macro_name("H1", "S->B", "margin") == "HOneSBMargin"
    tex = rp.macros({"HOneSBMargin": "+0.105", "AB": "1"})
    assert tex.splitlines()[1] == "\\newcommand{\\AB}{1}" and "\\newcommand{\\HOneSBMargin}{+0.105}" in tex


def test_latex_table():
    t = rp.latex_table(["a", "b"], [["1", "2"]])
    assert "\\toprule" in t and "1 & 2 \\\\" in t and t.strip().endswith("\\end{tabular}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
