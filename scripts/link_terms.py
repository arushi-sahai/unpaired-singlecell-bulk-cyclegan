"""Link defined terms on the results page to their entries in Definitions, the way an encyclopedia does.

    python scripts/link_terms.py

Each entry in the Definitions section gets an anchor (def-...). Then, in every section a reader might land on (the
top box, the cards, each setup section, each finding, the takeaways, the details), the first use of each term
links to its entry, so a reader who skims and a reader who dives in both meet the link near where they are.
Headings, existing links, summaries, phrases marked nolink, the Definitions section itself, and scripts are left alone.
The script is idempotent: it removes its own links first, so it can be rerun after any edit.
"""
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "index.html"

# Definitions entries: anchor id <- the entry's bold label (start of the <li>)
ENTRIES = {
    "def-singlecell": "Single-cell RNA-seq", "def-bulk": "Bulk RNA-seq", "def-profile": "Profile:",
    "def-cohort": "Cohort:", "def-draw": "Donor, draw, sample:", "def-pair": "Pair:", "def-splits": "Splits:",
    "def-crossassay": "Cross-assay prediction:", "def-gan": "GAN:", "def-cyclegan": "CycleGAN",
    "def-complabels": "CycleGAN + composition labels:", "def-ridge": "Paired ridge regression:",
    "def-pergene": "Per-gene correction:", "def-alignments": "Simple unpaired alignments,", "def-meanshift": "Mean shift",
    "def-combat": "ComBat", "def-coral": "CORAL", "def-checkpoint": "Chosen checkpoint, averaged weights:",
    "def-accuracy": "Accuracy (centered R²):", "def-realism": "Realism:", "def-composition": "Composition:",
    "def-markers": "Marker genes", "def-frontier": "Unpaired accuracy–realism frontier, blends, and margin:",
    "def-top1": "Within-donor top-1:", "def-ci": "95% CI:", "def-random": "Random real profile:",
    "def-samedonor": "Another draw of the same donor:", "def-mlp": "Paired neural network:",
    "def-ot": "Optimal-transport retrieval:", "def-program": "Gene program:", "def-personlevel": "Person-level and visit-level:",
    "def-basedev": "Between-donor (baseline) and within-donor (visit-deviation) correlation:", "def-tracked": "Tracked:",
    "def-donoryear": "Donor-year:", "def-partial": "Partial correlation of the change from the per-gene correction:",
    "def-pbmc": "PBMCs:", "def-pcs": "Principal components:", "def-plasmablast": "Plasmablasts:",
    "def-granulocyte": "Granulocytes and neutrophils:", "def-interferon": "Interferon response:", "def-hla": "HLA:",
    "def-cmv": "CMV:", "def-deconv": "Deconvolution:", "def-training": "Training details:",
}
READOUT = "def-readout"   # the marker-gene readout sub-entry, found by its text

# Hover text: a self-contained definition per entry; the full entry in Definitions adds detail and context.
SHORT = {
    "def-singlecell": ("Single-cell RNA-seq", "Measures gene expression in individual cells. In this project, the cells are PBMCs (the lymphocytes, monocytes, and dendritic cells of blood), and each sample's cells are summed into one pseudobulk profile."),
    "def-bulk": ("Whole-blood bulk RNA-seq", "Measures a whole blood sample at once. Unlike PBMC preparations, whole blood still contains granulocytes such as neutrophils."),
    "def-profile": ("Profile", "One sample's expression over the same 5,000 genes, normalized the same way in both assays so the two are comparable."),
    "def-cohort": ("Cohort", "96 healthy adults from the Allen Institute for Immunology's Sound Life study, each with several blood draws over about two years."),
    "def-draw": ("Donor, draw, sample", "Each donor gave several blood draws, or visits, over about two years. Each draw is one sample."),
    "def-pair": ("Pair", "The single-cell and bulk profiles measured from the same blood sample."),
    "def-splits": ("Splits", "Donors are divided into training, validation (used to choose settings), and test (scored once at the end), with all of a donor's draws in one split."),
    "def-crossassay": ("Cross-assay prediction", "Predicting a sample's profile in one assay from its profile in the other."),
    "def-gan": ("GAN and adversarial training", "A generative adversarial network trains a generator against a discriminator. The generator produces outputs, and the discriminator learns to tell them from real data. The generator is rewarded when the discriminator is fooled, and this setup is called adversarial training."),
    "def-cyclegan": ("CycleGAN", "A generative adversarial network (GAN) with two generators, one per direction, trained on two unpaired collections. Mapping a profile to the other assay and back should return the original."),
    "def-complabels": ("Composition labels", "Cell-type fractions read by small networks trained on paired samples. A loss asks the CycleGAN's predictions to keep their sample's fractions, while the CycleGAN itself still trains without pairs. In the cross-fitted variant, each sample's label comes from a network that never saw its donor."),
    "def-ridge": ("Paired ridge regression", "A linear regression fit on paired training samples, the reference that uses the most paired data. Plain ridge regression skips the spread step, so its predictions stay near the average and look much less real. The version with the spread step spreads its outputs out from the average so they look more real."),
    "def-pergene": ("Per-gene correction", "The simple alignment each CycleGAN builds on and learns a correction to. It is mean shift when predicting bulk and ComBat when predicting single-cell."),
    "def-alignments": ("Simple alignments", "Corrections that move one assay's profiles toward the other's without pairs. Mean shift, ComBat, and CORAL each match one more property of the two assays' data."),
    "def-trainingmean": ("Training mean", "A baseline that predicts the same profile for every sample, the average profile of the assay being predicted."),
    "def-meanshift": ("Mean shift", "Adds each gene's average difference between the two assays to the input."),
    "def-combat": ("ComBat", "Matches each gene's average and spread between the two assays. It is a widely used batch correction in genomics."),
    "def-coral": ("CORAL", "Matches each gene's average and spread and the correlations between genes. It is a standard unpaired domain-adaptation method."),
    "def-checkpoint": ("Chosen checkpoint and averaged weights", "Two ways to fix a model's final weights: the saved state chosen on validation, or a running average of the weights over training. Locked models had their settings and checkpoints fixed on validation, before the test split was scored."),
    "def-accuracy": ("Accuracy (centered R²)", "How close a prediction is to its sample's true partner. 1 is perfect, and 0 is no better than predicting the average profile."),
    "def-realism": ("Realism", "How hard predictions are to tell apart from real profiles. A classifier tries to separate them, and realism is 1 when it cannot and 0 when it separates them perfectly."),
    "def-composition": ("Composition", "The fractions of six cell types among a sample's PBMCs, counted from its cells."),
    "def-readout": ("Composition R²", "How well the cell-type fractions read from a profile, by a fixed readout of 28 marker genes, match the fractions counted from the sample's cells."),
    "def-markers": ("Marker genes", "Genes expressed mainly in one cell type, so their levels track how much of that cell type a sample holds."),
    "def-frontier": ("Frontier and margin", "The frontier is the most realism that blends of simple methods reach at each accuracy. A model's margin is how far above that edge it lands, so a margin above 0 means more realistic than any blend of equal accuracy."),
    "def-top1": ("Within-donor top-1", "How often a prediction is closer to its own sample's partner than to the same donor's other draws. Chance is about 1 in 8."),
    "def-ci": ("95% CI", "A 95% confidence interval from resampling donors. “Clearly” means it excludes the reference value, and “no clear difference” means it includes it."),
    "def-random": ("Random real profile", "A real profile drawn at random for each input. It looks real but carries no information about the input."),
    "def-samedonor": ("Another draw of the same donor", "The same donor's real profile from a different blood draw, which is the right person but the wrong day."),
    "def-mlp": ("Paired neural network", "A small neural network fit on paired training samples, a second paired reference."),
    "def-ot": ("Optimal-transport retrieval", "An unpaired method that matches each input to training profiles of the other assay and returns one of them."),
    "def-program": ("Gene program and program score", "A gene program is a set of genes that rise and fall together for one biological reason, such as one cell type's markers. A sample's program score is the average of those genes, each standardized first."),
    "def-personlevel": ("Person-level and visit-level", "A program is person-level when most of its variation lies between donors, and visit-level when most lies between one donor's visits."),
    "def-basedev": ("Baseline and visit deviation", "A donor's baseline is their average score for a program across visits. A visit deviation is how far one draw's score sits from that baseline."),
    "def-tracked": ("Tracking and the chance level", "A method tracks a program when its predicted scores correlate with the true scores clearly better than random real profiles do. That random-profile level is the chance level, and Δr is the change in that correlation from the per-gene correction."),
    "def-donoryear": ("Donor-year", "One donor's Day 0 and Day 7 draws in one flu-vaccine season."),
    "def-pbmc": ("PBMCs", "Peripheral blood mononuclear cells: the lymphocytes, monocytes, and dendritic cells of blood, separated from granulocytes and red cells before profiling."),
    "def-pcs": ("Principal components", "The directions along which profiles vary most. The top few summarize the main differences between samples."),
    "def-plasmablast": ("Plasmablasts", "Short-lived, antibody-secreting B cells that surge in the blood about a week after vaccination."),
    "def-granulocyte": ("Granulocytes and neutrophils", "White blood cells with granules, mostly neutrophils. They make up over half of blood's white cells and are removed when PBMCs are isolated."),
    "def-interferon": ("Interferon response", "Genes switched on by interferons, signaling proteins released during viral infection and some vaccine responses, peaking within a day or two."),
    "def-hla": ("HLA", "The human leukocyte antigen genes, which present antigens to T cells and differ greatly between people. Their measured expression is a stable trait of the person, partly because reads from different alleles map differently."),
    "def-cmv": ("CMV", "Cytomegalovirus, a common lifelong infection that expands cytotoxic NK and T cells and leaves a lasting expression signature."),
    "def-deconv": ("Deconvolution", "Estimating cell-type fractions in a bulk profile by modeling it as a mix of reference profiles for each cell type."),
    "def-training": ("Training details", "How the CycleGANs were built and trained: the generator, discriminator, losses, optimizer, and the settings searched on validation."),
    "def-partial": ("Partial correlation of the change", "How much a method's change from the per-gene correction relates to the true partner, beyond the correction itself. 0 means the change carries nothing about the specific sample."),
}

# term pattern -> anchor; longer and more specific patterns first
TERMS = [
    (r"per-gene correction|gene-by-gene correction", "def-pergene"),
    (r"CycleGAN \+ composition labels|composition labels?", "def-complabels"),
    (r"cross-fitted", "def-complabels"),
    (r"paired ridge regression|ridge regression without the spread step|spread step", "def-ridge"),
    (r"composition R²", READOUT), (r"marker-gene readout", READOUT),
    (r"simple alignments?", "def-alignments"), (r"training mean", "def-trainingmean"),
    (r"mean shift", "def-meanshift"), (r"ComBat", "def-combat"), (r"CORAL", "def-coral"),
    (r"(?:unpaired )?accuracy–realism frontier|frontier", "def-frontier"), (r"blends?", "def-frontier"), (r"margins?", "def-frontier"),
    (r"centered R²", "def-accuracy"), (r"realism", "def-realism"),
    (r"random real profiles?", "def-random"), (r"chance level", "def-tracked"),
    (r"another draw of the same donor|another visit of the same donor", "def-samedonor"),
    (r"optimal-transport retrieval", "def-ot"),
    (r"gene programs?|program scores?", "def-program"),
    (r"person-level|visit-level", "def-personlevel"),
    (r"donors?' baselines?|donor baselines?|baseline r|visit deviations?", "def-basedev"),
    (r"tracks|tracked|tracking", "def-tracked"),
    (r"donor-years?", "def-donoryear"), (r"partial correlation", "def-partial"),
    (r"averaged weights|weights averaged over training", "def-checkpoint"),
    (r"pseudobulk", "def-singlecell"), (r"PBMCs?", "def-pbmc"), (r"whole-blood bulk", "def-bulk"),
    (r"picked out its own sample|finding its own sample", "def-top1"), (r"Δr", "def-tracked"),
    (r"cross-assay prediction", "def-crossassay"), (r"adversarial training|discriminator", "def-gan"),
    (r"cycle-consistency|identity loss", "def-cyclegan"), (r"Sound Life", "def-cohort"),
    (r"held-out donors|test donors|validation donors", "def-splits"),
    (r"realistic", "def-realism"), (r"mixtures? of two of them|best mixture", "def-frontier"),
    (r"checkpoints?|locked", "def-checkpoint"), (r"marker genes?", "def-markers"),
    (r"95% CIs?|confidence intervals?|donor-bootstrap", "def-ci"), (r"ridge regression", "def-ridge"),
    (r"principal components?", "def-pcs"), (r"plasmablasts?", "def-plasmablast"),
    (r"granulocytes?|neutrophils?", "def-granulocyte"), (r"interferon(?: response)?", "def-interferon"),
    (r"HLA", "def-hla"), (r"CMV|cytomegalovirus", "def-cmv"), (r"deconvolution|deconvolving", "def-deconv"),
    (r"CycleGANs?", "def-cyclegan"),
]
ONCE = {"def-cyclegan"}                                   # linked at first use on the page only
EXCLUDE = {'<h2 id="comparator"': {"def-frontier", "def-realism"},   # don't link a term inside the section that defines it
           '<dl class="terms"': {"def-program", "def-basedev", "def-tracked", "def-personlevel"}}
PATTERN = re.compile("|".join(f"(?P<t{i}>\\b(?:{p})(?![\\w-]))" for i, (p, _) in enumerate(TERMS)), re.I)
SKIP = {"a", "h1", "h2", "h3", "summary", "script", "style", "title", "button", "th", "svg", "code", "caption", "nolink"}
OWN = re.compile(r'<a class="term" href="#def-[\w-]+">(.*?)</a>', re.S)


def add_anchors(html: str) -> str:
    cut = html.index('<h2 id="definitions">')                # definition anchors live in the Definitions section only
    head, defs = html[:cut], html[cut:]
    for anchor, label in ENTRIES.items():
        defs = re.sub(r'<li>(?=<b>' + re.escape(label) + ')', f'<li id="{anchor}">', defs, count=1)
    html = head + defs
    html = html.replace("<li>The <b>marker-gene readout</b>", f'<li id="{READOUT}">The <b>marker-gene readout</b>', 1)
    html = re.sub(r' data-title="[^"]*" data-short="[^"]*"', "", html)
    for anchor, (title, short) in SHORT.items():
        esc = lambda t: t.replace("&", "&amp;").replace('"', "&quot;")
        html, n = re.subn(f'id="{anchor}"', f'id="{anchor}" data-title="{esc(title)}" data-short="{esc(short)}"', html, count=1)
        assert n, anchor
    return html


def regions(html: str):
    """(start, end) spans: one per section a reader might land on; Definitions and everything after References are excluded."""
    cuts = [m.start() for m in re.finditer(r'<p class="dataline|<h2 id="|<div class="lead">|<div class="grid3">|<h3 id="f\d"|<details class="finding"|<div class="takeaways">|'
                                            r'<details id="meaning-detail"|<dl class="terms"|<p class="fgroup', html)]
    end_main = html.index('<h2 id="definitions">')
    spans = []
    for a, b in zip(cuts, cuts[1:] + [len(html)]):
        if a >= end_main and not html.startswith('<h2 id="limitations"', a):
            continue
        spans.append((a, min(b, len(html))))
    return spans


def link(html: str) -> str:
    html = OWN.sub(r"\1", html)
    out, pos, page_seen = [], 0, set()
    for a, b in regions(html):
        out.append(html[pos:a])
        seg = html[a:b]
        blocked = set(page_seen) | next((v for k, v in EXCLUDE.items() if seg.startswith(k)), set())   # anchors not linked here
        seen = set()                                                                                    # terms already linked here
        stack = []
        for tok in re.split(r"(<[^>]+>)", seg):
            if tok.startswith("<"):
                m = re.match(r"<(/?)(\w+)", tok)
                if m and not tok.endswith("/>"):
                    name = m.group(2).lower()
                    if name == "span" and not m.group(1):
                        stack.append("nolink" if 'class="nolink"' in tok else "span")
                    elif name == "span":
                        if stack and stack[-1] in ("span", "nolink"):
                            stack.pop()
                    elif m.group(1):
                        if name in stack:
                            del stack[len(stack) - 1 - stack[::-1].index(name)]
                    elif name in SKIP:
                        stack.append(name)
                out.append(tok)
                continue
            if any(x not in ("span",) for x in stack) or not tok.strip():
                out.append(tok)
                continue

            def sub(m):
                key = next(k for k, v in m.groupdict().items() if v)
                anchor = TERMS[int(key[1:])][1]
                if anchor in blocked or key in seen:       # one link per term (not per entry) in each region
                    return m.group(0)
                seen.add(key)
                if anchor in ONCE:
                    page_seen.add(anchor)
                return f'<a class="term" href="#{anchor}">{m.group(0)}</a>'
            out.append(PATTERN.sub(sub, tok))
        pos = b
    out.append(html[pos:])
    return "".join(out)


# References: (first-author pattern, year, link to the paper). In-text citations link to their entry, whose hover shows the
# full reference and the paper link. Links are DOIs where they exist, otherwise the proceedings or arXiv page.
REFS = {
    "abbas2009": ("Abbas", 2009, "https://doi.org/10.1371/journal.pone.0006098"),
    "amodio2018": ("Amodio", 2018, "https://proceedings.mlr.press/v80/amodio18a.html"),
    "blau2018": ("Blau", 2018, "https://arxiv.org/abs/1711.06077"),
    "brown1991": ("Brown", 1991, "https://doi.org/10.1038/349038a0"),
    "chu2017": ("Chu", 2017, "https://arxiv.org/abs/1712.02950"),
    "chu2022": ("Chu", 2022, "https://doi.org/10.1038/s43018-022-00356-3"),
    "cuturi2013": ("Cuturi", 2013, "https://proceedings.neurips.cc/paper_files/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html"),
    "debortoli2021": ("De Bortoli", 2021, "https://proceedings.neurips.cc/paper_files/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html"),
    "demetci2022": ("Demetci", 2022, "https://doi.org/10.1089/cmb.2021.0446"),
    "gong2025": ("Gong", 2025, "https://doi.org/10.1038/s41586-025-09686-5"),
    "goodfellow2014": ("Goodfellow", 2014, "https://proceedings.neurips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html"),
    "ho2020": ("Ho", 2020, "https://arxiv.org/abs/2006.11239"),
    "johnson2007": ("Johnson", 2007, "https://doi.org/10.1093/biostatistics/kxj037"),
    "kingma2015": ("Kingma", 2015, "https://arxiv.org/abs/1412.6980"),
    "korsunsky2019": ("Korsunsky", 2019, "https://doi.org/10.1038/s41592-019-0619-0"),
    "liao2022": ("Liao", 2022, "https://doi.org/10.1038/s41467-022-34271-z"),
    "liberzon2015": ("Liberzon", 2015, "https://doi.org/10.1016/j.cels.2015.12.004"),
    "lipman2023": ("Lipman", 2023, "https://arxiv.org/abs/2210.02747"),
    "lopezpaz2017": ("Lopez-Paz", 2017, "https://arxiv.org/abs/1610.06545"),
    "luecken2022": ("Luecken", 2022, "https://doi.org/10.1038/s41592-021-01336-8"),
    "mao2017": ("Mao", 2017, "https://arxiv.org/abs/1611.04076"),
    "miyato2018": ("Miyato", 2018, "https://arxiv.org/abs/1802.05957"),
    "nakaya2011": ("Nakaya", 2011, "https://doi.org/10.1038/ni.2067"),
    "newman2015": ("Newman", 2015, "https://doi.org/10.1038/nmeth.3337"),
    "newman2019": ("Newman", 2019, "https://doi.org/10.1038/s41587-019-0114-2"),
    "provost2001": ("Provost", 2001, "https://doi.org/10.1023/A:1007601015854"),
    "sun2016": ("Sun", 2016, "https://doi.org/10.1609/aaai.v30i1.10306"),
    "wrammert2008": ("Wrammert", 2008, "https://doi.org/10.1038/nature06890"),
    "zhu2017": ("Zhu", 2017, "https://arxiv.org/abs/1703.10593"),
    "guma2004": ("Gumá", 2004, "https://doi.org/10.1182/blood-2004-05-2058"),
    "obermoser2013": ("Obermoser", 2013, "https://doi.org/10.1016/j.immuni.2012.12.008"),
    "sonderby2017": ("Sønderby", 2017, "https://arxiv.org/abs/1610.04490"),
    "yazici2019": ("Yazıcı", 2019, "https://openreview.net/forum?id=SJgw_sRqFQ"),
}
CITE = re.compile("|".join(rf"(?P<{k}>\b{re.escape(n)}(?: et al\.| and [A-Z][\w-]+)?, {y}\b)" for k, (n, y, _) in REFS.items()))
OWN_CITE = re.compile(r'<a class="cite" href="#ref-[\w-]+">(.*?)</a>', re.S)
OWN_PAPER = re.compile(r' <a class="paper" href="[^"]*">[^<]*</a>')


def split_reference(entry: str):
    """Authors and title of a reference entry written as 'Surname, I., Surname, I. &amp; Surname, I. Title. <i>Venue</i> ...'."""
    head = re.split(r"<i>| arXiv:", entry)[0]
    text = re.sub(r"<[^>]+>", "", head).replace("&amp;", "&")
    i = 0
    while True:
        j = text.index(". ", i)
        nxt = text[j + 2:]
        if re.match(r"[A-Z]\.|[A-Z]-[A-Z]\.|& |et al\.|[A-Z][\w'-]+(?: [A-Z][\w'-]+)?, (?:[A-Z]\.|[A-Z]-)", nxt):
            i = j + 2
            continue
        return text[:j + 1].strip(), nxt.strip().rstrip(".").strip()

def link_references(html: str) -> str:
    """Anchor each References entry, add its paper link, and link every in-text citation to its entry."""
    html = OWN_PAPER.sub("", OWN_CITE.sub(r"\1", html))
    a = html.index('<h2 id="references">'); b = html.index("</ul>", a)
    refs = html[a:b]
    for k, (n, y, url) in REFS.items():
        m = re.search(rf'<li(?: id="ref-[\w-]+"[^>]*)?>({re.escape(n)}, (?:(?!</li>).)*?\({y}\)\.)</li>', refs, re.S)   # one entry only
        assert m, k
        authors, title = split_reference(m.group(1))
        esc = lambda t: t.replace("&", "&amp;").replace('"', "&quot;")
        refs = refs[:m.start()] + (f'<li id="ref-{k}" data-title="{esc(title)}" data-authors="{esc(authors)}" data-url="{url}">'
                                   f'{m.group(1)} <a class="paper" href="{url}">Paper</a></li>') + refs[m.end():]
    html = html[:a] + refs + html[b:]
    start = html.index("<main"); end = html.index('<h2 id="references">')
    out, stack = [], []
    for tok in re.split(r"(<[^>]+>)", html[start:end]):
        if tok.startswith("<"):
            m = re.match(r"<(/?)(\w+)", tok)
            if m and not tok.endswith("/>") and m.group(2).lower() in {"a", "h1", "h2", "h3", "script", "style", "svg", "title", "summary"}:
                name = m.group(2).lower()
                if m.group(1):
                    if name in stack:
                        stack.remove(name)
                else:
                    stack.append(name)
            out.append(tok)
        elif stack:
            out.append(tok)
        else:
            out.append(CITE.sub(lambda m: f'<a class="cite" href="#ref-{m.lastgroup}">{m.group(0)}</a>', tok))
    return html[:start] + "".join(out) + html[end:]


XREF = re.compile(r"\bFigure (\d+)([a-z]?)(?=[;),])|\bFinding (\d)\b|\bFigures (\d+) and (\d+)\b")
OWN_XREF = re.compile(r'<a class="xref" href="#(?:fig|f)\d+">(.*?)</a>', re.S)


def link_xrefs(html: str) -> str:
    """Link in-text figure and finding references ("(Figure 7d)", "Finding 4's Details") to their targets; a link into a
    closed dropdown opens it. Caption labels ("Figure 7.") and existing links are left alone."""
    html = OWN_XREF.sub(r"\1", html)
    start = html.index("<main"); end = html.index("<script")
    out, stack = [], []
    for tok in re.split(r"(<[^>]+>)", html[start:end]):
        if tok.startswith("<"):
            m = re.match(r"<(/?)(\w+)", tok)
            if m and not tok.endswith("/>") and m.group(2).lower() in {"a", "h1", "h2", "h3", "summary", "script", "style", "svg", "title"}:
                name = m.group(2).lower()
                if m.group(1):
                    if name in stack:
                        stack.remove(name)
                else:
                    stack.append(name)
            out.append(tok)
        elif stack:
            out.append(tok)
        else:
            def sub(m):
                if m.group(1):
                    return f'<a class="xref" href="#fig{m.group(1)}">{m.group(0)}</a>'
                if m.group(3):
                    return f'<a class="xref" href="#f{m.group(3)}">{m.group(0)}</a>'
                return (f'Figures <a class="xref" href="#fig{m.group(4)}">{m.group(4)}</a> and '
                        f'<a class="xref" href="#fig{m.group(5)}">{m.group(5)}</a>')
            out.append(XREF.sub(sub, tok))
    return html[:start] + "".join(out) + html[end:]


def main():
    html = add_anchors(PAGE.read_text())
    html = link_references(html)
    html = link_xrefs(html)
    html = link(html)
    PAGE.write_text(html)
    print(len(re.findall(r'class="term"', html)), "term links,", len(re.findall(r'class="cite"', html)), "citation links,", len(re.findall(r'class="xref"', html)), "cross-references")


if __name__ == "__main__":
    main()
