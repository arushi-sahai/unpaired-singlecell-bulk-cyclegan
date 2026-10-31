"""Rebuild the eight SoundLife single-cell "slim" files from the AIFI portal files.

The single-cell input of this project is one slim file per portal file (age group x sex x
CMV status): a uniform random 3,000 cells per sample (all cells of a sample with fewer),
drawn before any QC, with raw counts. The kept cell barcodes of each slim are in
scripts/slim_barcodes/. This script selects exactly those barcodes from the portal files and
checks each result's content checksum against the pinned one, so anyone with the portal files
reproduces the slims exactly (selecting by barcode keeps the result independent of the
random-number library's version).

How the barcodes were drawn: per portal file, numpy.random.default_rng(42 + int(md5(<file
stem>)[:8], 16)); per sample in order of first appearance, 3,000 cells chosen without
replacement where the sample had more, all cells otherwise; rows kept in file order.

Data: AIFI Sound Life (Gong et al. 2025; doi:10.57785/bn17-3h16; CC BY 4.0). The barcode
lists are derived from those data. Adapted from an earlier, unpublished pipeline by the same author.

    python scripts/make_slims.py <directory with the 8 portal files> [<output directory>]

The output directory defaults to data/raw/soundlife/ (what notebooks/phase0_preprocessing.ipynb reads).
"""
import argparse
import gzip
import hashlib
import os
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
BARCODES = REPO / "scripts" / "slim_barcodes"
PORTAL_FILES = tuple(f"SoundLife_{age}Adult_{sex}_CMV{cmv}.h5ad"
                     for age in ("Young", "Older") for sex in ("Male", "Female") for cmv in ("pos", "neg"))
SAMPLE_COLUMNS = ("sample_id", "donor_id", "numeric", "age_group", "sex", "cmv", "timepoint", "batch_id")
CELL_TYPES = ("AIFI_L1", "AIFI_L2", "AIFI_L3")
SLIM_OBS = SAMPLE_COLUMNS + CELL_TYPES + ("n_cells_full",)
PORTAL_COLUMNS = {"sample.sampleKitGuid": "sample_id", "subject.subjectGuid": "donor_id",   # portal column -> slim column
                  "sample.subjectAgeAtDraw": "numeric", "subject.ageGroup": "age_group",
                  "subject.biologicalSex": "sex", "subject.cmv": "cmv", "sample.visitName": "timepoint",
                  "batch_id": "batch_id", "AIFI_L1": "AIFI_L1", "AIFI_L2": "AIFI_L2", "AIFI_L3": "AIFI_L3"}
CONTENT_SHA256 = {   # content checksum of each slim (see content_checksum)
    "SoundLife_YoungAdult_Male_CMVpos__slim.h5ad": "52717e0628c4e6432dcfbcf39ad946c212de6d8935a6078bd9dbcbe14ba2b400",
    "SoundLife_YoungAdult_Male_CMVneg__slim.h5ad": "ef36ee767bf4aba07fea7956038659b0d2d1506b9db094a4b0c89be266c18ddf",
    "SoundLife_YoungAdult_Female_CMVpos__slim.h5ad": "be9d8a921f3fb6fd8bda2794e0f9734fce8dca5a3d74c91ded2ab8f0f5396f0a",
    "SoundLife_YoungAdult_Female_CMVneg__slim.h5ad": "1461acffd25cd7aacce1e97333e066077b79f4081e434e0bc0ffe42f8f71f215",
    "SoundLife_OlderAdult_Male_CMVpos__slim.h5ad": "31c3810241836a81fb3347cb101127bfe91cc9d7e1a307dff3bf775164a748eb",
    "SoundLife_OlderAdult_Male_CMVneg__slim.h5ad": "f285d69694b6b1b429dd2b03f1a511b22f02ee6dd77e1f057ad3e1666bf55915",
    "SoundLife_OlderAdult_Female_CMVpos__slim.h5ad": "26829620626600f56f5a03933558e08c4a2483dda6f3bb33d2006bb3ece4ebd8",
    "SoundLife_OlderAdult_Female_CMVneg__slim.h5ad": "2e1f1902bf70cdda37407480eb8d2e51db143618319c3a5f6e85d2d25cc386cb",
}


def slim_name(portal_name):
    return portal_name.removesuffix(".h5ad") + "__slim.h5ad"


def slim_barcodes(portal_name):
    with gzip.open(BARCODES / (portal_name.removesuffix(".h5ad") + ".txt.gz"), "rt") as f:
        return [line.rstrip("\n") for line in f]


def content_checksum(adata) -> str:
    """SHA-256 of a slim's content, independent of the h5ad library version: the shape, the
    canonical CSR counts (indices sorted, duplicates summed, zeros removed; data uint16,
    indices int32, indptr int64, little-endian), the cell barcodes and gene names, and the
    obs columns in a fixed order, as strings. Each section is prefixed by its length."""
    X = sp.csr_matrix(adata.X)
    assert np.array_equal(X.data, np.round(X.data)) and X.data.min() >= 0 and X.data.max() <= np.iinfo(np.uint16).max, \
        "counts must be integers in the uint16 range"
    X = X.astype(np.uint16)
    X.sum_duplicates()
    X.eliminate_zeros()
    X.sort_indices()
    sections = [np.asarray(X.shape, dtype="<i8").tobytes(), X.data.astype("<u2").tobytes(),
                X.indices.astype("<i4").tobytes(), X.indptr.astype("<i8").tobytes(),
                "\n".join(adata.obs_names.astype(str)).encode(), "\n".join(adata.var_names.astype(str)).encode(),
                *["\n".join(adata.obs[column].astype(str)).encode() for column in SLIM_OBS]]
    digest = hashlib.sha256()
    for section in sections:
        digest.update(len(section).to_bytes(8, "little"))
        digest.update(section)
    return digest.hexdigest()


def build_slim(portal_path, barcodes):
    """The slim of one portal file: the listed cells (in the listed order) with their raw counts
    as uint16, the slim obs columns, and the gene symbols as given."""
    portal = anndata.read_h5ad(portal_path, backed="r")
    assert portal.var_names.is_unique, f"{portal_path}: gene names are not unique"
    rows = portal.obs_names.get_indexer(barcodes)
    assert (rows >= 0).all(), f"{portal_path}: some listed barcodes are not in the file"
    sub = portal[np.sort(rows)].to_memory()
    sub = sub[pd.Index(sub.obs_names).get_indexer(barcodes)]
    n_cells_full = pd.Series(portal.obs["sample.sampleKitGuid"].astype(str).to_numpy()).value_counts()
    portal.file.close()
    obs = pd.DataFrame({slim: sub.obs[col].to_numpy() for col, slim in PORTAL_COLUMNS.items()}, index=sub.obs_names.astype(str))
    obs["numeric"] = pd.to_numeric(obs["numeric"], errors="raise").astype(float)
    obs["n_cells_full"] = obs["sample_id"].astype(str).map(n_cells_full).astype(int).to_numpy()
    for column in SAMPLE_COLUMNS + CELL_TYPES:
        if column != "numeric":
            obs[column] = obs[column].astype(str)
    X = sp.csr_matrix(sub.X)
    assert X.max() <= np.iinfo(np.uint16).max, f"{portal_path}: a count exceeds uint16"
    return anndata.AnnData(X=X.astype(np.uint16), obs=obs, var=pd.DataFrame(index=sub.var_names.astype(str)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("portal_directory", type=Path)
    parser.add_argument("output_directory", type=Path, nargs="?", default=REPO / "data" / "raw" / "soundlife")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for portal in PORTAL_FILES:
        slim = build_slim(args.portal_directory / portal, slim_barcodes(portal))
        checksum = content_checksum(slim)
        expected = CONTENT_SHA256[slim_name(portal)]
        if checksum != expected:
            sys.exit(f"{portal}: rebuilt slim content {checksum} does not match the pinned {expected}")
        path = args.output_directory / slim_name(portal)
        partial = path.with_name(".tmp." + path.name)
        slim.write_h5ad(partial)
        os.replace(partial, path)   # replaces a symlink rather than writing through it
        print(f"{slim_name(portal)}: {slim.n_obs} cells, content verified", flush=True)


if __name__ == "__main__":
    main()
