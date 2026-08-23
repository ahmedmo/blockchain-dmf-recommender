# Blockchain-Based Decentralized Deep Matrix Factorization Recommender System

This repository contains the Deep Matrix Factorization (DMF) recommendation code used to evaluate three controlled recommendation settings on the Last.fm and Douban Movie datasets:

- **DMF-Whole network** — DMF trained/evaluated on the whole eligible network.
- **CC-DMF** — community-level DMF using the centralized/real community partition used by the experiments.
- **DC-DMF** — community-level DMF using the decentralized/discovered community partition used by the experiments.

The repository contains the six finalized experiment notebooks (Whole, CC-DMF, and DC-DMF for each dataset), Python-script exports, and the supplied CC/DC community-level files.

## Repository structure

```text
src/
  lastfm/
    dmf_whole_lastfm.py
    cc_dmf_lastfm.py
    dc_dmf_lastfm.py
  douban/
    dmf_whole_douban.py
    cc_dmf_douban.py
    dc_dmf_douban.py
notebooks/
  lastfm/
    DMF_Whole_LastFM.ipynb
    CC_DMF_LastFM.ipynb
    DC_DMF_LastFM.ipynb
  douban/
    DMF_Whole_Douban.ipynb
    CC_DMF_Douban.ipynb
    DC_DMF_Douban.ipynb
data/
  lastfm/communities/cc/
  lastfm/communities/dc/
  douban/communities/cc/
  douban/communities/dc/
  COMMUNITY_FILES_MANIFEST.csv
  README.md
outputs/
requirements.txt
.gitignore
```

## Evaluation organization

The finalized notebooks contain the small-K sampled ranking evaluation and the large-K ranking over the eligible item space where implemented by the corresponding experiment. These are treated as distinct evaluation protocols rather than as a single continuous K-range.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

The same prepared code is exported as Python scripts under `src/`. Then start Jupyter with:

```bash
jupyter notebook
```

## Data and community files

The supplied CC/DC community-level files are included. The original source datasets are not redistributed in this repository. See [`data/README.md`](data/README.md) for the expected source-data layout and the community-file audit.

All local working paths such as `D:\PHD\...` were replaced in this repository copy by relative `data/` and `outputs/` paths; the experimental logic was not intentionally changed by this path cleanup.

### Community-file audit

All supplied community sets now match the finalized experiment code completely:

- Douban CC-DMF: 6/6 files matched.
- Douban DC-DMF: 6/6 files matched.
- Last.fm CC-DMF: 7/7 files matched.
- Last.fm DC-DMF: 13/13 files matched.

The finalized Last.fm DC-DMF set includes `comm_U_25_I_787_104.txt`, exactly as referenced by the finalized notebook and Python export. No alternate community file is substituted.

## Reproducibility note

The repository copy retains the random seeds and experimental configuration encoded in the selected finalized notebooks. Generated outputs and trained checkpoints are excluded from version control. SHA-256 hashes for the included community files are recorded in `data/COMMUNITY_FILES_MANIFEST.csv`.

## Citation

If this code is used in academic work, please cite the associated manuscript:

**A Blockchain-Based Decentralized Deep Matrix Factorization Recommender System.**

A formal publication citation can be added once the bibliographic details are finalized.
