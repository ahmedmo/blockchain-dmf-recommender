# Data layout

The community-level files supplied for the experiments are included in this repository under `data/<dataset>/communities/`.
The original Last.fm and Douban Movie source datasets are not redistributed here and must be obtained separately from their original/public sources before running the whole-network experiments.

## Last.fm

Whole-network input expected by the prepared code:

- `data/lastfm/user_artists_ratings_per_user_norm_1dec.txt`

### CC-DMF

Directory: `data/lastfm/communities/cc/`

The seven supplied CC-DMF files match the seven files referenced by the finalized CC-DMF notebook.

### DC-DMF

Directory: `data/lastfm/communities/dc/`

All 13 supplied DC-DMF files match the 13 community files referenced by the finalized DC-DMF notebook. No community substitution is required.

## Douban Movie

Place the whole-network Douban input files required by `DMF_Whole_Douban.ipynb` under `data/douban/`.

### CC-DMF

Directory: `data/douban/communities/cc/`

All six supplied files match the six files referenced by the finalized CC-DMF notebook.

### DC-DMF

Directory: `data/douban/communities/dc/`

All six supplied files match the six files referenced by the finalized DC-DMF notebook.

## Integrity manifest

See `COMMUNITY_FILES_MANIFEST.csv` for file names, sizes, and SHA-256 hashes of the supplied community files included in this repository.
