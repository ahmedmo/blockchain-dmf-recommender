# Upload Instructions

This folder is prepared as an additive update to the existing public repository `ahmedmo/blockchain-dmf-recommender`.

## Replace at repository root

- Replace the existing `README.md` with this package's `README.md`.
- Replace `requirements.txt` with this package's `requirements.txt`.
- Add `PROVENANCE_MANIFEST.md`.

## Add new stage directories

Upload directories `01_data_and_communities/` through `07_environment_and_configuration/` at the repository root.

The existing repository `data/` directory should remain in place. The new `01_data_and_communities/README.md` points to those already-uploaded community files, so they do not need to be duplicated.

## Do not upload

- Trained `.pt` checkpoints.
- Private keys or Ganache mnemonic.
- Duplicate files named `Copy`.
- Temporary/cache files.
- Large individual payload directories unless the editor explicitly requests them; the selected machine-readable delivery and verification reports and regeneration notebooks are already included.

## Final verification

After upload, open `PROVENANCE_MANIFEST.md` in GitHub and verify that every referenced path resolves. Then confirm that the repository root README no longer states that all generated outputs are excluded from version control.
