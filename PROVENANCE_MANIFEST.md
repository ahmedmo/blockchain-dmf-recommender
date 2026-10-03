# Experimental Provenance Manifest

This manifest maps the numerical tables in the manuscript to the finalized code and the experimental artifacts included in this repository package.

| Manuscript table | Reported content | Primary provenance |
|---|---|---|
| Table 1 | Social-relationship availability summary for voting stage | `03_authorized_node_selection/code/authorized_node_selection_douban.ipynb`; `03_authorized_node_selection/code/authorized_node_selection_lastfm.ipynb` |
| Table 2 | Dataset-level summary of announcement and finalization results | `04_announcement_confirmation_finalization/{dataset}/announce_ak_report.csv`; `confirm_ak_report.csv`; `finalized_ak_report.csv` |
| Table 3 | Representative sample of finalized announcement outcomes | `04_announcement_confirmation_finalization/{dataset}/finalized_ak_report.csv`; `ak_chain_announcements.csv` |
| Table 4 | Tampering test for changing the announced trusted authorized node | Finalized announcement/finalization notebooks in `04_announcement_confirmation_finalization/code/`; recorded Douban validation/test report where applicable |
| Table 5 | Required confirmations: community-level and network-wide reference | Authorized-node selection and announcement/finalization regeneration notebooks in `03_authorized_node_selection/code/` and `04_announcement_confirmation_finalization/code/` |
| Table 6 | Aggregate announcement and confirmation transactions and gas consumption | `05_blockchain_performance/{dataset}/blockchain_performance_transactions.csv`; `blockchain_performance_by_stage.csv`; `run_metadata.json` |
| Table 7 | Measured time to finalization and serial transaction rate | `05_blockchain_performance/{dataset}/blockchain_performance_transactions.csv`; `blockchain_performance_by_stage.csv`; `run_metadata.json` |
| Table 8 | Experimental configurations for the evaluation scenarios | Six finalized multi-seed notebooks under `02_recommendation/code/` |
| Tables 9–10 | Last.fm recommendation results | Finalized Last.fm notebooks under `02_recommendation/code/lastfm/`; CC-DMF/DC-DMF raw per-user files under `02_recommendation/outputs/lastfm/`; `02_recommendation/RAW_OUTPUT_MANIFEST.csv` |
| Tables 11–12 | Douban Movie recommendation results | Finalized Douban notebooks under `02_recommendation/code/douban/`; CC-DMF/DC-DMF raw per-user files under `02_recommendation/outputs/douban/`; `02_recommendation/RAW_OUTPUT_MANIFEST.csv` |
| Table 13 | Dataset-level recommendation delivery and hash verification | `06_recommendation_delivery/douban/08_onchain_delivery_records_top5_v1.csv`; `09_verification_results_top5_v1.csv`; `06_recommendation_delivery/lastfm/07_recommendation_hash_storage_report.csv`; `08_validate_stored_hashes_report.csv` |
| Table 14 | Representative community-level delivery records | Dataset-specific delivery/storage reports under `06_recommendation_delivery/` |
| Table 15 | Hash-based recommendation-package modification-detection results | Finalized delivery/verification notebooks under `06_recommendation_delivery/code/` and verification outputs under `06_recommendation_delivery/{dataset}/` |

## Reviewer Comment 10 coverage

The repository package directly covers the requested provenance categories:

- **Blockchain contracts:** `04_announcement_confirmation_finalization/contracts/AuthorizedNodeElection.sol` and dataset-specific `RecommendationDelivery.sol` files under `06_recommendation_delivery/contracts/`.
- **Deployment scripts/procedure:** finalized notebooks in `04_announcement_confirmation_finalization/code/`, `05_blockchain_performance/code/`, and `06_recommendation_delivery/code/` contain the contract source/compile/deploy/execution procedure used in the experiments.
- **Ganache configuration:** `07_environment_and_configuration/ganache_douban.json`, `ganache_lastfm.json`, and the original `run_metadata.json` files.
- **Registered-address mappings:** `04_announcement_confirmation_finalization/{dataset}/address_to_userid.csv` and associated mapping reports.
- **Transaction receipts or equivalent machine-readable execution logs:** `05_blockchain_performance/{dataset}/blockchain_performance_transactions.csv` together with the deployment and stage reports.
- **Raw per-run outputs or regeneration scripts supporting reported numerical tables:** finalized notebooks for all three recommendation scenarios; raw multi-seed CC-DMF/DC-DMF per-user outputs organized under `02_recommendation/outputs/`; blockchain and delivery execution outputs under stages 04–06.

## Seeds and repeated-run reporting

Recommendation training seeds: `42`, `52`, `62`.

Split seed: `42`.

Evaluation seed: `42`.

The raw output manifest records these fields for each bundled CC-DMF/DC-DMF per-user output. The DMF-Whole network notebooks serve as the supplied regeneration scripts for that scenario.
