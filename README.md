# Blockchain-Based Decentralized Deep Matrix Factorization Recommender System

This repository package is organized according to the experimental workflow reported in the manuscript. It provides the finalized experiment code, selected raw recommendation outputs, blockchain contracts, deployment/execution artifacts, registered-address mappings, and machine-readable blockchain logs needed to trace the reported numerical results.

## Repository organization

1. `01_data_and_communities/` — points to the community files already stored under the repository `data/` directory.
2. `02_recommendation/` — DMF-Whole network, CC-DMF, and DC-DMF experiment code for Douban Movie and Last.fm, plus raw per-user multi-seed outputs for CC-DMF and DC-DMF.
3. `03_authorized_node_selection/` — finalized trusted authorized-node selection code using the community-level social-distance procedure.
4. `04_announcement_confirmation_finalization/` — `AuthorizedNodeElection` smart contract, deployed addresses, registered-address mappings, and announcement/confirmation/finalization execution reports.
5. `05_blockchain_performance/` — transaction-level machine-readable logs, stage summaries, community summaries, and run metadata used for blockchain performance reporting.
6. `06_recommendation_delivery/` — `RecommendationDelivery` contracts, deployment records, on-chain registration/delivery records, recommendation-hash storage, and verification outputs.
7. `07_environment_and_configuration/` — Ganache/Solidity configuration extracted from the recorded experiment metadata.

See `PROVENANCE_MANIFEST.md` for the mapping between manuscript tables and repository artifacts.

## Recommendation repeated-run evaluation

The principal recommendation experiments were repeated using training seeds `42`, `52`, and `62`. Within each evaluation configuration, the split seed and evaluation seed were fixed at `42`, and the model architecture, loss, hyperparameters, and best-model selection procedure were unchanged.

Raw per-user outputs are organized by:

`dataset / scenario / training seed / community`

The included `RAW_OUTPUT_MANIFEST.csv` records the dataset, scenario, seed values, community tag, original source filename, and repository path for every bundled raw output file.

For DMF-Whole network, the finalized multi-seed notebooks are provided as regeneration scripts. For CC-DMF and DC-DMF, both the finalized notebooks and the raw per-user outputs are provided.

## Blockchain provenance

The finalized blockchain notebooks include the Solidity contract source, compilation, Ganache connection, deployment, transaction submission, receipt handling, and output generation used in the experiments.

The repository also includes the resulting artifacts needed for direct inspection:

- Solidity contract sources and ABIs.
- Deployed contract addresses.
- Registered blockchain-address mappings.
- Announcement, confirmation, and finalization reports.
- Transaction-level machine-readable execution logs with transaction hashes and receipt-derived fields.
- Blockchain performance summaries and run metadata.
- Recommendation-delivery records and hash-verification outputs.

The common `AuthorizedNodeElection.sol` source is identical for the Douban Movie and Last.fm announcement/finalization experiments. The recommendation-delivery contracts are retained separately for the two dataset workflows because the recorded source files differ.

## Environment

Core Python dependencies are listed in `requirements.txt`. The blockchain experiments use Web3 and `py-solc-x`, with Solidity compiler `0.8.20`. Recorded Ganache settings are provided under `07_environment_and_configuration/`.

No private keys, Ganache mnemonic, or trained model checkpoints are included.

## Experimental provenance note

Selected raw experimental outputs, machine-readable blockchain execution logs, deployment records, and regeneration code supporting the reported numerical results are included. Trained model checkpoints are intentionally omitted because they are not required to reproduce the reported tables from the supplied code and experiment inputs.
