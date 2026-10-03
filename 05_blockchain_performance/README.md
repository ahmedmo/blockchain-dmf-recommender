# Blockchain performance provenance

This stage contains the finalized blockchain performance notebooks and the complete recorded output set selected for each dataset.

`blockchain_performance_transactions.csv` is the transaction-level machine-readable execution log. It records transaction and receipt-derived information such as transaction hash, receipt status, block number, sender, contract address, gas usage, gas price, timing, and stage/community identifiers where applicable.

`blockchain_performance_by_stage.csv` and `blockchain_performance_by_community.csv` are aggregate summaries derived from the transaction log.

`run_metadata.json` records the Ganache connection, chain ID, account count, Solidity compiler, measured scope, contract address, and run identifiers.
