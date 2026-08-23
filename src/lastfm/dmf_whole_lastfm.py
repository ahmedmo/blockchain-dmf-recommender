# -*- coding: utf-8 -*-
"""
DMF-2 BCE-Implicit for Last.fm — Matched CC/DC Settings — V9
================================================================

This script uses the two-tower Deep Matrix Factorization (DMF) architecture
of Xue et al. (IJCAI 2017) while restoring the multi-positive evaluation
protocol used for the Last.fm experiments in this research.

Canonical DMF backbone
----------------------
1. User input: training interaction-matrix row Y[u, :].
2. Item input: training interaction-matrix column Y[:, i].
3. Separate two-layer user and item projection networks.
4. Cosine similarity as the recommendation score.
5. Point-wise BCE for implicit feedback: observed training interactions are
   positive targets and sampled unseen interactions are negative targets.

Last.fm multi-positive protocol
-------------------------------
- All observed user-artist interactions are treated as positive implicit events.
- Legacy-compatible deterministic per-user fractional split:
      test_count = ceil(40% of interactions), at least 1
      validation_count = ceil(15% of the remaining interactions), when possible
      all remaining interactions are retained for training
      user-specific shuffle seed = global seed + encoded user index
- Small-K evaluation:
      sampled_99_multi_positive, K = {5, 10, 20}
- Large-K evaluation:
      all_items_multi_positive, K = {50, 100, 150, 200, 250, 300}
- The same held-out multi-positive test sets are used by both evaluation
  protocols; only the candidate set changes.
- Already observed training/validation interactions are excluded from final
  test ranking, while all held-out test positives remain eligible.
- Test metrics are never used for checkpoint selection.

Checkpoint selection
--------------------
The saved checkpoint is selected using the same sampled validation rule used
by the matched CC-DMF and DC-DMF community codes:
1. highest sampled validation NDCG@10;
2. if tied, highest sampled validation Recall@10;
3. if still tied, lowest validation BCE loss.

The selected checkpoint is then evaluated under BOTH final protocols:
- sampled_99_multi_positive for K={5,10,20};
- all_items_multi_positive for K={50,100,150,200,250,300}.

Important split note
--------------------
Although earlier code used the variable name LAST_N_TEST, the established
Last.fm protocol first applies a deterministic per-user shuffle and then takes
ceil(test_fraction * positive_count) interactions for test. Therefore this is
not a chronological split and does not require timestamps.

Input TSV required columns:
    userID    artistID    rating

Outputs:
- JSON run metadata
- Excel metrics and training history
- per-user Top-N recommendations CSV
- train/validation/test split CSV
- user and item mapping CSV files
- trained PyTorch state_dict

Reference:
H.-J. Xue, X.-Y. Dai, J. Zhang, S. Huang, and J. Chen,
"Deep Matrix Factorization Models for Recommender Systems," IJCAI 2017.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# 0) CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class Config:
    # Increment this value whenever any code change is made: V1, V2, V3, V4, V5, ...
    code_version: str = "V9"

    dataset_name: str = "LastFM"
    dataset_tag: str = "LastFM"

    ratings_path: str = r"data/lastfm/user_artists_ratings_per_user_norm_1dec.txt"
    output_root: str = (
        r"data/lastfm"
        r"\artis_info_json معتمد"
        r"\Uplift Recommendation"
        r"\Canonical DMF IJCAI2017"
    )

    seed: int = 42
    min_user_interactions: int = 4
    positive_threshold: float = 1.0

    # Legacy-compatible multi-positive fractional split.
    # Each user's interactions are deterministically shuffled with seed+u_idx.
    test_fraction: float = 0.40
    validation_fraction_of_remaining: float = 0.15
    timestamp_column: Optional[str] = None

    # Matched to the canonical CC-DMF / DC-DMF Last.fm community runs.
    # Observed interactions are encoded as 1 and sampled unobserved pairs as 0.
    negative_ratio: int = 1
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 80
    patience: int = 8
    min_delta: float = 1e-6

    # A two-layer DMF tower means:
    # input -> first projection (linear, no bias) -> final latent ReLU layer.
    # The paper studies two-layer DMF and initializes parameters N(0, 0.01).
    first_projection_dim: int = 64
    latent_dim: int = 64
    init_std: float = 0.01
    numerical_epsilon: float = 1e-6

    # Checkpoint selection matched to CC-DMF / DC-DMF.
    # Validation uses sampled multi-positive ranking with 99 sampled unseen items.
    checkpoint_k: int = 10

    # Final evaluation.
    sampled_negative_count: int = 99
    k_list_sampled: Tuple[int, ...] = (5, 10, 20)
    k_list_all: Tuple[int, ...] = (50, 100, 150, 200, 250, 300)
    topn_all_items: int = 300
    representation_batch_size: int = 2048

    # Runtime.
    num_workers: int = 0
    pin_memory: bool = True


CFG = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 1) GENERAL UTILITIES
# ============================================================================


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_float_tag(value: float) -> str:
    return str(value).replace(".", "d")


def build_version_tag(cfg: Config) -> str:
    return (
        f"{cfg.code_version}_DMF2_BCEIMP_MP_"
        f"TF{format_float_tag(cfg.test_fraction)}_"
        f"VF{format_float_tag(cfg.validation_fraction_of_remaining)}_"
        f"NEG{cfg.negative_ratio}_"
        f"P{cfg.first_projection_dim}_Z{cfg.latent_dim}_"
        f"LR{format_float_tag(cfg.learning_rate)}_"
        f"WD{format_float_tag(cfg.weight_decay)}_"
        f"BS{cfg.batch_size}_SELNDCG{cfg.checkpoint_k}_"
        f"NNEG{cfg.sampled_negative_count}_SEED{cfg.seed}"
    )


def build_run_dir(cfg: Config, version_tag: str) -> str:
    return os.path.join(cfg.output_root, f"RUN_{version_tag}")


def validate_config(cfg: Config) -> None:
    if cfg.min_user_interactions < 3:
        raise ValueError(
            "min_user_interactions must be at least 3 to preserve train, "
            "validation, and test interactions."
        )
    if not 0.0 < cfg.test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1.")
    if not 0.0 < cfg.validation_fraction_of_remaining < 1.0:
        raise ValueError(
            "validation_fraction_of_remaining must lie strictly between 0 and 1."
        )
    if cfg.negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1.")
    if cfg.first_projection_dim < 1 or cfg.latent_dim < 1:
        raise ValueError("DMF layer dimensions must be positive integers.")
    if cfg.checkpoint_k not in cfg.k_list_sampled:
        raise ValueError("checkpoint_k must be included in k_list_sampled.")
    if cfg.sampled_negative_count < 1:
        raise ValueError("sampled_negative_count must be at least 1.")
    if cfg.topn_all_items < max(cfg.k_list_all):
        raise ValueError(
            "topn_all_items must be greater than or equal to max(k_list_all)."
        )


# ============================================================================
# 2) DATA READING, FILTERING, AND ID ENCODING
# ============================================================================


def read_ratings_tsv(path: str, timestamp_column: Optional[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ratings file does not exist: {path}")

    df = pd.read_csv(path, sep="\t")
    if df.shape[1] < 3:
        raise ValueError(
            "The ratings file must contain at least three columns: "
            "userID, artistID/itemID, and rating."
        )

    original_columns = list(df.columns)
    rename_map = {
        original_columns[0]: "userID",
        original_columns[1]: "itemID",
        original_columns[2]: "rating",
    }
    df = df.rename(columns=rename_map)

    required = {"userID", "itemID", "rating"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")

    df["userID"] = pd.to_numeric(df["userID"], errors="raise").astype(np.int64)
    df["itemID"] = pd.to_numeric(df["itemID"], errors="raise").astype(np.int64)
    df["rating"] = pd.to_numeric(df["rating"], errors="raise").astype(np.float32)

    if not np.isfinite(df["rating"].to_numpy()).all():
        raise ValueError("The rating column contains NaN or infinite values.")
    if (df["rating"] <= 0).any():
        raise ValueError(
            "Canonical DMF reserves zero for unobserved entries; therefore, "
            "all observed ratings must be strictly greater than zero."
        )

    if timestamp_column is not None:
        if timestamp_column not in df.columns:
            raise ValueError(
                f"Configured timestamp column '{timestamp_column}' was not found."
            )
        df[timestamp_column] = pd.to_numeric(
            df[timestamp_column], errors="raise"
        )

    # A user-item matrix requires one value per pair. If duplicates exist,
    # retain the latest record when a timestamp is available; otherwise average.
    duplicate_count = int(df.duplicated(["userID", "itemID"], keep=False).sum())
    if duplicate_count > 0:
        print(
            f"[WARNING] Found {duplicate_count} rows belonging to duplicate "
            "user-item pairs."
        )
        if timestamp_column is not None:
            df = (
                df.sort_values(timestamp_column)
                .drop_duplicates(["userID", "itemID"], keep="last")
                .reset_index(drop=True)
            )
        else:
            # Match CC-DMF / DC-DMF duplicate handling.
            df = (
                df.groupby(["userID", "itemID"], as_index=False)["rating"]
                .max()
                .reset_index(drop=True)
            )

    return df.reset_index(drop=True)


def filter_users(df: pd.DataFrame, minimum_interactions: int) -> pd.DataFrame:
    counts = df.groupby("userID").size()
    retained_users = counts[counts >= minimum_interactions].index
    filtered = df[df["userID"].isin(retained_users)].copy()
    return filtered.reset_index(drop=True)


def encode_ids(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int], np.ndarray, np.ndarray]:
    user_ids = np.array(sorted(df["userID"].unique()), dtype=np.int64)
    item_ids = np.array(sorted(df["itemID"].unique()), dtype=np.int64)

    user_to_index = {int(user_id): index for index, user_id in enumerate(user_ids)}
    item_to_index = {int(item_id): index for index, item_id in enumerate(item_ids)}

    encoded = df.copy()
    encoded["u_idx"] = encoded["userID"].map(user_to_index).astype(np.int64)
    encoded["i_idx"] = encoded["itemID"].map(item_to_index).astype(np.int64)

    return encoded, user_to_index, item_to_index, user_ids, item_ids


# ============================================================================
# 3) LEGACY-COMPATIBLE FRACTIONAL MULTI-POSITIVE SPLIT
# ============================================================================


def split_fractional_per_user(
    df_encoded: pd.DataFrame,
    test_fraction: float,
    validation_fraction_of_remaining: float,
    seed: int,
) -> pd.DataFrame:
    """
    Reproduce the established Last.fm multi-positive split while materializing
    train/validation/test labels.

    For each user:
      test_count = ceil(test_fraction * observed_count), at least 1.
      validation_count = ceil(validation_fraction * remaining_count), at least 1
                         when at least two interactions remain.
      all remaining interactions are training interactions.

    The user-specific shuffle is deterministic: seed + encoded user index.
    """
    parts: List[pd.DataFrame] = []

    for user_index, user_rows in df_encoded.groupby("u_idx", sort=True):
        user_rows = user_rows.copy().reset_index(drop=True)
        interaction_count = len(user_rows)
        if interaction_count < 3:
            raise RuntimeError(
                f"User index {user_index} has fewer than three interactions."
            )

        rng = np.random.RandomState(seed + int(user_index))
        permutation = rng.permutation(interaction_count)

        test_count = max(1, int(math.ceil(test_fraction * interaction_count)))
        test_count = min(test_count, interaction_count - 1)

        test_positions = permutation[:test_count]
        remaining_positions = permutation[test_count:]

        if remaining_positions.size == 0:
            raise RuntimeError(f"User index {user_index} has no training remainder.")

        if remaining_positions.size >= 2:
            validation_count = max(
                1,
                int(
                    math.ceil(
                        validation_fraction_of_remaining
                        * int(remaining_positions.size)
                    )
                ),
            )
            validation_count = min(
                validation_count, int(remaining_positions.size) - 1
            )
        else:
            validation_count = 0

        validation_positions = remaining_positions[:validation_count]
        train_positions = remaining_positions[validation_count:]

        if train_positions.size == 0:
            train_positions = validation_positions[-1:]
            validation_positions = validation_positions[:-1]

        user_rows["split"] = "train"
        user_rows.loc[test_positions, "split"] = "test"
        if validation_positions.size > 0:
            user_rows.loc[validation_positions, "split"] = "validation"
        parts.append(user_rows)

    split_df = pd.concat(parts, ignore_index=True)

    counts = split_df.groupby(["u_idx", "split"]).size().unstack(fill_value=0)
    if not (counts.get("test", 0) >= 1).all():
        raise RuntimeError("Every retained user must have at least one test item.")
    if not (counts.get("train", 0) >= 1).all():
        raise RuntimeError("Every retained user must have at least one training item.")

    return split_df


def split_to_item_sets(
    split_df: pd.DataFrame,
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, Set[int]]]:
    def build(split_name: str) -> Dict[int, Set[int]]:
        subset = split_df[split_df["split"] == split_name]
        result: Dict[int, Set[int]] = {}
        for user_index, group in subset.groupby("u_idx", sort=True):
            result[int(user_index)] = set(group["i_idx"].astype(int).tolist())
        return result

    return build("train"), build("validation"), build("test")


def build_user_seen_all(split_df: pd.DataFrame) -> Dict[int, Set[int]]:
    result: Dict[int, Set[int]] = {}
    for user_index, group in split_df.groupby("u_idx", sort=True):
        result[int(user_index)] = set(group["i_idx"].astype(int).tolist())
    return result


# ============================================================================
# 4) CANONICAL DMF INTERACTION MATRIX
# ============================================================================


def build_training_interaction_matrix(
    split_df: pd.DataFrame,
    number_of_users: int,
    number_of_items: int,
) -> torch.Tensor:
    """
    Construct Y using Equation (2) of the DMF paper:

        Y[u, i] = 1 for an observed training interaction
        Y[u, i] = 0 for unobserved / held-out entries

    Validation and test interactions are intentionally excluded to prevent
    information leakage into the row/column input representations.
    """
    matrix = np.zeros(
        (number_of_users, number_of_items),
        dtype=np.float32,
    )

    train_rows = split_df[split_df["split"] == "train"]
    matrix[
        train_rows["u_idx"].to_numpy(dtype=np.int64),
        train_rows["i_idx"].to_numpy(dtype=np.int64),
    ] = 1.0

    return torch.from_numpy(matrix)


# ============================================================================
# 5) DMF-BCE-Implicit TRAINING SAMPLES
# ============================================================================


def rating_lookup_for_split(
    split_df: pd.DataFrame,
    split_name: str,
) -> Dict[Tuple[int, int], float]:
    subset = split_df[split_df["split"] == split_name]
    return {
        (int(row.u_idx), int(row.i_idx)): float(row.rating)
        for row in subset.itertuples(index=False)
    }


def sample_unobserved_items(
    rng: np.random.RandomState,
    unobserved_pool: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    if unobserved_pool.size == 0:
        return np.empty(0, dtype=np.int64)
    replace = unobserved_pool.size < sample_count
    return rng.choice(
        unobserved_pool,
        size=sample_count,
        replace=replace,
    ).astype(np.int64)


def create_dmf_bce_implicit_samples(
    observed_items: Mapping[int, Set[int]],
    user_seen_all: Mapping[int, Set[int]],
    number_of_items: int,
    negative_ratio: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a fixed implicit-feedback pointwise training set.

    Targets:
        observed interaction: 1
        sampled unobserved interaction: 0

    All validation and test positives are excluded from negative sampling through
    user_seen_all, so held-out interactions are never mislabeled as negatives.
    """
    users: List[int] = []
    items: List[int] = []
    targets: List[float] = []
    all_item_indices = np.arange(number_of_items, dtype=np.int64)

    for user_index in sorted(observed_items):
        positive_items = sorted(observed_items[user_index])
        if not positive_items:
            continue

        seen = np.fromiter(
            sorted(user_seen_all.get(user_index, set())),
            dtype=np.int64,
        )
        unobserved_pool = np.setdiff1d(
            all_item_indices,
            seen,
            assume_unique=True,
        )
        if unobserved_pool.size == 0:
            raise RuntimeError(
                f"User {user_index} has no unobserved items available for "
                "negative sampling."
            )

        rng = np.random.RandomState(seed + int(user_index))

        for item_index in positive_items:
            users.append(int(user_index))
            items.append(int(item_index))
            targets.append(1.0)

            negatives = sample_unobserved_items(
                rng=rng,
                unobserved_pool=unobserved_pool,
                sample_count=negative_ratio,
            )
            for negative_item in negatives:
                users.append(int(user_index))
                items.append(int(negative_item))
                targets.append(0.0)

    return (
        np.asarray(users, dtype=np.int64),
        np.asarray(items, dtype=np.int64),
        np.asarray(targets, dtype=np.float32),
    )


class DMFPointwiseDataset(Dataset):
    def __init__(
        self,
        users: np.ndarray,
        items: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        if not (len(users) == len(items) == len(targets)):
            raise ValueError("users, items, and targets must have equal lengths.")
        self.users = torch.from_numpy(users.astype(np.int64, copy=False))
        self.items = torch.from_numpy(items.astype(np.int64, copy=False))
        self.targets = torch.from_numpy(targets.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.users.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[index], self.items[index], self.targets[index]


# ============================================================================
# 6) CANONICAL TWO-TOWER DMF MODEL
# ============================================================================


class DMFTwoLayerTower(nn.Module):
    """
    Two-layer projection used by DMF-2:

        l1 = W1 x                         (no bias, no activation)
        h  = ReLU(W2 l1 + b2)            (final latent representation)

    This follows Equations (5)-(7) of the IJCAI 2017 paper more closely than
    applying ReLU after every projection indiscriminately.
    """

    def __init__(
        self,
        input_dimension: int,
        first_projection_dimension: int,
        latent_dimension: int,
        initialization_std: float,
    ) -> None:
        super().__init__()
        self.first_projection = nn.Linear(
            input_dimension,
            first_projection_dimension,
            bias=False,
        )
        self.final_projection = nn.Linear(
            first_projection_dimension,
            latent_dimension,
            bias=True,
        )
        self.reset_parameters(initialization_std)

    def reset_parameters(self, initialization_std: float) -> None:
        nn.init.normal_(
            self.first_projection.weight,
            mean=0.0,
            std=initialization_std,
        )
        nn.init.normal_(
            self.final_projection.weight,
            mean=0.0,
            std=initialization_std,
        )
        nn.init.zeros_(self.final_projection.bias)

    def forward(self, matrix_vectors: torch.Tensor) -> torch.Tensor:
        first_layer = self.first_projection(matrix_vectors)
        latent_representation = F.relu(self.final_projection(first_layer))
        return latent_representation


class CanonicalDMF(nn.Module):
    """
    Deep Matrix Factorization using interaction-matrix rows and columns.

    User representation:
        p_u = f_U(Y[u, :])

    Item representation:
        q_i = f_I(Y[:, i])

    Score:
        cosine(p_u, q_i)
    """

    def __init__(
        self,
        training_interaction_matrix: torch.Tensor,
        first_projection_dimension: int,
        latent_dimension: int,
        initialization_std: float = 0.01,
    ) -> None:
        super().__init__()

        if training_interaction_matrix.ndim != 2:
            raise ValueError(
                "training_interaction_matrix must have shape [users, items]."
            )
        if not training_interaction_matrix.is_floating_point():
            training_interaction_matrix = training_interaction_matrix.float()

        number_of_users, number_of_items = training_interaction_matrix.shape
        self.number_of_users = int(number_of_users)
        self.number_of_items = int(number_of_items)

        # The matrix is an input, not a learned parameter. persistent=False avoids
        # duplicating it inside state_dict; the saved split reconstructs it exactly.
        self.register_buffer(
            "training_interaction_matrix",
            training_interaction_matrix.float(),
            persistent=False,
        )

        self.user_network = DMFTwoLayerTower(
            input_dimension=self.number_of_items,
            first_projection_dimension=first_projection_dimension,
            latent_dimension=latent_dimension,
            initialization_std=initialization_std,
        )
        self.item_network = DMFTwoLayerTower(
            input_dimension=self.number_of_users,
            first_projection_dimension=first_projection_dimension,
            latent_dimension=latent_dimension,
            initialization_std=initialization_std,
        )

    def encode_users(self, user_indices: torch.Tensor) -> torch.Tensor:
        user_rows = self.training_interaction_matrix[user_indices, :]
        user_representations = self.user_network(user_rows)
        return F.normalize(user_representations, p=2, dim=-1, eps=1e-12)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        item_columns = self.training_interaction_matrix[:, item_indices].transpose(0, 1)
        item_representations = self.item_network(item_columns)
        return F.normalize(item_representations, p=2, dim=-1, eps=1e-12)

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        user_representations = self.encode_users(user_indices)
        item_representations = self.encode_items(item_indices)
        return torch.sum(
            user_representations * item_representations,
            dim=-1,
        )


# ============================================================================
# 7) IMPLICIT BCE AND TRAINING
# ============================================================================


def binary_cross_entropy_cosine_loss(
    cosine_scores: torch.Tensor,
    normalized_targets: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Binary cross-entropy over the DMF cosine score.

      -[y*log(score) + (1-y)*log(1-score)]

    Here y is 1 for every observed interaction and 0 for a sampled unobserved
    interaction. ReLU tower outputs keep cosine scores non-negative; clamping
    prevents log(0) and protects against floating-point drift.
    """
    probabilities = cosine_scores.clamp(
        min=epsilon,
        max=1.0 - epsilon,
    )
    return F.binary_cross_entropy(probabilities, normalized_targets)


def train_dmf_bce_implicit(
    model: CanonicalDMF,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    validation_items: Mapping[int, Set[int]],
    user_seen_all: Mapping[int, Set[int]],
    cfg: Config,
    device: str,
) -> Tuple[CanonicalDMF, List[Dict[str, float | int]], Dict[str, float | int]]:
    """
    Train canonical implicit-feedback DMF and select the checkpoint using the
    SAME validation rule as the matched CC-DMF / DC-DMF runs.

    Selection order:
      1. Higher sampled validation NDCG@checkpoint_k (default: NDCG@10).
      2. If tied within min_delta, higher sampled validation Recall@checkpoint_k.
      3. If still tied, lower validation BCE loss.

    Validation candidates contain all held-out validation positives jointly
    with up to 99 deterministically sampled unseen items. Training/test/other
    observed interactions are excluded from the negative pool via user_seen_all.

    Test metrics are never used during checkpoint selection.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_ndcg = float("-inf")
    best_recall = float("-inf")
    best_validation_loss = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0
    history: List[Dict[str, float | int]] = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        training_losses: List[float] = []

        for users, items, targets in train_loader:
            users = users.to(device, non_blocking=True)
            items = items.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            scores = model(users, items)
            loss = binary_cross_entropy_cosine_loss(
                cosine_scores=scores,
                normalized_targets=targets,
                epsilon=cfg.numerical_epsilon,
            )
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.item()))

        model.eval()
        validation_losses: List[float] = []
        with torch.no_grad():
            for users, items, targets in validation_loader:
                users = users.to(device, non_blocking=True)
                items = items.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                scores = model(users, items)
                loss = binary_cross_entropy_cosine_loss(
                    cosine_scores=scores,
                    normalized_targets=targets,
                    epsilon=cfg.numerical_epsilon,
                )
                validation_losses.append(float(loss.item()))

        train_loss = float(np.mean(training_losses)) if training_losses else float("nan")
        validation_loss = (
            float(np.mean(validation_losses))
            if validation_losses
            else float("nan")
        )
        if not math.isfinite(validation_loss):
            raise FloatingPointError(
                "Validation loss became non-finite. Check data and model settings."
            )

        validation_user_vectors = encode_all_users(
            model=model,
            device=device,
            batch_size=cfg.representation_batch_size,
        )
        validation_item_vectors = encode_all_items(
            model=model,
            device=device,
            batch_size=cfg.representation_batch_size,
        )

        validation_metrics = evaluate_sampled_negatives(
            user_vectors=validation_user_vectors,
            item_vectors=validation_item_vectors,
            test_items=validation_items,
            user_seen_all=user_seen_all,
            number_of_items=model.number_of_items,
            k_values=(cfg.checkpoint_k,),
            negative_count=cfg.sampled_negative_count,
            seed=cfg.seed + 2_000_000,
        )
        if validation_metrics.empty:
            raise RuntimeError("No users were available for sampled validation.")

        row = validation_metrics.iloc[0]
        validation_hr = float(row["HR"])
        validation_ndcg = float(row["NDCG"])
        validation_recall = float(row["Recall"])

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                f"validation_HR@{cfg.checkpoint_k}": validation_hr,
                f"validation_NDCG@{cfg.checkpoint_k}": validation_ndcg,
                f"validation_Recall@{cfg.checkpoint_k}": validation_recall,
            }
        )

        print(
            f"[DMF-WHOLE-MATCHED] epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_HR@{cfg.checkpoint_k}={validation_hr:.6f} "
            f"validation_NDCG@{cfg.checkpoint_k}={validation_ndcg:.6f} "
            f"validation_Recall@{cfg.checkpoint_k}={validation_recall:.6f}"
        )

        ndcg_improved = validation_ndcg > best_ndcg + cfg.min_delta
        ndcg_tied = abs(validation_ndcg - best_ndcg) <= cfg.min_delta
        recall_improved_on_tie = (
            ndcg_tied and validation_recall > best_recall + cfg.min_delta
        )
        recall_tied = (
            ndcg_tied and abs(validation_recall - best_recall) <= cfg.min_delta
        )
        loss_improved_on_full_tie = (
            recall_tied
            and validation_loss < best_validation_loss - cfg.min_delta
        )

        if ndcg_improved or recall_improved_on_tie or loss_improved_on_full_tie:
            best_ndcg = validation_ndcg
            best_recall = validation_recall
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
            print(
                f"[CHECKPOINT] epoch={epoch:03d} selected by "
                f"validation_NDCG@{cfg.checkpoint_k}={validation_ndcg:.6f}"
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(
                    f"[DMF-WHOLE-MATCHED] Early stopping: no validation "
                    f"NDCG@{cfg.checkpoint_k} improvement for "
                    f"{cfg.patience} epochs."
                )
                break

    if best_state is None or best_epoch < 1:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)
    best_checkpoint: Dict[str, float | int] = {
        "best_epoch": int(best_epoch),
        "validation_loss": float(best_validation_loss),
        f"validation_NDCG@{cfg.checkpoint_k}": float(best_ndcg),
        f"validation_Recall@{cfg.checkpoint_k}": float(best_recall),
    }

    print(
        f"[BEST CHECKPOINT] epoch={best_epoch:03d} "
        f"validation_loss={best_validation_loss:.6f} "
        f"validation_NDCG@{cfg.checkpoint_k}={best_ndcg:.6f} "
        f"validation_Recall@{cfg.checkpoint_k}={best_recall:.6f}"
    )

    return model, history, best_checkpoint


# ============================================================================
# 8) METRICS
# ============================================================================


def dcg_at_positions(one_based_hit_positions: Iterable[int]) -> float:
    return float(
        sum(1.0 / math.log2(position + 1) for position in one_based_hit_positions)
    )


def ndcg_binary(top_k_items: Sequence[int], positive_items: Set[int]) -> float:
    hit_positions = [
        rank
        for rank, item_index in enumerate(top_k_items, start=1)
        if item_index in positive_items
    ]
    if not hit_positions:
        return 0.0

    dcg = dcg_at_positions(hit_positions)
    ideal_length = min(len(positive_items), len(top_k_items))
    idcg = dcg_at_positions(range(1, ideal_length + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def average_precision_at_k(
    top_k_items: Sequence[int],
    positive_items: Set[int],
) -> float:
    if not positive_items:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item_index in enumerate(top_k_items, start=1):
        if item_index in positive_items:
            hits += 1
            precision_sum += hits / rank

    denominator = min(len(positive_items), len(top_k_items))
    return float(precision_sum / denominator) if denominator > 0 else 0.0


def evaluate_top_k_for_user(
    ranked_items: Sequence[int],
    positive_items: Set[int],
    k: int,
) -> Dict[str, float]:
    selected = list(ranked_items[:k])
    hit_count = sum(item in positive_items for item in selected)

    hit_ratio = 1.0 if hit_count > 0 else 0.0
    recall = hit_count / len(positive_items) if positive_items else 0.0
    precision = hit_count / k if k > 0 else 0.0
    ndcg = ndcg_binary(selected, positive_items)
    mean_average_precision = average_precision_at_k(selected, positive_items)

    return {
        "HR": float(hit_ratio),
        "NDCG": float(ndcg),
        "Recall": float(recall),
        "Precision": float(precision),
        "MAP": float(mean_average_precision),
    }


# ============================================================================
# 9) REPRESENTATION PRECOMPUTATION AND RANKING
# ============================================================================


@torch.no_grad()
def encode_all_users(
    model: CanonicalDMF,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    vectors: List[np.ndarray] = []

    for start in range(0, model.number_of_users, batch_size):
        stop = min(start + batch_size, model.number_of_users)
        indices = torch.arange(start, stop, dtype=torch.long, device=device)
        batch_vectors = model.encode_users(indices)
        vectors.append(batch_vectors.cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(vectors, axis=0)


@torch.no_grad()
def encode_all_items(
    model: CanonicalDMF,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    vectors: List[np.ndarray] = []

    for start in range(0, model.number_of_items, batch_size):
        stop = min(start + batch_size, model.number_of_items)
        indices = torch.arange(start, stop, dtype=torch.long, device=device)
        batch_vectors = model.encode_items(indices)
        vectors.append(batch_vectors.cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(vectors, axis=0)


def rank_candidate_items(
    user_vector: np.ndarray,
    item_vectors: np.ndarray,
    candidate_items: Sequence[int],
) -> List[int]:
    candidate_array = np.asarray(candidate_items, dtype=np.int64)
    if candidate_array.size == 0:
        return []

    scores = item_vectors[candidate_array] @ user_vector
    # Primary key: descending score. Secondary deterministic key: smaller item index.
    order = np.lexsort((candidate_array, -scores))
    return candidate_array[order].astype(int).tolist()


# ============================================================================
# 10) MULTI-POSITIVE SAMPLED-99 AND ALL-ITEMS EVALUATION
# ============================================================================


def _deterministic_top_k_from_scores(
    scores: np.ndarray,
    candidate_mask: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return exact top-k with score-desc / smaller-item-index tie-breaking."""
    candidate_indices = np.flatnonzero(candidate_mask).astype(np.int64)
    if candidate_indices.size == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)

    k_eff = min(int(k), int(candidate_indices.size))
    candidate_scores = scores[candidate_indices]

    if candidate_indices.size <= k_eff:
        order = np.lexsort((candidate_indices, -candidate_scores))
        return candidate_indices[order]

    threshold = float(np.partition(candidate_scores, -k_eff)[-k_eff])
    strictly_better = candidate_indices[candidate_scores > threshold]
    tied = np.sort(candidate_indices[candidate_scores == threshold])
    need_from_ties = k_eff - int(strictly_better.size)
    selected = np.concatenate([strictly_better, tied[:need_from_ties]])
    selected_scores = scores[selected]
    order = np.lexsort((selected, -selected_scores))
    return selected[order]


def evaluate_validation_all_items_recall(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    train_items: Mapping[int, Set[int]],
    validation_items: Mapping[int, Set[int]],
    test_items: Mapping[int, Set[int]],
    k_values: Sequence[int],
) -> Dict[int, float]:
    """
    Exact macro-mean all-items Recall@K for multi-positive validation sets.

    Training and held-out test interactions are excluded from the validation
    candidate set. All validation positives remain eligible together with all
    truly unseen items.
    """
    k_values_int = tuple(sorted({int(k) for k in k_values}))
    if not k_values_int:
        raise ValueError("k_values must not be empty.")

    recall_sums = {k: 0.0 for k in k_values_int}
    users_evaluated = 0
    number_of_items = int(item_vectors.shape[0])
    max_k = max(k_values_int)

    for user_index in sorted(validation_items):
        positives = set(validation_items.get(user_index, set()))
        if not positives:
            continue

        scores = item_vectors @ user_vectors[user_index]
        candidate_mask = np.ones(number_of_items, dtype=bool)

        excluded = set(train_items.get(user_index, set()))
        excluded.update(test_items.get(user_index, set()))
        excluded.difference_update(positives)
        if excluded:
            candidate_mask[np.fromiter(sorted(excluded), dtype=np.int64)] = False

        ranked_top = _deterministic_top_k_from_scores(
            scores=scores,
            candidate_mask=candidate_mask,
            k=max_k,
        )

        users_evaluated += 1
        positive_count = len(positives)
        for k in k_values_int:
            selected = ranked_top[: min(k, ranked_top.size)]
            hit_count = sum(int(item) in positives for item in selected)
            recall_sums[k] += hit_count / positive_count

    if users_evaluated == 0:
        raise RuntimeError("No users were available for validation ranking.")

    return {k: recall_sums[k] / users_evaluated for k in k_values_int}


def evaluate_sampled_negatives(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    test_items: Mapping[int, Set[int]],
    user_seen_all: Mapping[int, Set[int]],
    number_of_items: int,
    k_values: Sequence[int],
    negative_count: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate all held-out test positives jointly with sampled unseen negatives."""
    rows: List[Dict[str, float]] = []
    all_items = np.arange(number_of_items, dtype=np.int64)

    for user_index in sorted(test_items):
        positive_set = set(test_items[user_index])
        if not positive_set:
            continue

        seen = np.fromiter(
            sorted(user_seen_all.get(user_index, set())),
            dtype=np.int64,
        )
        negative_pool = np.setdiff1d(all_items, seen, assume_unique=True)

        rng = np.random.RandomState(seed + int(user_index))
        effective_count = min(negative_count, int(negative_pool.size))
        negatives = (
            rng.choice(negative_pool, size=effective_count, replace=False)
            .astype(np.int64)
            .tolist()
            if effective_count > 0
            else []
        )

        candidates = sorted(positive_set) + negatives
        ranked = rank_candidate_items(
            user_vector=user_vectors[user_index],
            item_vectors=item_vectors,
            candidate_items=candidates,
        )

        for k in k_values:
            metrics = evaluate_top_k_for_user(ranked, positive_set, int(k))
            rows.append(
                {
                    "user": int(user_index),
                    "k": int(k),
                    "test_positive_count": int(len(positive_set)),
                    **metrics,
                }
            )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "EvalProtocol",
                "k",
                "HR",
                "NDCG",
                "Recall",
                "Precision",
                "MAP",
                "N_NEG_CANDIDATES_requested",
                "multi_positive",
                "mean_test_positives_per_user",
                "users_eval",
            ]
        )

    mean_metrics = (
        detail.groupby("k")[["HR", "NDCG", "Recall", "Precision", "MAP"]]
        .mean()
        .reindex(list(k_values))
        .reset_index()
    )
    mean_metrics.insert(0, "EvalProtocol", f"sampled_{negative_count}_multi_positive")
    mean_metrics["N_NEG_CANDIDATES_requested"] = int(negative_count)
    mean_metrics["multi_positive"] = True
    mean_metrics["mean_test_positives_per_user"] = float(
        detail.groupby("user")["test_positive_count"].first().mean()
    )
    mean_metrics["users_eval"] = int(detail["user"].nunique())
    return mean_metrics


def build_all_items_topn(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    train_items: Mapping[int, Set[int]],
    validation_items: Mapping[int, Set[int]],
    top_n: int,
) -> Dict[int, List[int]]:
    number_of_items = int(item_vectors.shape[0])
    all_item_indices = np.arange(number_of_items, dtype=np.int64)
    recommendations: Dict[int, List[int]] = {}

    for user_index in range(user_vectors.shape[0]):
        excluded = set(train_items.get(user_index, set()))
        excluded.update(validation_items.get(user_index, set()))

        if excluded:
            excluded_array = np.fromiter(sorted(excluded), dtype=np.int64)
            candidate_mask = np.ones(number_of_items, dtype=bool)
            candidate_mask[excluded_array] = False
            candidates = all_item_indices[candidate_mask]
        else:
            candidates = all_item_indices

        ranked = rank_candidate_items(
            user_vector=user_vectors[user_index],
            item_vectors=item_vectors,
            candidate_items=candidates,
        )
        recommendations[user_index] = ranked[:top_n]

    return recommendations


def evaluate_all_items(
    recommendations: Mapping[int, Sequence[int]],
    test_items: Mapping[int, Set[int]],
    k_values: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for user_index in sorted(recommendations):
        positive_set = set(test_items.get(user_index, set()))
        if not positive_set:
            continue

        ranked = recommendations[user_index]
        for k in k_values:
            metrics = evaluate_top_k_for_user(ranked, positive_set, int(k))
            rows.append(
                {
                    "user": int(user_index),
                    "k": int(k),
                    **metrics,
                }
            )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "EvalProtocol",
                "k",
                "HR",
                "NDCG",
                "Recall",
                "Precision",
                "MAP",
                "users_eval",
            ]
        )

    mean_metrics = (
        detail.groupby("k")[["HR", "NDCG", "Recall", "Precision", "MAP"]]
        .mean()
        .reset_index()
    )
    mean_metrics.insert(0, "EvalProtocol", "all_items_multi_positive")
    mean_metrics["multi_positive"] = True
    mean_metrics["users_eval"] = int(detail["user"].nunique())
    return mean_metrics


# ============================================================================
# 11) OUTPUT HELPERS
# ============================================================================


def save_id_mappings(
    run_dir: str,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
) -> Tuple[str, str]:
    user_mapping_path = os.path.join(run_dir, "user_id_mapping.csv")
    item_mapping_path = os.path.join(run_dir, "item_id_mapping.csv")

    pd.DataFrame(
        {
            "u_idx": np.arange(len(user_ids), dtype=np.int64),
            "userID": user_ids,
        }
    ).to_csv(user_mapping_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "i_idx": np.arange(len(item_ids), dtype=np.int64),
            "itemID": item_ids,
        }
    ).to_csv(item_mapping_path, index=False, encoding="utf-8-sig")

    return user_mapping_path, item_mapping_path


def save_recommendations(
    recommendations: Mapping[int, Sequence[int]],
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    output_path: str,
) -> None:
    rows: List[Dict[str, int]] = []
    for user_index, ranked_items in recommendations.items():
        for rank, item_index in enumerate(ranked_items, start=1):
            rows.append(
                {
                    "u_idx": int(user_index),
                    "userID": int(user_ids[user_index]),
                    "i_idx": int(item_index),
                    "itemID": int(item_ids[item_index]),
                    "rank": int(rank),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


# ============================================================================
# 12) MAIN
# ============================================================================


def main() -> None:
    validate_config(CFG)
    set_all_seeds(CFG.seed)

    version_tag = build_version_tag(CFG)
    run_dir = build_run_dir(CFG, version_tag)
    ensure_dir(run_dir)

    print(f"[DEVICE] {DEVICE}")
    print(f"[INPUT] {CFG.ratings_path}")
    print(f"[VERSION] {version_tag}")

    df = read_ratings_tsv(CFG.ratings_path, CFG.timestamp_column)
    print(
        f"[RAW] rows={len(df)} users={df['userID'].nunique()} "
        f"items={df['itemID'].nunique()} "
        f"rating_min={df['rating'].min():.4f} "
        f"rating_max={df['rating'].max():.4f}"
    )

    # Match CC-DMF / DC-DMF implicit-positive threshold.
    df_positive = df[df["rating"] >= CFG.positive_threshold].copy().reset_index(drop=True)
    df_filtered = filter_users(df_positive, CFG.min_user_interactions)
    if df_filtered.empty:
        raise RuntimeError(
            "No users remain after positive-threshold/minimum-interaction filtering."
        )

    print(
        f"[FILTERED] rows={len(df_filtered)} "
        f"users={df_filtered['userID'].nunique()} "
        f"items={df_filtered['itemID'].nunique()}"
    )

    (
        df_encoded,
        _user_to_index,
        _item_to_index,
        user_ids,
        item_ids,
    ) = encode_ids(df_filtered)

    number_of_users = int(len(user_ids))
    number_of_items = int(len(item_ids))
    maximum_rating = float(df_encoded["rating"].max())

    split_df = split_fractional_per_user(
        df_encoded=df_encoded,
        test_fraction=CFG.test_fraction,
        validation_fraction_of_remaining=CFG.validation_fraction_of_remaining,
        seed=CFG.seed,
    )

    train_items, validation_items, test_items = split_to_item_sets(split_df)
    user_seen_all = build_user_seen_all(split_df)

    split_counts = split_df["split"].value_counts()
    test_positive_counts = (
        split_df[split_df["split"] == "test"]
        .groupby("u_idx")
        .size()
    )
    validation_positive_counts = (
        split_df[split_df["split"] == "validation"]
        .groupby("u_idx")
        .size()
    )
    print(
        f"[SPLIT] train={int(split_counts.get('train', 0))} "
        f"validation={int(split_counts.get('validation', 0))} "
        f"test={int(split_counts.get('test', 0))} "
        f"mean_test_pos/user={test_positive_counts.mean():.3f} "
        f"mean_val_pos/user={validation_positive_counts.mean():.3f}"
    )

    training_matrix = build_training_interaction_matrix(
        split_df=split_df,
        number_of_users=number_of_users,
        number_of_items=number_of_items,
    )

    train_users, train_sample_items, train_targets = create_dmf_bce_implicit_samples(
        observed_items=train_items,
        user_seen_all=user_seen_all,
        number_of_items=number_of_items,
        negative_ratio=CFG.negative_ratio,
        seed=CFG.seed,
    )
    validation_users, validation_sample_items, validation_targets = (
        create_dmf_bce_implicit_samples(
            observed_items=validation_items,
            user_seen_all=user_seen_all,
            number_of_items=number_of_items,
            negative_ratio=CFG.negative_ratio,
            seed=CFG.seed + 1_000_000,
        )
    )

    print(
        f"[BCE-IMPLICIT SAMPLES] train={len(train_targets)} "
        f"validation={len(validation_targets)} "
        f"negative_ratio={CFG.negative_ratio}"
    )

    train_dataset = DMFPointwiseDataset(
        train_users,
        train_sample_items,
        train_targets,
    )
    validation_dataset = DMFPointwiseDataset(
        validation_users,
        validation_sample_items,
        validation_targets,
    )

    pin_memory = bool(CFG.pin_memory and DEVICE == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=CFG.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=CFG.num_workers,
        pin_memory=pin_memory,
    )

    model = CanonicalDMF(
        training_interaction_matrix=training_matrix,
        first_projection_dimension=CFG.first_projection_dim,
        latent_dimension=CFG.latent_dim,
        initialization_std=CFG.init_std,
    ).to(DEVICE)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[MODEL] users={number_of_users} items={number_of_items} "
        f"parameters={parameter_count:,} "
        f"user_input_dim={number_of_items} "
        f"item_input_dim={number_of_users}"
    )

    model, history, best_checkpoint = train_dmf_bce_implicit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        validation_items=validation_items,
        user_seen_all=user_seen_all,
        cfg=CFG,
        device=DEVICE,
    )

    print("[EVAL] Precomputing user and item latent representations...")
    user_vectors = encode_all_users(
        model=model,
        device=DEVICE,
        batch_size=CFG.representation_batch_size,
    )
    item_vectors = encode_all_items(
        model=model,
        device=DEVICE,
        batch_size=CFG.representation_batch_size,
    )

    print(
        f"[EVAL] sampled_{CFG.sampled_negative_count}_multi_positive"
    )
    sampled_metrics = evaluate_sampled_negatives(
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        test_items=test_items,
        user_seen_all=user_seen_all,
        number_of_items=number_of_items,
        k_values=CFG.k_list_sampled,
        negative_count=CFG.sampled_negative_count,
        seed=CFG.seed,
    )
    print(sampled_metrics)

    print(f"[RECS] Building all-items Top-{CFG.topn_all_items} recommendations...")
    recommendations = build_all_items_topn(
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        train_items=train_items,
        validation_items=validation_items,
        top_n=CFG.topn_all_items,
    )

    all_items_metrics = evaluate_all_items(
        recommendations=recommendations,
        test_items=test_items,
        k_values=CFG.k_list_all,
    )
    print(all_items_metrics)

    print("\n" + "=" * 100)
    print("[TABLE 6 READY — DMF-WHOLE MATCHED — SAMPLED_99]")
    print(sampled_metrics[["k", "HR", "NDCG", "Recall", "Precision", "MAP"]])

    print("\n" + "=" * 100)
    print("[TABLE 7 READY — DMF-WHOLE MATCHED — ALL ITEMS]")
    print(all_items_metrics[["k", "Recall"]])

    variant_name = f"DMF_GLOBAL_{CFG.dataset_tag}_DMF2_BCEIMP_{version_tag}"
    sampled_output = sampled_metrics.copy()
    all_items_output = all_items_metrics.copy()
    sampled_output.insert(1, "Variant", variant_name)
    all_items_output.insert(1, "Variant", variant_name)
    combined_metrics = pd.concat(
        [sampled_output, all_items_output],
        ignore_index=True,
        sort=False,
    )

    # ------------------------------------------------------------------------
    # Save reproducibility artifacts.
    # ------------------------------------------------------------------------
    split_path = os.path.join(run_dir, f"{CFG.dataset_tag}_split_{version_tag}.csv")
    split_df.to_csv(split_path, index=False, encoding="utf-8-sig")

    user_mapping_path, item_mapping_path = save_id_mappings(
        run_dir=run_dir,
        user_ids=user_ids,
        item_ids=item_ids,
    )

    recommendations_path = os.path.join(
        run_dir,
        f"{CFG.dataset_tag}_DMF_GLOBAL_top{CFG.topn_all_items}_{version_tag}.csv",
    )
    save_recommendations(
        recommendations=recommendations,
        user_ids=user_ids,
        item_ids=item_ids,
        output_path=recommendations_path,
    )

    model_path = os.path.join(
        run_dir,
        f"{CFG.dataset_tag}_DMF2_BCEIMP_model_{version_tag}.pt",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(CFG),
            "version_tag": version_tag,
            "number_of_users": number_of_users,
            "number_of_items": number_of_items,
            "maximum_rating": maximum_rating,
            "best_checkpoint": best_checkpoint,
        },
        model_path,
    )

    metadata = {
        **asdict(CFG),
        "device": DEVICE,
        "matched_to": "LastFM canonical CC-DMF/DC-DMF training settings",
        "version_tag": version_tag,
        "number_of_users": number_of_users,
        "number_of_items": number_of_items,
        "number_of_filtered_rows": int(len(df_encoded)),
        "maximum_rating": maximum_rating,
        "model_parameter_count": int(parameter_count),
        "split_protocol": {
            "name": "deterministic_fractional_multi_positive",
            "test_fraction": float(CFG.test_fraction),
            "validation_fraction_of_remaining": float(CFG.validation_fraction_of_remaining),
            "shuffle": "RandomState(seed + encoded_user_index)",
            "integer_rule": "ceil",
            "temporal_interpretation": False,
            "mean_test_positives_per_user": float(test_positive_counts.mean()),
            "mean_validation_positives_per_user": float(validation_positive_counts.mean()),
        },
        "evaluation_protocols": {
            "small_k": f"sampled_{CFG.sampled_negative_count}_multi_positive",
            "large_k": "all_items_multi_positive",
            "small_k_values": list(CFG.k_list_sampled),
            "large_k_values": list(CFG.k_list_all),
        },
        "checkpoint_selection": {
            "criterion": f"sampled validation NDCG@{CFG.checkpoint_k}",
            "sampled_negative_count": int(CFG.sampled_negative_count),
            "tie_break_1": f"sampled validation Recall@{CFG.checkpoint_k}",
            "tie_break_2": "validation BCE loss",
            "test_metrics_used_for_selection": False,
            **best_checkpoint,
        },
        "architecture": {
            "user_input": "training interaction-matrix row Y[u, :]",
            "item_input": "training interaction-matrix column Y[:, i]",
            "user_network": "two-layer DMF projection",
            "item_network": "two-layer DMF projection",
            "score": "cosine similarity",
            "interaction_matrix": "binary observed=1, unobserved/held-out=0",
            "loss": "binary cross-entropy with observed=1 and sampled unobserved=0",
        },
        "artifacts": {
            "split_csv": split_path,
            "user_mapping_csv": user_mapping_path,
            "item_mapping_csv": item_mapping_path,
            "recommendations_csv": recommendations_path,
            "model_state_dict": model_path,
        },
    }

    metadata_path = os.path.join(
        run_dir,
        f"{CFG.dataset_tag}_DMF2_BCEIMP_meta_{version_tag}.json",
    )
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    excel_path = os.path.join(
        run_dir,
        f"{CFG.dataset_tag}_DMF_GLOBAL_{version_tag}_BOTH_PROTOCOLS.xlsx",
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        combined_metrics.to_excel(
            writer,
            index=False,
            sheet_name="metrics_both",
        )
        sampled_metrics.to_excel(
            writer,
            index=False,
            sheet_name="sampled_99_mean",
        )
        all_items_metrics.to_excel(
            writer,
            index=False,
            sheet_name="all_items_mean",
        )
        pd.DataFrame(history).to_excel(
            writer,
            index=False,
            sheet_name="training_history",
        )

    summary_path = os.path.join(
        CFG.output_root,
        f"DMF_GLOBAL_summary_metrics_{CFG.dataset_tag}.xlsx",
    )
    try:
        if os.path.exists(summary_path):
            previous = pd.read_excel(summary_path)
            summary = pd.concat(
                [previous, combined_metrics],
                ignore_index=True,
                sort=False,
            )
        else:
            summary = combined_metrics.copy()
        summary.to_excel(summary_path, index=False)
    except PermissionError:
        print(
            f"[WARNING] Summary file is open or locked; skipped: {summary_path}"
        )

    print("\n[SAVED]")
    print(f"Metadata:        {metadata_path}")
    print(f"Metrics:         {excel_path}")
    print(f"Summary:         {summary_path}")
    print(f"Split:           {split_path}")
    print(f"Recommendations: {recommendations_path}")
    print(f"Model:           {model_path}")


if __name__ == "__main__":
    main()
