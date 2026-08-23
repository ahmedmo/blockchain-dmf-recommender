# -*- coding: utf-8 -*-
"""
Canonical DMF-2 for Douban Movie — Whole-network / All-items — V1
=================================================================

Purpose
-------
This script replaces the previous user-ID/item-ID embedding + concatenation +
MLP implementation with the two-tower Deep Matrix Factorization architecture
described by Xue et al. (IJCAI 2017), while preserving the Douban-specific
experimental choices from the previous implementation as far as possible.

Canonical DMF backbone
----------------------
1. User input: the corresponding row Y[u, :] of the training interaction matrix.
2. Item input: the corresponding column Y[:, i] of the training interaction matrix.
3. Separate user and item projection networks.
4. Cosine similarity as the recommendation score.
5. Pointwise binary cross-entropy for implicit relevance:
       rating >= POSITIVE_THRESHOLD -> positive target 1
       sampled non-positive item    -> negative target 0

Douban protocol retained from the previous code
-----------------------------------------------
- Input: "um - MOD.txt"
- Positive/relevant threshold: rating >= 4
- File-order Last-N positive split:
      20% of positive items for test
      15% of the remaining positive items for validation
- Negative ratio: 4
- Projection sizes: P=128, Z=64
- Batch size: 512
- Learning rate: 1e-3
- Maximum epochs: 50
- Patience: 5
- K = {5, 10, 20}
- sampled_99 multi-positive evaluation
- true all-items multi-positive evaluation

Checkpoint selection
--------------------
The saved checkpoint is selected using validation data only:
1. highest all-items validation NDCG@10;
2. then highest all-items validation Recall@10;
3. then lowest validation BCE loss.

Important evaluation correction
-------------------------------
The old script generated all-items recommendation files but evaluated its main
metrics only under sampled_99, and its recommendation generator did not exclude
already observed items. This implementation:
- reports sampled_99 and true all-items metrics separately;
- excludes all known rated items from recommendation candidates, except the
  held-out positive items being evaluated;
- never uses test metrics for model selection.

Versioning
----------
This is V1 of the independent Douban Canonical-DMF whole-network code line.
Increment code_version for every later code change.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 0) CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class Config:
    code_version: str = "V1"

    dataset_name: str = "Douban Movie"
    dataset_tag: str = "Douban"

    ratings_path: str = (
        r"data/douban"
        r"\Unique\UserMovie profile\um - MOD.txt"
    )
    output_root: str = (
        r"outputs/douban"
        r"\Canonical_DMF_Douban_AllItems"
    )

    seed: int = 42
    min_user_interactions: int = 4

    # Relevance definition retained from the old Douban code.
    positive_threshold: float = 4.0

    # File-order Last-N split on positive interactions.
    test_fraction: float = 0.20
    validation_fraction_of_remaining: float = 0.15

    # Training settings adapted from the previous Douban MLP/BPR experiment.
    negative_ratio: int = 4
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 50
    patience: int = 5
    min_delta: float = 1e-6

    # The old architecture used EMB=128 and hidden=[128,64].
    # For canonical DMF-2 these become P=128 and final latent Z=64.
    first_projection_dim: int = 128
    latent_dim: int = 64
    init_std: float = 0.01
    numerical_epsilon: float = 1e-6

    # Small-K checkpoint selection and evaluation.
    checkpoint_k: int = 10
    k_values: Tuple[int, ...] = (5, 10, 20)
    sampled_negative_count: int = 99
    topn_all_items: int = 20

    representation_batch_size: int = 2048
    score_user_batch_size: int = 256

    num_workers: int = 0
    pin_memory: bool = True


CFG = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 1) GENERAL UTILITIES
# =============================================================================


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
    text = f"{value:.12g}"
    return text.replace(".", "d").replace("-", "m")


def build_version_tag(cfg: Config) -> str:
    return (
        f"{cfg.code_version}_DMF2_BCEIMP_DBN_LASTN_"
        f"TF{format_float_tag(cfg.test_fraction)}_"
        f"VF{format_float_tag(cfg.validation_fraction_of_remaining)}_"
        f"POS{format_float_tag(cfg.positive_threshold)}_"
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
    if cfg.min_user_interactions < 1:
        raise ValueError("min_user_interactions must be positive.")
    if not 0.0 < cfg.test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1.")
    if not 0.0 <= cfg.validation_fraction_of_remaining < 1.0:
        raise ValueError(
            "validation_fraction_of_remaining must be in [0, 1)."
        )
    if cfg.negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1.")
    if cfg.first_projection_dim < 1 or cfg.latent_dim < 1:
        raise ValueError("DMF layer dimensions must be positive.")
    if cfg.checkpoint_k not in cfg.k_values:
        raise ValueError("checkpoint_k must be present in k_values.")
    if cfg.topn_all_items < max(cfg.k_values):
        raise ValueError("topn_all_items must be at least max(k_values).")
    if cfg.sampled_negative_count < 1:
        raise ValueError("sampled_negative_count must be at least 1.")


# =============================================================================
# 2) DATA READING, FILTERING, AND ID ENCODING
# =============================================================================


def read_douban_ratings(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ratings file does not exist: {path}")

    raw = pd.read_csv(
        path,
        sep=r"[\s,]+",
        header=None,
        engine="python",
        comment="#",
    )
    if raw.shape[1] < 3:
        raise ValueError(
            "The Douban file must contain at least three columns: "
            "user, item, rating."
        )

    df = raw.iloc[:, :3].copy()
    df.columns = ["userID", "itemID", "rating"]

    for column in ["userID", "itemID", "rating"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["userID", "itemID", "rating"]).copy()
    dropped = before - len(df)
    if dropped:
        print(f"[WARNING] Dropped {dropped} non-numeric/header rows.")

    df["userID"] = df["userID"].astype(np.int64)
    df["itemID"] = df["itemID"].astype(np.int64)
    df["rating"] = df["rating"].astype(np.float32)
    df["_row_order"] = np.arange(len(df), dtype=np.int64)

    if not np.isfinite(df["rating"].to_numpy()).all():
        raise ValueError("The rating column contains NaN or infinite values.")

    duplicate_count = int(
        df.duplicated(["userID", "itemID"], keep=False).sum()
    )
    if duplicate_count > 0:
        print(
            f"[WARNING] Found {duplicate_count} rows belonging to duplicate "
            "user-item pairs; retaining the last row in file order."
        )
        df = (
            df.sort_values("_row_order", kind="mergesort")
            .drop_duplicates(["userID", "itemID"], keep="last")
            .reset_index(drop=True)
        )
        df["_row_order"] = np.arange(len(df), dtype=np.int64)

    return df.reset_index(drop=True)


def filter_users(
    df: pd.DataFrame,
    minimum_interactions: int,
) -> pd.DataFrame:
    counts = df.groupby("userID").size()
    retained_users = counts[counts >= minimum_interactions].index
    filtered = df[df["userID"].isin(retained_users)].copy()
    return filtered.reset_index(drop=True)


def encode_ids(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    Dict[int, int],
    Dict[int, int],
    np.ndarray,
    np.ndarray,
]:
    user_ids = np.array(sorted(df["userID"].unique()), dtype=np.int64)
    item_ids = np.array(sorted(df["itemID"].unique()), dtype=np.int64)

    user_to_index = {
        int(user_id): index for index, user_id in enumerate(user_ids)
    }
    item_to_index = {
        int(item_id): index for index, item_id in enumerate(item_ids)
    }

    encoded = df.copy()
    encoded["u_idx"] = (
        encoded["userID"].map(user_to_index).astype(np.int64)
    )
    encoded["i_idx"] = (
        encoded["itemID"].map(item_to_index).astype(np.int64)
    )

    return encoded, user_to_index, item_to_index, user_ids, item_ids


# =============================================================================
# 3) POSITIVE LAST-N TRAIN / VALIDATION / TEST SPLIT
# =============================================================================


def split_positive_last_n(
    df_encoded: pd.DataFrame,
    positive_threshold: float,
    test_fraction: float,
    validation_fraction_of_remaining: float,
) -> pd.DataFrame:
    """
    Assign positive interactions to train/validation/test in file order.

    Non-positive observed ratings remain labeled 'known_nonpositive'. They do
    not enter the positive interaction matrix, but they remain known rated
    items and are excluded from final recommendation candidates.

    Users with:
    - >=3 positives: train + validation + test;
    - 2 positives: train + test, no validation;
    - 1 positive: train only, no validation/test.

    This preserves as many users as possible while preventing empty training
    profiles. Evaluation automatically uses only users with held-out positives.
    """
    result = df_encoded.copy()
    result["split"] = "known_nonpositive"

    positive_mask = result["rating"] >= positive_threshold
    positive_rows = result[positive_mask]

    for user_index, group in positive_rows.groupby("u_idx", sort=True):
        ordered = group.sort_values(
            ["_row_order", "i_idx"],
            ascending=[True, True],
            kind="mergesort",
        )
        indices = ordered.index.to_list()
        positive_count = len(indices)

        if positive_count == 1:
            train_indices = indices
            validation_indices: List[int] = []
            test_indices: List[int] = []
        else:
            proposed_test = max(
                1,
                int(math.ceil(test_fraction * positive_count)),
            )
            # Always preserve at least one positive for training.
            max_test = positive_count - 1
            test_count = min(proposed_test, max_test)

            test_indices = indices[-test_count:]
            remaining = indices[:-test_count]

            if len(remaining) >= 2 and validation_fraction_of_remaining > 0:
                proposed_validation = max(
                    1,
                    int(
                        math.ceil(
                            validation_fraction_of_remaining * len(remaining)
                        )
                    ),
                )
                validation_count = min(
                    proposed_validation,
                    len(remaining) - 1,
                )
            else:
                validation_count = 0

            if validation_count > 0:
                validation_indices = remaining[-validation_count:]
                train_indices = remaining[:-validation_count]
            else:
                validation_indices = []
                train_indices = remaining

        result.loc[train_indices, "split"] = "train"
        if validation_indices:
            result.loc[validation_indices, "split"] = "validation"
        if test_indices:
            result.loc[test_indices, "split"] = "test"

    if not (result.loc[result["split"] == "train", "rating"] >= positive_threshold).all():
        raise RuntimeError("Train split contains a non-positive interaction.")
    if not (result.loc[result["split"] == "validation", "rating"] >= positive_threshold).all():
        raise RuntimeError("Validation split contains a non-positive interaction.")
    if not (result.loc[result["split"] == "test", "rating"] >= positive_threshold).all():
        raise RuntimeError("Test split contains a non-positive interaction.")

    return result


def split_to_positive_sets(
    split_df: pd.DataFrame,
) -> Tuple[
    Dict[int, Set[int]],
    Dict[int, Set[int]],
    Dict[int, Set[int]],
]:
    def build(split_name: str) -> Dict[int, Set[int]]:
        subset = split_df[split_df["split"] == split_name]
        result: Dict[int, Set[int]] = {}
        for user_index, group in subset.groupby("u_idx", sort=True):
            result[int(user_index)] = set(
                group["i_idx"].astype(int).tolist()
            )
        return result

    return build("train"), build("validation"), build("test")


def build_all_observed_item_sets(
    split_df: pd.DataFrame,
) -> Dict[int, Set[int]]:
    """All rated items, including ratings below the relevance threshold."""
    result: Dict[int, Set[int]] = {}
    for user_index, group in split_df.groupby("u_idx", sort=True):
        result[int(user_index)] = set(
            group["i_idx"].astype(int).tolist()
        )
    return result


def build_all_positive_item_sets(
    split_df: pd.DataFrame,
    positive_threshold: float,
) -> Dict[int, Set[int]]:
    """All relevant items across train/validation/test."""
    positives = split_df[split_df["rating"] >= positive_threshold]
    result: Dict[int, Set[int]] = {}
    for user_index, group in positives.groupby("u_idx", sort=True):
        result[int(user_index)] = set(
            group["i_idx"].astype(int).tolist()
        )
    return result


# =============================================================================
# 4) CANONICAL DMF TRAINING MATRIX
# =============================================================================


def build_training_interaction_matrix(
    split_df: pd.DataFrame,
    number_of_users: int,
    number_of_items: int,
) -> torch.Tensor:
    """
    Binary positive training matrix:
        Y[u, i] = 1 when rating >= threshold and split == train
        Y[u, i] = 0 otherwise

    Validation and test positives are excluded to prevent leakage.
    Ratings below the relevance threshold are encoded as 0.
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


# =============================================================================
# 5) POINTWISE BCE SAMPLES
# =============================================================================


def sample_nonpositive_items(
    rng: np.random.RandomState,
    nonpositive_pool: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    if nonpositive_pool.size == 0:
        return np.empty(0, dtype=np.int64)
    replace = nonpositive_pool.size < sample_count
    return rng.choice(
        nonpositive_pool,
        size=sample_count,
        replace=replace,
    ).astype(np.int64)


def create_dmf_bce_samples(
    positive_items: Mapping[int, Set[int]],
    all_positive_items: Mapping[int, Set[int]],
    number_of_items: int,
    negative_ratio: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One positive target plus negative_ratio non-positive targets per positive.

    Held-out relevant items are excluded from negative sampling through
    all_positive_items. Items rated below the threshold may act as legitimate
    non-positive training alternatives, matching the old thresholded implicit
    formulation.
    """
    users: List[int] = []
    items: List[int] = []
    targets: List[float] = []
    all_item_indices = np.arange(number_of_items, dtype=np.int64)

    for user_index in sorted(positive_items):
        user_positive_items = sorted(positive_items[user_index])
        if not user_positive_items:
            continue

        all_relevant = np.fromiter(
            sorted(all_positive_items.get(user_index, set())),
            dtype=np.int64,
        )
        nonpositive_pool = np.setdiff1d(
            all_item_indices,
            all_relevant,
            assume_unique=True,
        )
        if nonpositive_pool.size == 0:
            continue

        rng = np.random.RandomState(seed + int(user_index))

        for item_index in user_positive_items:
            users.append(int(user_index))
            items.append(int(item_index))
            targets.append(1.0)

            negatives = sample_nonpositive_items(
                rng=rng,
                nonpositive_pool=nonpositive_pool,
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

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[index], self.items[index], self.targets[index]


# =============================================================================
# 6) CANONICAL TWO-TOWER DMF MODEL
# =============================================================================


class DMFTwoLayerTower(nn.Module):
    """
    DMF-2 projection:
        l1 = W1 x
        h  = ReLU(W2 l1 + b2)
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
        latent = F.relu(self.final_projection(first_layer))
        return latent


class CanonicalDMF(nn.Module):
    """
    User representation: p_u = f_U(Y[u, :])
    Item representation: q_i = f_I(Y[:, i])
    Score: cosine(p_u, q_i)
    """

    def __init__(
        self,
        training_interaction_matrix: torch.Tensor,
        first_projection_dimension: int,
        latent_dimension: int,
        initialization_std: float,
    ) -> None:
        super().__init__()

        if training_interaction_matrix.ndim != 2:
            raise ValueError(
                "training_interaction_matrix must have shape [users, items]."
            )

        number_of_users, number_of_items = (
            training_interaction_matrix.shape
        )
        self.number_of_users = int(number_of_users)
        self.number_of_items = int(number_of_items)

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
        representations = self.user_network(user_rows)
        return F.normalize(representations, p=2, dim=-1, eps=1e-12)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        item_columns = self.training_interaction_matrix[
            :, item_indices
        ].transpose(0, 1)
        representations = self.item_network(item_columns)
        return F.normalize(representations, p=2, dim=-1, eps=1e-12)

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        users = self.encode_users(user_indices)
        items = self.encode_items(item_indices)
        return torch.sum(users * items, dim=-1)


def binary_cross_entropy_cosine_loss(
    cosine_scores: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    probabilities = cosine_scores.clamp(
        min=epsilon,
        max=1.0 - epsilon,
    )
    return F.binary_cross_entropy(probabilities, targets)


# =============================================================================
# 7) METRICS AND REPRESENTATION HELPERS
# =============================================================================


def ndcg_binary(
    ranked_items: Sequence[int],
    positive_items: Set[int],
    k: int,
) -> float:
    top_k = list(ranked_items[:k])
    relevance = np.asarray(
        [1.0 if int(item) in positive_items else 0.0 for item in top_k],
        dtype=np.float64,
    )
    if relevance.sum() == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(relevance) + 2))
    dcg = float(np.sum(relevance / discounts))

    ideal_length = min(len(positive_items), k)
    if ideal_length == 0:
        return 0.0
    ideal_relevance = np.ones(ideal_length, dtype=np.float64)
    ideal_discounts = np.log2(np.arange(2, ideal_length + 2))
    idcg = float(np.sum(ideal_relevance / ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(
    ranked_items: Sequence[int],
    positive_items: Set[int],
    k: int,
) -> float:
    top_k = list(ranked_items[:k])
    if not positive_items:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(top_k, start=1):
        if int(item) in positive_items:
            hits += 1
            precision_sum += hits / rank

    denominator = min(len(positive_items), k)
    return precision_sum / denominator if denominator > 0 else 0.0


def evaluate_top_k_for_user(
    ranked_items: Sequence[int],
    positive_items: Set[int],
    k: int,
) -> Dict[str, float]:
    top_k = list(ranked_items[:k])
    hit_count = sum(
        1 for item in top_k if int(item) in positive_items
    )
    return {
        "HR": 1.0 if hit_count > 0 else 0.0,
        "NDCG": ndcg_binary(ranked_items, positive_items, k),
        "Recall": (
            hit_count / len(positive_items)
            if positive_items
            else 0.0
        ),
        "Precision": hit_count / k if k > 0 else 0.0,
        "MAP": average_precision_at_k(
            ranked_items,
            positive_items,
            k,
        ),
    }


@torch.no_grad()
def encode_all_users(
    model: CanonicalDMF,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    for start in range(0, model.number_of_users, batch_size):
        end = min(start + batch_size, model.number_of_users)
        indices = torch.arange(
            start,
            end,
            dtype=torch.long,
            device=device,
        )
        outputs.append(model.encode_users(indices).cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def encode_all_items(
    model: CanonicalDMF,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    for start in range(0, model.number_of_items, batch_size):
        end = min(start + batch_size, model.number_of_items)
        indices = torch.arange(
            start,
            end,
            dtype=torch.long,
            device=device,
        )
        outputs.append(model.encode_items(indices).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def rank_candidate_items(
    user_vector: np.ndarray,
    item_vectors: np.ndarray,
    candidate_items: Sequence[int],
) -> List[int]:
    candidates = np.asarray(candidate_items, dtype=np.int64)
    if candidates.size == 0:
        return []
    scores = item_vectors[candidates] @ user_vector
    order = np.lexsort((candidates, -scores))
    return candidates[order].astype(int).tolist()


def build_holdout_candidates(
    number_of_items: int,
    all_observed_items: Set[int],
    holdout_positive_items: Set[int],
) -> np.ndarray:
    """
    Candidate items are:
      - the held-out positive items; and
      - items never rated by the user.

    Every other known rated item is excluded.
    """
    excluded = set(all_observed_items)
    excluded.difference_update(holdout_positive_items)

    mask = np.ones(number_of_items, dtype=bool)
    if excluded:
        excluded_array = np.fromiter(
            sorted(excluded),
            dtype=np.int64,
        )
        mask[excluded_array] = False
    return np.arange(number_of_items, dtype=np.int64)[mask]


def aggregate_metric_rows(
    rows: List[Dict[str, float]],
    protocol_name: str,
    k_values: Sequence[int],
) -> pd.DataFrame:
    if not rows:
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

    detail = pd.DataFrame(rows)
    mean_metrics = (
        detail.groupby("k")[
            ["HR", "NDCG", "Recall", "Precision", "MAP"]
        ]
        .mean()
        .reindex(list(k_values))
        .reset_index()
    )
    mean_metrics.insert(0, "EvalProtocol", protocol_name)
    mean_metrics["users_eval"] = int(detail["user"].nunique())
    return mean_metrics


# =============================================================================
# 8) VALIDATION AND TRAINING
# =============================================================================


def evaluate_validation_all_items(
    model: CanonicalDMF,
    validation_items: Mapping[int, Set[int]],
    all_observed_items: Mapping[int, Set[int]],
    cfg: Config,
    device: str,
) -> Tuple[float, float]:
    """
    Return all-items validation NDCG@checkpoint_k and Recall@checkpoint_k.
    """
    user_vectors = encode_all_users(
        model,
        device,
        cfg.representation_batch_size,
    )
    item_vectors = encode_all_items(
        model,
        device,
        cfg.representation_batch_size,
    )

    ndcg_values: List[float] = []
    recall_values: List[float] = []

    for user_index in sorted(validation_items):
        positives = set(validation_items[user_index])
        if not positives:
            continue

        candidates = build_holdout_candidates(
            number_of_items=item_vectors.shape[0],
            all_observed_items=set(
                all_observed_items.get(user_index, set())
            ),
            holdout_positive_items=positives,
        )
        ranked = rank_candidate_items(
            user_vector=user_vectors[user_index],
            item_vectors=item_vectors,
            candidate_items=candidates,
        )
        metrics = evaluate_top_k_for_user(
            ranked,
            positives,
            cfg.checkpoint_k,
        )
        ndcg_values.append(metrics["NDCG"])
        recall_values.append(metrics["Recall"])

    if not ndcg_values:
        raise RuntimeError(
            "No validation users with positive held-out items were available."
        )

    return float(np.mean(ndcg_values)), float(np.mean(recall_values))


def train_dmf(
    model: CanonicalDMF,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    validation_items: Mapping[int, Set[int]],
    all_observed_items: Mapping[int, Set[int]],
    cfg: Config,
    device: str,
) -> Tuple[
    CanonicalDMF,
    List[Dict[str, float | int]],
    Dict[str, float | int],
]:
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
                scores,
                targets,
                cfg.numerical_epsilon,
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
                    scores,
                    targets,
                    cfg.numerical_epsilon,
                )
                validation_losses.append(float(loss.item()))

        train_loss = (
            float(np.mean(training_losses))
            if training_losses
            else float("nan")
        )
        validation_loss = (
            float(np.mean(validation_losses))
            if validation_losses
            else float("nan")
        )
        if not math.isfinite(validation_loss):
            raise FloatingPointError(
                "Validation loss became non-finite."
            )

        validation_ndcg, validation_recall = (
            evaluate_validation_all_items(
                model=model,
                validation_items=validation_items,
                all_observed_items=all_observed_items,
                cfg=cfg,
                device=device,
            )
        )

        history_row: Dict[str, float | int] = {
            "epoch": int(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            f"validation_NDCG@{cfg.checkpoint_k}": validation_ndcg,
            f"validation_Recall@{cfg.checkpoint_k}": validation_recall,
        }
        history.append(history_row)

        print(
            f"[DMF-2-BCE-IMP] epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_NDCG@{cfg.checkpoint_k}={validation_ndcg:.6f} "
            f"validation_Recall@{cfg.checkpoint_k}={validation_recall:.6f}"
        )

        improves_ndcg = validation_ndcg > best_ndcg + cfg.min_delta
        ties_ndcg = abs(validation_ndcg - best_ndcg) <= cfg.min_delta
        improves_recall = validation_recall > best_recall + cfg.min_delta
        ties_recall = abs(validation_recall - best_recall) <= cfg.min_delta
        improves_loss = (
            validation_loss < best_validation_loss - cfg.min_delta
        )

        is_better = (
            improves_ndcg
            or (
                ties_ndcg
                and (
                    improves_recall
                    or (
                        ties_recall
                        and improves_loss
                    )
                )
            )
        )

        if is_better:
            best_ndcg = validation_ndcg
            best_recall = validation_recall
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
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
                    "[DMF-2-BCE-IMP] Early stopping: no validation "
                    f"NDCG@{cfg.checkpoint_k} improvement for "
                    f"{cfg.patience} epochs."
                )
                break

    if best_state is None:
        raise RuntimeError("Training ended without a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    best_checkpoint: Dict[str, float | int] = {
        "epoch": int(best_epoch),
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


# =============================================================================
# 9) SAMPLED_99 AND TRUE ALL-ITEMS EVALUATION
# =============================================================================


def evaluate_sampled_99(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    test_items: Mapping[int, Set[int]],
    all_observed_items: Mapping[int, Set[int]],
    k_values: Sequence[int],
    negative_count: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    number_of_items = int(item_vectors.shape[0])
    all_items = np.arange(number_of_items, dtype=np.int64)

    for user_index in sorted(test_items):
        positives = set(test_items[user_index])
        if not positives:
            continue

        observed = np.fromiter(
            sorted(all_observed_items.get(user_index, set())),
            dtype=np.int64,
        )
        unseen_pool = np.setdiff1d(
            all_items,
            observed,
            assume_unique=True,
        )

        rng = np.random.RandomState(seed + int(user_index))
        effective_negative_count = min(
            negative_count,
            int(unseen_pool.size),
        )
        negatives = (
            rng.choice(
                unseen_pool,
                size=effective_negative_count,
                replace=False,
            )
            .astype(np.int64)
            .tolist()
            if effective_negative_count > 0
            else []
        )

        candidates = sorted(positives) + negatives
        ranked = rank_candidate_items(
            user_vector=user_vectors[user_index],
            item_vectors=item_vectors,
            candidate_items=candidates,
        )

        for k in k_values:
            metrics = evaluate_top_k_for_user(
                ranked,
                positives,
                int(k),
            )
            rows.append(
                {
                    "user": int(user_index),
                    "k": int(k),
                    **metrics,
                }
            )

    output = aggregate_metric_rows(
        rows=rows,
        protocol_name=f"sampled_{negative_count}_multi_positive",
        k_values=k_values,
    )
    if not output.empty:
        output["N_NEG_CANDIDATES_requested"] = int(negative_count)
        output["multi_positive"] = True
    return output


def build_all_items_recommendations(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    test_items: Mapping[int, Set[int]],
    all_observed_items: Mapping[int, Set[int]],
    top_n: int,
) -> Dict[int, List[int]]:
    recommendations: Dict[int, List[int]] = {}

    for user_index in range(user_vectors.shape[0]):
        positives = set(test_items.get(user_index, set()))
        candidates = build_holdout_candidates(
            number_of_items=item_vectors.shape[0],
            all_observed_items=set(
                all_observed_items.get(user_index, set())
            ),
            holdout_positive_items=positives,
        )
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

    for user_index in sorted(test_items):
        positives = set(test_items[user_index])
        if not positives:
            continue

        ranked = list(recommendations.get(user_index, []))
        for k in k_values:
            metrics = evaluate_top_k_for_user(
                ranked,
                positives,
                int(k),
            )
            rows.append(
                {
                    "user": int(user_index),
                    "k": int(k),
                    **metrics,
                }
            )

    return aggregate_metric_rows(
        rows=rows,
        protocol_name="all_items_multi_positive",
        k_values=k_values,
    )


# =============================================================================
# 10) OUTPUT HELPERS
# =============================================================================


def save_id_mappings(
    run_dir: str,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
) -> Tuple[str, str]:
    user_mapping_path = os.path.join(
        run_dir,
        "user_id_mapping.csv",
    )
    item_mapping_path = os.path.join(
        run_dir,
        "item_id_mapping.csv",
    )

    pd.DataFrame(
        {
            "u_idx": np.arange(len(user_ids), dtype=np.int64),
            "userID": user_ids,
        }
    ).to_csv(
        user_mapping_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "i_idx": np.arange(len(item_ids), dtype=np.int64),
            "itemID": item_ids,
        }
    ).to_csv(
        item_mapping_path,
        index=False,
        encoding="utf-8-sig",
    )

    return user_mapping_path, item_mapping_path


def build_test_rating_lookup(
    split_df: pd.DataFrame,
) -> Dict[Tuple[int, int], float]:
    subset = split_df[split_df["split"] == "test"]
    return {
        (int(row.u_idx), int(row.i_idx)): float(row.rating)
        for row in subset.itertuples(index=False)
    }


def save_recommendations(
    recommendations: Mapping[int, Sequence[int]],
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    test_rating_lookup: Mapping[Tuple[int, int], float],
    output_path: str,
) -> None:
    rows: List[Dict[str, float | int]] = []

    for user_index, ranked_items in recommendations.items():
        for rank, item_index in enumerate(ranked_items, start=1):
            score = float(
                item_vectors[int(item_index)] @ user_vectors[int(user_index)]
            )
            rows.append(
                {
                    "u_idx": int(user_index),
                    "userID": int(user_ids[user_index]),
                    "i_idx": int(item_index),
                    "itemID": int(item_ids[item_index]),
                    "rank": int(rank),
                    "PredScore": score,
                    "TrueRating_Test": test_rating_lookup.get(
                        (int(user_index), int(item_index)),
                        np.nan,
                    ),
                    "is_positive_in_test": int(
                        (int(user_index), int(item_index))
                        in test_rating_lookup
                    ),
                }
            )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# 11) MAIN
# =============================================================================


def main() -> None:
    validate_config(CFG)
    set_all_seeds(CFG.seed)

    version_tag = build_version_tag(CFG)
    run_dir = build_run_dir(CFG, version_tag)
    ensure_dir(run_dir)

    print(f"[DEVICE] {DEVICE}")
    print(f"[INPUT] {CFG.ratings_path}")
    print(f"[VERSION] {version_tag}")

    df = read_douban_ratings(CFG.ratings_path)
    print(
        f"[RAW] rows={len(df)} "
        f"users={df['userID'].nunique()} "
        f"items={df['itemID'].nunique()} "
        f"rating_min={df['rating'].min():.4f} "
        f"rating_max={df['rating'].max():.4f}"
    )

    df_filtered = filter_users(
        df,
        CFG.min_user_interactions,
    )
    if df_filtered.empty:
        raise RuntimeError(
            "No users remain after minimum-interaction filtering."
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

    split_df = split_positive_last_n(
        df_encoded=df_encoded,
        positive_threshold=CFG.positive_threshold,
        test_fraction=CFG.test_fraction,
        validation_fraction_of_remaining=(
            CFG.validation_fraction_of_remaining
        ),
    )

    train_items, validation_items, test_items = (
        split_to_positive_sets(split_df)
    )
    all_observed_items = build_all_observed_item_sets(split_df)
    all_positive_items = build_all_positive_item_sets(
        split_df,
        CFG.positive_threshold,
    )

    split_counts = split_df["split"].value_counts()
    print(
        f"[SPLIT] train_positive={int(split_counts.get('train', 0))} "
        f"validation_positive={int(split_counts.get('validation', 0))} "
        f"test_positive={int(split_counts.get('test', 0))} "
        f"known_nonpositive={int(split_counts.get('known_nonpositive', 0))}"
    )
    print(
        f"[EVAL USERS] validation={sum(bool(v) for v in validation_items.values())} "
        f"test={sum(bool(v) for v in test_items.values())}"
    )

    training_matrix = build_training_interaction_matrix(
        split_df=split_df,
        number_of_users=number_of_users,
        number_of_items=number_of_items,
    )

    train_users, train_sample_items, train_targets = (
        create_dmf_bce_samples(
            positive_items=train_items,
            all_positive_items=all_positive_items,
            number_of_items=number_of_items,
            negative_ratio=CFG.negative_ratio,
            seed=CFG.seed,
        )
    )
    (
        validation_users,
        validation_sample_items,
        validation_targets,
    ) = create_dmf_bce_samples(
        positive_items=validation_items,
        all_positive_items=all_positive_items,
        number_of_items=number_of_items,
        negative_ratio=CFG.negative_ratio,
        seed=CFG.seed + 1_000_000,
    )

    if len(train_targets) == 0:
        raise RuntimeError("No training samples were generated.")
    if len(validation_targets) == 0:
        raise RuntimeError(
            "No validation samples were generated. "
            "Check the positive threshold and split."
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

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(
        f"[MODEL] users={number_of_users} "
        f"items={number_of_items} "
        f"parameters={parameter_count:,} "
        f"user_input_dim={number_of_items} "
        f"item_input_dim={number_of_users}"
    )

    model, history, best_checkpoint = train_dmf(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        validation_items=validation_items,
        all_observed_items=all_observed_items,
        cfg=CFG,
        device=DEVICE,
    )

    print("[EVAL] Precomputing user and item latent representations...")
    user_vectors = encode_all_users(
        model,
        DEVICE,
        CFG.representation_batch_size,
    )
    item_vectors = encode_all_items(
        model,
        DEVICE,
        CFG.representation_batch_size,
    )

    print(
        f"[EVAL] sampled_{CFG.sampled_negative_count} "
        "multi-positive protocol"
    )
    sampled_metrics = evaluate_sampled_99(
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        test_items=test_items,
        all_observed_items=all_observed_items,
        k_values=CFG.k_values,
        negative_count=CFG.sampled_negative_count,
        seed=CFG.seed,
    )
    print(sampled_metrics)

    print(
        f"[RECS] Building true all-items Top-{CFG.topn_all_items} "
        "recommendations..."
    )
    recommendations = build_all_items_recommendations(
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        test_items=test_items,
        all_observed_items=all_observed_items,
        top_n=CFG.topn_all_items,
    )
    all_items_metrics = evaluate_all_items(
        recommendations=recommendations,
        test_items=test_items,
        k_values=CFG.k_values,
    )
    print(all_items_metrics)

    variant_name = (
        f"DMF_GLOBAL_{CFG.dataset_tag}_DMF2_BCEIMP_{version_tag}"
    )
    sampled_output = sampled_metrics.copy()
    all_items_output = all_items_metrics.copy()
    sampled_output.insert(1, "Variant", variant_name)
    all_items_output.insert(1, "Variant", variant_name)
    combined_metrics = pd.concat(
        [sampled_output, all_items_output],
        ignore_index=True,
        sort=False,
    )

    # -------------------------------------------------------------------------
    # Save reproducibility artifacts.
    # -------------------------------------------------------------------------
    split_path = os.path.join(
        run_dir,
        f"{CFG.dataset_tag}_split_{version_tag}.csv",
    )
    split_df.to_csv(
        split_path,
        index=False,
        encoding="utf-8-sig",
    )

    user_mapping_path, item_mapping_path = save_id_mappings(
        run_dir,
        user_ids,
        item_ids,
    )

    recommendations_path = os.path.join(
        run_dir,
        f"{CFG.dataset_tag}_DMF_GLOBAL_top"
        f"{CFG.topn_all_items}_{version_tag}.csv",
    )
    test_rating_lookup = build_test_rating_lookup(split_df)
    save_recommendations(
        recommendations=recommendations,
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        user_ids=user_ids,
        item_ids=item_ids,
        test_rating_lookup=test_rating_lookup,
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
            "best_checkpoint": best_checkpoint,
        },
        model_path,
    )

    metadata = {
        **asdict(CFG),
        "device": DEVICE,
        "version_tag": version_tag,
        "number_of_users": number_of_users,
        "number_of_items": number_of_items,
        "number_of_filtered_rows": int(len(df_encoded)),
        "model_parameter_count": int(parameter_count),
        "positive_interactions": int(
            (df_encoded["rating"] >= CFG.positive_threshold).sum()
        ),
        "split_counts": {
            str(key): int(value)
            for key, value in split_counts.to_dict().items()
        },
        "evaluation_users": {
            "validation": int(
                sum(bool(value) for value in validation_items.values())
            ),
            "test": int(
                sum(bool(value) for value in test_items.values())
            ),
        },
        "checkpoint_selection": {
            "criterion": (
                f"all-items validation NDCG@{CFG.checkpoint_k}"
            ),
            "tie_break_1": (
                f"all-items validation Recall@{CFG.checkpoint_k}"
            ),
            "tie_break_2": "validation BCE loss",
            "test_metrics_used_for_selection": False,
            **best_checkpoint,
        },
        "architecture": {
            "user_input": "positive training-matrix row Y[u, :]",
            "item_input": "positive training-matrix column Y[:, i]",
            "user_network": "two-layer DMF projection",
            "item_network": "two-layer DMF projection",
            "score": "cosine similarity",
            "interaction_matrix": (
                "binary rating>=4 train positive=1; otherwise=0"
            ),
            "loss": (
                "binary cross-entropy with thresholded positive=1 "
                "and sampled non-positive=0"
            ),
        },
        "candidate_policy": {
            "sampled_99_negatives": (
                "truly unseen items only"
            ),
            "all_items": (
                "all never-rated items plus held-out test positives; "
                "all other rated items excluded"
            ),
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
            f"[WARNING] Summary file is open or locked; skipped: "
            f"{summary_path}"
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
