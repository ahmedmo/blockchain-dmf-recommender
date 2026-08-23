# -*- coding: utf-8 -*-
"""
Canonical CC-DMF for Douban centralized/real community files — V1
==============================================

This script converts the former per-community ID-embedding + concatenation + MLP
workflow into a genuine two-tower Deep Matrix Factorization (DMF) workflow while
retaining the Douban community files and the compatible parts of the former
experimental protocol.

Canonical DMF backbone
----------------------
1. Each community is trained independently.
2. The user input is row Y_k[u, :] of that community's TRAINING matrix.
3. The item input is column Y_k[:, i] of that community's TRAINING matrix.
4. Separate two-layer user and item towers produce latent representations.
5. The recommendation score is cosine similarity.
6. Training uses pointwise BCE: observed training positives=1 and sampled
   non-positive/unobserved alternatives=0.
7. Validation and test positives are excluded from the DMF input matrix.

Douban CC-DMF settings matched exactly to the finalized DC-DMF V7 configuration
--------------------------------------------------------------------------
- Six centralized/real-community files from the former MLP/BPR code.
- Positive threshold for training: rating >= 3.
- Positive threshold for testing: rating >= 4.
- Last-N test fraction: 20% of each user's rating>=4 interactions.
- Validation fraction: 15% of the remaining rating>=3 interactions.
- Negative ratio: 1, matched exactly to the finalized DC-DMF V7 setting.
- P=128 and Z=128, matched exactly to the finalized DC-DMF V7 settings.
- Batch size 256, learning rate 1e-3, weight decay 1e-5, 80 epochs,
  patience 8.
- Deterministic multi-positive sampled evaluation using 20 negatives, exactly
  matching the effective N_NEG_CANDIDATES value in the former Douban community
  script (despite its older sampled_99 function name).
- K={5,10,20}.

Two test protocols are reported
-------------------------------
1. sampled_20_multi_positive: the primary legacy-compatible comparison.
2. all_community_items_multi_positive: a stricter diagnostic ranking against
   all eligible items within each community.

The best checkpoint is selected from VALIDATION data only:
1. highest sampled validation NDCG@10;
2. highest sampled validation Recall@10 on a tie;
3. lowest validation BCE loss on a full tie.

CC-DMF code versioning starts independently from V1. This V1 keeps the same model, split, training, checkpointing, and evaluation settings as DC-DMF V7; only the community membership files and CC-specific output labels differ.
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


# =============================================================================
# 0) CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class Config:
    # Increment whenever this Douban-community DMF code is modified.
    code_version: str = "V1"

    dataset_name: str = "Douban"
    dataset_tag: str = "Douban"

    community_files: Tuple[str, ...] = (
        r"data/douban/communities/cc/comm_U_89_I_1573_comm_11.txt",
        r"data/douban/communities/cc/comm_U_63_I_1233_comm_25.txt",
        r"data/douban/communities/cc/comm_U_62_I_1375_comm_14.txt",
        r"data/douban/communities/cc/comm_U_61_I_1228_comm_5.txt",
        r"data/douban/communities/cc/comm_U_57_I_1240_comm_2.txt",
        r"data/douban/communities/cc/comm_U_55_I_1344_comm_13.txt",
    )

    output_root: str = (
        r"outputs/douban"
        r"\Canonical_CC_DMF_Douban_Communities_V1"
    )

    seed: int = 42
    min_user_interactions: int = 4

    # Retained from the former Douban community MLP/BPR code.
    train_positive_threshold: float = 3.0
    test_positive_threshold: float = 4.0
    test_fraction: float = 0.20
    validation_fraction_of_remaining: float = 0.15

    negative_ratio: int = 1
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 80
    patience: int = 8
    min_delta: float = 1e-6

    # Matched exactly to finalized DC-DMF V7 capacity settings.
    first_projection_dim: int = 128
    latent_dim: int = 128
    init_std: float = 0.01
    numerical_epsilon: float = 1e-6

    # The old script's effective value was 20, although its function retained
    # the historical sampled_99 name.
    sampled_negative_count: int = 20
    k_values: Tuple[int, ...] = (5, 10, 20)
    checkpoint_k: int = 10

    # Also report the stricter all-community-items protocol.
    evaluate_all_community_items: bool = True
    all_items_k_values: Tuple[int, ...] = (5, 10, 20)

    representation_batch_size: int = 2048
    num_workers: int = 0
    pin_memory: bool = True

    save_per_user_recommendations: bool = True
    save_sampled_debug_scores: bool = True
    debug_users: Optional[Tuple[int, ...]] = None  # original IDs; None = all


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
    return f"{value:.10g}".replace(".", "d").replace("-", "m")


def community_tag_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    # Supports both naming schemes:
    #   003_comm_8_U104_I3510 -> 8
    #   comm_U_89_I_1573_comm_11 -> 11
    parts = stem.split("_")
    candidate_positions = [
        idx for idx, token in enumerate(parts[:-1]) if token.lower() == "comm"
    ]
    for position in reversed(candidate_positions):
        next_token = parts[position + 1]
        if str(next_token).isdigit():
            return str(next_token)
    return stem


def build_base_version_tag(cfg: Config) -> str:
    return (
        f"{cfg.code_version}_CCDMF_DMF2_BCEIMP_DBNCCCOMM_LASTN_"
        f"TF{format_float_tag(cfg.test_fraction)}_"
        f"VF{format_float_tag(cfg.validation_fraction_of_remaining)}_"
        f"POS{format_float_tag(cfg.train_positive_threshold)}train"
        f"{format_float_tag(cfg.test_positive_threshold)}test_"
        f"NEG{cfg.negative_ratio}_P{cfg.first_projection_dim}_Z{cfg.latent_dim}_"
        f"LR{format_float_tag(cfg.learning_rate)}_"
        f"WD{format_float_tag(cfg.weight_decay)}_BS{cfg.batch_size}_"
        f"SELNDCG{cfg.checkpoint_k}_NNEG{cfg.sampled_negative_count}_SEED{cfg.seed}"
    )


def build_community_run_dir(
    cfg: Config,
    community_tag: str,
    version_tag: str,
) -> str:
    return os.path.join(
        cfg.output_root,
        f"COMM_{community_tag}",
        f"RUN_{version_tag}",
    )


def validate_config(cfg: Config) -> None:
    if not cfg.community_files:
        raise ValueError("community_files must not be empty.")
    if cfg.min_user_interactions < 2:
        raise ValueError("min_user_interactions must be at least 2.")
    if cfg.train_positive_threshold > cfg.test_positive_threshold:
        raise ValueError(
            "train_positive_threshold must not exceed test_positive_threshold."
        )
    if not (0.0 < cfg.test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0 and 1.")
    if not (0.0 <= cfg.validation_fraction_of_remaining < 1.0):
        raise ValueError("validation_fraction_of_remaining must be in [0,1).")
    if cfg.negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1.")
    if cfg.sampled_negative_count < 1:
        raise ValueError("sampled_negative_count must be at least 1.")
    if cfg.first_projection_dim < 1 or cfg.latent_dim < 1:
        raise ValueError("DMF dimensions must be positive.")
    if cfg.checkpoint_k not in cfg.k_values:
        raise ValueError("checkpoint_k must be included in k_values.")
    if max(cfg.k_values) > cfg.sampled_negative_count + 1_000_000:
        raise ValueError("Invalid K configuration.")


# =============================================================================
# 2) DATA READING, FILTERING, AND ENCODING
# =============================================================================


def read_ratings_file(path: str) -> pd.DataFrame:
    """Read whitespace/comma separated user, item, rating rows robustly."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Community ratings file does not exist: {path}")

    raw = pd.read_csv(path, sep=r"[\s,]+", header=None, engine="python")
    if raw.shape[1] < 3:
        raise ValueError(
            "Each community file must contain at least user, item, and rating."
        )

    dataframe = raw.iloc[:, :3].copy()
    dataframe.columns = ["userID", "itemID", "rating"]
    dataframe["source_order"] = np.arange(len(dataframe), dtype=np.int64)

    for column in ["userID", "itemID", "rating"]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    before = len(dataframe)
    dataframe = dataframe.dropna(subset=["userID", "itemID", "rating"]).copy()
    dropped = before - len(dataframe)
    if dropped:
        print(f"[WARNING] Dropped {dropped} non-numeric/header rows.")

    dataframe["userID"] = dataframe["userID"].astype(np.int64)
    dataframe["itemID"] = dataframe["itemID"].astype(np.int64)
    dataframe["rating"] = dataframe["rating"].astype(np.float32)

    if not np.isfinite(dataframe["rating"].to_numpy()).all():
        raise ValueError("rating contains NaN or infinite values.")

    # Canonical matrix construction requires one value per user-item pair.
    if dataframe.duplicated(["userID", "itemID"]).any():
        duplicate_rows = int(
            dataframe.duplicated(["userID", "itemID"], keep=False).sum()
        )
        print(
            f"[WARNING] Found {duplicate_rows} rows belonging to duplicate "
            "user-item pairs; retaining maximum rating and earliest order."
        )
        dataframe = (
            dataframe.groupby(["userID", "itemID"], as_index=False)
            .agg(rating=("rating", "max"), source_order=("source_order", "min"))
            .sort_values("source_order", kind="mergesort")
            .reset_index(drop=True)
        )

    return dataframe.reset_index(drop=True)


def filter_users_by_total_interactions(
    dataframe: pd.DataFrame,
    minimum_interactions: int,
) -> pd.DataFrame:
    counts = dataframe.groupby("userID").size()
    retained = counts[counts >= minimum_interactions].index
    return dataframe[dataframe["userID"].isin(retained)].copy().reset_index(drop=True)


def encode_ids(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int], np.ndarray, np.ndarray]:
    user_ids = np.array(sorted(dataframe["userID"].unique()), dtype=np.int64)
    item_ids = np.array(sorted(dataframe["itemID"].unique()), dtype=np.int64)
    user_to_index = {int(value): index for index, value in enumerate(user_ids)}
    item_to_index = {int(value): index for index, value in enumerate(item_ids)}

    encoded = dataframe.copy()
    encoded["u_idx"] = encoded["userID"].map(user_to_index).astype(np.int64)
    encoded["i_idx"] = encoded["itemID"].map(item_to_index).astype(np.int64)
    return encoded, user_to_index, item_to_index, user_ids, item_ids


# =============================================================================
# 3) DOUBAN LAST-N MULTI-POSITIVE SPLIT
# =============================================================================


def split_douban_community(
    encoded: pd.DataFrame,
    train_positive_threshold: float,
    test_positive_threshold: float,
    test_fraction: float,
    validation_fraction_of_remaining: float,
    seed: int,
) -> pd.DataFrame:
    """
    Materialize a leakage-safe split close to the former Douban community code.

    Per user:
    - test: the last ceil(test_fraction * count) interactions with rating>=4;
    - validation: deterministic 15% of the remaining rating>=3 positives;
    - train: all remaining rating>=3 positives;
    - known_nonpositive: ratings below the training threshold.

    A test-positive user is retained only when at least one rating>=3 interaction
    can remain for training. This avoids evaluating a zero-information user row.
    """
    parts: List[pd.DataFrame] = []

    for user_index, user_rows in encoded.groupby("u_idx", sort=True):
        user_rows = user_rows.copy().sort_values(
            ["source_order", "i_idx"], kind="mergesort"
        )
        user_rows["split"] = "known_nonpositive"

        train_positive_positions = user_rows.index[
            user_rows["rating"] >= train_positive_threshold
        ].to_numpy(dtype=np.int64)
        test_eligible_positions = user_rows.index[
            user_rows["rating"] >= test_positive_threshold
        ].to_numpy(dtype=np.int64)

        test_positions = np.empty(0, dtype=np.int64)
        if len(train_positive_positions) >= 2 and len(test_eligible_positions) >= 1:
            requested_test_count = max(
                1,
                int(math.ceil(test_fraction * len(test_eligible_positions))),
            )
            # Preserve at least one rating>=3 training positive.
            maximum_test_count = min(
                len(test_eligible_positions),
                len(train_positive_positions) - 1,
            )
            test_count = min(requested_test_count, maximum_test_count)
            if test_count > 0:
                test_positions = test_eligible_positions[-test_count:]

        remaining_positive_positions = np.setdiff1d(
            train_positive_positions,
            test_positions,
            assume_unique=False,
        )

        validation_positions = np.empty(0, dtype=np.int64)
        train_positions = remaining_positive_positions.copy()
        if len(remaining_positive_positions) >= 2:
            validation_count = max(
                1,
                int(
                    math.ceil(
                        validation_fraction_of_remaining
                        * len(remaining_positive_positions)
                    )
                ),
            )
            validation_count = min(
                validation_count,
                len(remaining_positive_positions) - 1,
            )
            rng = np.random.RandomState(seed + int(user_index))
            permutation = rng.permutation(remaining_positive_positions)
            validation_positions = permutation[:validation_count]
            train_positions = permutation[validation_count:]

        if len(train_positions):
            user_rows.loc[train_positions, "split"] = "train"
        if len(validation_positions):
            user_rows.loc[validation_positions, "split"] = "validation"
        if len(test_positions):
            user_rows.loc[test_positions, "split"] = "test"

        parts.append(user_rows)

    split_df = pd.concat(parts, ignore_index=True)

    # Safety checks for every user who is actually evaluated.
    evaluated_users = set(
        split_df.loc[split_df["split"] == "test", "u_idx"].astype(int).tolist()
    )
    train_users = set(
        split_df.loc[split_df["split"] == "train", "u_idx"].astype(int).tolist()
    )
    if not evaluated_users.issubset(train_users):
        missing = sorted(evaluated_users - train_users)
        raise RuntimeError(
            f"Test users without a training positive were found: {missing[:10]}"
        )

    return split_df


def item_sets_for_split(
    split_df: pd.DataFrame,
    split_name: str,
) -> Dict[int, Set[int]]:
    all_users = sorted(split_df["u_idx"].astype(int).unique().tolist())
    result: Dict[int, Set[int]] = {user: set() for user in all_users}
    subset = split_df[split_df["split"] == split_name]
    for user_index, group in subset.groupby("u_idx", sort=True):
        result[int(user_index)] = set(group["i_idx"].astype(int).tolist())
    return result


def build_positive_items_all(split_df: pd.DataFrame) -> Dict[int, Set[int]]:
    positive = split_df[split_df["split"].isin(["train", "validation", "test"])]
    result: Dict[int, Set[int]] = {}
    for user_index, group in positive.groupby("u_idx", sort=True):
        result[int(user_index)] = set(group["i_idx"].astype(int).tolist())
    return result


def build_known_items_all(split_df: pd.DataFrame) -> Dict[int, Set[int]]:
    result: Dict[int, Set[int]] = {}
    for user_index, group in split_df.groupby("u_idx", sort=True):
        result[int(user_index)] = set(group["i_idx"].astype(int).tolist())
    return result


# =============================================================================
# 4) COMMUNITY TRAINING INTERACTION MATRIX
# =============================================================================


def build_training_interaction_matrix(
    split_df: pd.DataFrame,
    number_of_users: int,
    number_of_items: int,
) -> torch.Tensor:
    matrix = np.zeros((number_of_users, number_of_items), dtype=np.float32)
    train_rows = split_df[split_df["split"] == "train"]
    matrix[
        train_rows["u_idx"].to_numpy(dtype=np.int64),
        train_rows["i_idx"].to_numpy(dtype=np.int64),
    ] = 1.0
    return torch.from_numpy(matrix)


# =============================================================================
# 5) POINTWISE BCE SAMPLES
# =============================================================================


def sample_items(
    rng: np.random.RandomState,
    pool: np.ndarray,
    count: int,
) -> np.ndarray:
    if pool.size == 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(pool, size=count, replace=pool.size < count).astype(np.int64)


def create_bce_implicit_samples(
    positive_items: Mapping[int, Set[int]],
    positive_items_all: Mapping[int, Set[int]],
    number_of_items: int,
    negative_ratio: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    users: List[int] = []
    items: List[int] = []
    targets: List[float] = []
    all_items = np.arange(number_of_items, dtype=np.int64)

    for user_index in sorted(positive_items):
        positives = sorted(positive_items.get(user_index, set()))
        if not positives:
            continue

        excluded_positives = np.fromiter(
            sorted(positive_items_all.get(user_index, set())),
            dtype=np.int64,
        )
        negative_pool = np.setdiff1d(
            all_items,
            excluded_positives,
            assume_unique=True,
        )
        if negative_pool.size == 0:
            continue

        rng = np.random.RandomState(seed + int(user_index))
        for positive_item in positives:
            users.append(int(user_index))
            items.append(int(positive_item))
            targets.append(1.0)

            negatives = sample_items(rng, negative_pool, negative_ratio)
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
# 6) CANONICAL TWO-TOWER DMF
# =============================================================================


class DMFTwoLayerTower(nn.Module):
    """DMF-2: linear first projection, then ReLU final latent layer."""

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
        return F.relu(self.final_projection(first_layer))


class CanonicalDMF(nn.Module):
    def __init__(
        self,
        training_interaction_matrix: torch.Tensor,
        first_projection_dimension: int,
        latent_dimension: int,
        initialization_std: float,
    ) -> None:
        super().__init__()
        if training_interaction_matrix.ndim != 2:
            raise ValueError("training_interaction_matrix must be two-dimensional.")

        matrix = training_interaction_matrix.float()
        number_of_users, number_of_items = matrix.shape
        self.number_of_users = int(number_of_users)
        self.number_of_items = int(number_of_items)
        self.register_buffer(
            "training_interaction_matrix",
            matrix,
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
        rows = self.training_interaction_matrix[user_indices, :]
        vectors = self.user_network(rows)
        return F.normalize(vectors, p=2, dim=-1, eps=1e-12)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        columns = self.training_interaction_matrix[:, item_indices].transpose(0, 1)
        vectors = self.item_network(columns)
        return F.normalize(vectors, p=2, dim=-1, eps=1e-12)

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.encode_users(user_indices)
        item_vectors = self.encode_items(item_indices)
        return torch.sum(user_vectors * item_vectors, dim=-1)


def binary_cross_entropy_cosine_loss(
    cosine_scores: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    probabilities = cosine_scores.clamp(min=epsilon, max=1.0 - epsilon)
    return F.binary_cross_entropy(probabilities, targets)


# =============================================================================
# 7) METRICS AND RANKING
# =============================================================================


def dcg_at_positions(one_based_positions: Iterable[int]) -> float:
    return float(
        sum(1.0 / math.log2(position + 1) for position in one_based_positions)
    )


def ndcg_binary(top_k_items: Sequence[int], positive_items: Set[int]) -> float:
    positions = [
        rank
        for rank, item in enumerate(top_k_items, start=1)
        if item in positive_items
    ]
    if not positions:
        return 0.0
    dcg = dcg_at_positions(positions)
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
    for rank, item in enumerate(top_k_items, start=1):
        if item in positive_items:
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
    hits = sum(item in positive_items for item in selected)
    recall = float(hits / len(positive_items) if positive_items else 0.0)
    precision = float(hits / k if k > 0 else 0.0)
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if precision + recall > 0
        else 0.0
    )
    return {
        "HR": float(1.0 if hits > 0 else 0.0),
        "NDCG": float(ndcg_binary(selected, positive_items)),
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "MAP": float(average_precision_at_k(selected, positive_items)),
    }


@torch.no_grad()
def encode_all_users(
    model: CanonicalDMF,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    batches: List[np.ndarray] = []
    for start in range(0, model.number_of_users, batch_size):
        stop = min(start + batch_size, model.number_of_users)
        indices = torch.arange(start, stop, dtype=torch.long, device=device)
        batches.append(model.encode_users(indices).cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0)


@torch.no_grad()
def encode_all_items(
    model: CanonicalDMF,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    batches: List[np.ndarray] = []
    for start in range(0, model.number_of_items, batch_size):
        stop = min(start + batch_size, model.number_of_items)
        indices = torch.arange(start, stop, dtype=torch.long, device=device)
        batches.append(model.encode_items(indices).cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0)


def rank_candidate_items_with_scores(
    user_vector: np.ndarray,
    item_vectors: np.ndarray,
    candidate_items: Sequence[int],
) -> Tuple[List[int], Dict[int, float]]:
    candidates = np.asarray(candidate_items, dtype=np.int64)
    if candidates.size == 0:
        return [], {}
    scores = item_vectors[candidates] @ user_vector
    order = np.lexsort((candidates, -scores))
    ranked = candidates[order].astype(int).tolist()
    score_map = {
        int(item): float(score)
        for item, score in zip(candidates, scores)
    }
    return ranked, score_map


# =============================================================================
# 8) DETERMINISTIC SAMPLED CANDIDATES
# =============================================================================


def build_sampled_candidate_sets(
    positive_items: Mapping[int, Set[int]],
    positive_items_all: Mapping[int, Set[int]],
    number_of_items: int,
    negative_count: int,
    seed: int,
) -> Dict[int, Tuple[List[int], Set[int], int]]:
    all_items = np.arange(number_of_items, dtype=np.int64)
    result: Dict[int, Tuple[List[int], Set[int], int]] = {}

    for user_index in sorted(positive_items):
        positives = set(positive_items.get(user_index, set()))
        if not positives:
            continue

        excluded_positives = np.fromiter(
            sorted(positive_items_all.get(user_index, set())),
            dtype=np.int64,
        )
        negative_pool = np.setdiff1d(
            all_items,
            excluded_positives,
            assume_unique=True,
        )
        effective_count = min(negative_count, int(negative_pool.size))
        rng = np.random.RandomState(seed + int(user_index))
        negatives = (
            rng.choice(negative_pool, size=effective_count, replace=False)
            .astype(np.int64)
            .tolist()
            if effective_count > 0
            else []
        )
        candidates = sorted(positives) + [int(value) for value in negatives]
        result[int(user_index)] = (candidates, positives, effective_count)

    return result


def evaluate_fixed_sampled_candidates(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    candidate_sets: Mapping[int, Tuple[List[int], Set[int], int]],
    k_values: Sequence[int],
    protocol_name: str,
    negative_count_requested: int,
    include_details: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, float | int]] = []
    detail_rows: List[Dict[str, float | int]] = []

    for user_index in sorted(candidate_sets):
        candidates, positives, effective_negative_count = candidate_sets[user_index]
        ranked, score_map = rank_candidate_items_with_scores(
            user_vector=user_vectors[user_index],
            item_vectors=item_vectors,
            candidate_items=candidates,
        )

        for k in k_values:
            metric_rows.append(
                {
                    "user": int(user_index),
                    "k": int(k),
                    **evaluate_top_k_for_user(ranked, positives, int(k)),
                }
            )

        if include_details:
            for rank, item_index in enumerate(ranked, start=1):
                detail_rows.append(
                    {
                        "user": int(user_index),
                        "rank_sampled": int(rank),
                        "i_idx": int(item_index),
                        "score": float(score_map[int(item_index)]),
                        "is_positive_in_test": int(item_index in positives),
                        "positive_count": int(len(positives)),
                        "negative_count_effective": int(effective_negative_count),
                        "candidate_count": int(len(candidates)),
                    }
                )

    detail = pd.DataFrame(metric_rows)
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()

    metric_columns = ["HR", "NDCG", "Recall", "Precision", "F1", "MAP"]
    mean_metrics = detail.groupby("k")[metric_columns].mean().reset_index()
    mean_metrics.insert(0, "EvalProtocol", protocol_name)
    mean_metrics["N_NEG_CANDIDATES_requested"] = int(negative_count_requested)
    mean_metrics["multi_positive"] = True
    mean_metrics["users_eval"] = int(detail["user"].nunique())
    return mean_metrics, pd.DataFrame(detail_rows)


# =============================================================================
# 9) TRAINING WITH VALIDATION NDCG CHECKPOINTING
# =============================================================================


def train_dmf_community(
    model: CanonicalDMF,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    validation_candidate_sets: Mapping[int, Tuple[List[int], Set[int], int]],
    cfg: Config,
    device: str,
) -> Tuple[CanonicalDMF, List[Dict[str, float | int]], Dict[str, float | int]]:
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
        train_losses: List[float] = []
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
            train_losses.append(float(loss.item()))

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

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        validation_loss = (
            float(np.mean(validation_losses))
            if validation_losses
            else train_loss
        )

        user_vectors = encode_all_users(model, device, cfg.representation_batch_size)
        item_vectors = encode_all_items(model, device, cfg.representation_batch_size)
        validation_metrics, _ = evaluate_fixed_sampled_candidates(
            user_vectors=user_vectors,
            item_vectors=item_vectors,
            candidate_sets=validation_candidate_sets,
            k_values=(cfg.checkpoint_k,),
            protocol_name=f"validation_sampled_{cfg.sampled_negative_count}",
            negative_count_requested=cfg.sampled_negative_count,
            include_details=False,
        )
        if validation_metrics.empty:
            raise RuntimeError("No users were available for validation ranking.")

        row = validation_metrics.iloc[0]
        validation_hr = float(row["HR"])
        validation_ndcg = float(row["NDCG"])
        validation_recall = float(row["Recall"])

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                f"validation_HR@{cfg.checkpoint_k}": validation_hr,
                f"validation_NDCG@{cfg.checkpoint_k}": validation_ndcg,
                f"validation_Recall@{cfg.checkpoint_k}": validation_recall,
            }
        )

        print(
            f"[DMF-COMM-BCE] epoch={epoch:03d} "
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
        recall_tied = ndcg_tied and abs(validation_recall - best_recall) <= cfg.min_delta
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
                    f"[DMF-COMM-BCE] Early stopping: no validation "
                    f"NDCG@{cfg.checkpoint_k} improvement for "
                    f"{cfg.patience} epochs."
                )
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    checkpoint = {
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
    return model, history, checkpoint


# =============================================================================
# 10) TRUE ALL-COMMUNITY-ITEM EVALUATION AND RECOMMENDATIONS
# =============================================================================


def evaluate_all_community_items(
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    known_items_all: Mapping[int, Set[int]],
    test_items: Mapping[int, Set[int]],
    k_values: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, float | int]] = []
    recommendation_rows: List[Dict[str, float | int]] = []
    all_item_indices = np.arange(item_vectors.shape[0], dtype=np.int64)
    top_k_max = int(max(k_values))

    for user_index in sorted(test_items):
        positives = set(test_items.get(user_index, set()))
        if not positives:
            continue

        # Exclude all previously known items except held-out test positives.
        excluded = set(known_items_all.get(user_index, set())) - positives
        mask = np.ones(item_vectors.shape[0], dtype=bool)
        if excluded:
            mask[np.fromiter(sorted(excluded), dtype=np.int64)] = False
        candidates = all_item_indices[mask]

        ranked, score_map = rank_candidate_items_with_scores(
            user_vector=user_vectors[user_index],
            item_vectors=item_vectors,
            candidate_items=candidates,
        )

        for k in k_values:
            metric_rows.append(
                {
                    "user": int(user_index),
                    "k": int(k),
                    **evaluate_top_k_for_user(ranked, positives, int(k)),
                }
            )

        for rank, item_index in enumerate(ranked[:top_k_max], start=1):
            recommendation_rows.append(
                {
                    "user": int(user_index),
                    "rank": int(rank),
                    "i_idx": int(item_index),
                    "PredScore": float(score_map[int(item_index)]),
                    "is_positive_in_test": int(item_index in positives),
                    "test_positive_count": int(len(positives)),
                    "candidate_count": int(len(candidates)),
                }
            )

    detail = pd.DataFrame(metric_rows)
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()

    metric_columns = ["HR", "NDCG", "Recall", "Precision", "F1", "MAP"]
    mean_metrics = detail.groupby("k")[metric_columns].mean().reset_index()
    mean_metrics.insert(0, "EvalProtocol", "all_community_items_multi_positive")
    mean_metrics["multi_positive"] = True
    mean_metrics["users_eval"] = int(detail["user"].nunique())
    return mean_metrics, pd.DataFrame(recommendation_rows)


# =============================================================================
# 11) OUTPUT HELPERS
# =============================================================================


def save_mappings(
    run_dir: str,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
) -> Tuple[str, str]:
    user_path = os.path.join(run_dir, "user_mapping.csv")
    item_path = os.path.join(run_dir, "item_mapping.csv")
    pd.DataFrame(
        {"u_idx": np.arange(len(user_ids)), "userID": user_ids}
    ).to_csv(user_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"i_idx": np.arange(len(item_ids)), "movieID": item_ids}
    ).to_csv(item_path, index=False, encoding="utf-8-sig")
    return user_path, item_path


def add_original_ids(
    dataframe: pd.DataFrame,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    result = dataframe.copy()
    result["userID"] = result["user"].map(
        lambda value: int(user_ids[int(value)])
    )
    result["movieID"] = result["i_idx"].map(
        lambda value: int(item_ids[int(value)])
    )
    return result


def add_true_test_ratings(
    recommendations: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations
    rating_map = {
        (int(row.u_idx), int(row.i_idx)): float(row.rating)
        for row in split_df[split_df["split"] == "test"].itertuples(index=False)
    }
    result = recommendations.copy()
    result["TrueRating_Test"] = [
        rating_map.get((int(user), int(item)), np.nan)
        for user, item in zip(result["user"], result["i_idx"])
    ]
    return result


# =============================================================================
# 12) ONE COMMUNITY RUN
# =============================================================================


def run_one_community(
    ratings_path: str,
    cfg: Config,
    base_version_tag: str,
) -> pd.DataFrame:
    set_all_seeds(cfg.seed)

    community_tag = community_tag_from_path(ratings_path)
    version_tag = f"{base_version_tag}_COMM{community_tag}"
    run_dir = build_community_run_dir(cfg, community_tag, version_tag)
    ensure_dir(run_dir)

    print("\n" + "=" * 100)
    print(f"[COMMUNITY] tag={community_tag}")
    print(f"[INPUT] {ratings_path}")
    print(f"[OUTPUT] {run_dir}")
    print(f"[DEVICE] {DEVICE}")
    print(f"[VERSION] {version_tag}")
    print("=" * 100)

    raw = read_ratings_file(ratings_path)
    print(
        f"[RAW] rows={len(raw)} users={raw['userID'].nunique()} "
        f"items={raw['itemID'].nunique()} "
        f"rating_min={raw['rating'].min():.4f} "
        f"rating_max={raw['rating'].max():.4f}"
    )

    filtered = filter_users_by_total_interactions(
        raw,
        cfg.min_user_interactions,
    )
    if filtered.empty:
        raise RuntimeError("No users remain after minimum-interaction filtering.")

    counts = filtered.groupby("userID").size()
    print(
        f"[FILTERED] rows={len(filtered)} users={filtered['userID'].nunique()} "
        f"items={filtered['itemID'].nunique()} min_inter={counts.min()} "
        f"mean_inter={counts.mean():.2f} median_inter={counts.median():.2f}"
    )

    encoded, _u2i, _i2i, user_ids, item_ids = encode_ids(filtered)
    number_of_users = int(len(user_ids))
    number_of_items = int(len(item_ids))

    split_df = split_douban_community(
        encoded=encoded,
        train_positive_threshold=cfg.train_positive_threshold,
        test_positive_threshold=cfg.test_positive_threshold,
        test_fraction=cfg.test_fraction,
        validation_fraction_of_remaining=cfg.validation_fraction_of_remaining,
        seed=cfg.seed,
    )

    train_items = item_sets_for_split(split_df, "train")
    validation_items = item_sets_for_split(split_df, "validation")
    test_items = item_sets_for_split(split_df, "test")
    positive_items_all = build_positive_items_all(split_df)
    known_items_all = build_known_items_all(split_df)

    split_counts = split_df["split"].value_counts().to_dict()
    evaluation_users = sum(bool(items) for items in test_items.values())
    validation_users = sum(bool(items) for items in validation_items.values())
    print(
        f"[SPLIT] train_positive={split_counts.get('train', 0)} "
        f"validation_positive={split_counts.get('validation', 0)} "
        f"test_positive={split_counts.get('test', 0)} "
        f"known_nonpositive={split_counts.get('known_nonpositive', 0)}"
    )
    print(
        f"[EVAL USERS] validation={validation_users} test={evaluation_users}"
    )
    if evaluation_users == 0:
        raise RuntimeError("No users have rating>=4 test positives in this community.")

    training_matrix = build_training_interaction_matrix(
        split_df,
        number_of_users,
        number_of_items,
    )

    train_users, train_sample_items, train_targets = create_bce_implicit_samples(
        positive_items=train_items,
        positive_items_all=positive_items_all,
        number_of_items=number_of_items,
        negative_ratio=cfg.negative_ratio,
        seed=cfg.seed,
    )
    validation_users_array, validation_sample_items, validation_targets = (
        create_bce_implicit_samples(
            positive_items=validation_items,
            positive_items_all=positive_items_all,
            number_of_items=number_of_items,
            negative_ratio=cfg.negative_ratio,
            seed=cfg.seed + 1_000_000,
        )
    )

    if len(train_targets) == 0:
        raise RuntimeError("No training samples were generated.")
    if len(validation_targets) == 0:
        validation_users_array = train_users.copy()
        validation_sample_items = train_sample_items.copy()
        validation_targets = train_targets.copy()
        validation_fallback = True
    else:
        validation_fallback = False

    print(
        f"[BCE-IMPLICIT SAMPLES] train={len(train_targets)} "
        f"validation={len(validation_targets)} "
        f"negative_ratio={cfg.negative_ratio}"
    )

    train_dataset = DMFPointwiseDataset(
        train_users,
        train_sample_items,
        train_targets,
    )
    validation_dataset = DMFPointwiseDataset(
        validation_users_array,
        validation_sample_items,
        validation_targets,
    )
    pin_memory = bool(cfg.pin_memory and DEVICE == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    validation_candidate_sets = build_sampled_candidate_sets(
        positive_items=validation_items,
        positive_items_all=positive_items_all,
        number_of_items=number_of_items,
        negative_count=cfg.sampled_negative_count,
        seed=cfg.seed + 2_000_000,
    )
    if not validation_candidate_sets:
        validation_candidate_sets = build_sampled_candidate_sets(
            positive_items=train_items,
            positive_items_all=positive_items_all,
            number_of_items=number_of_items,
            negative_count=cfg.sampled_negative_count,
            seed=cfg.seed + 2_000_000,
        )

    model = CanonicalDMF(
        training_interaction_matrix=training_matrix,
        first_projection_dimension=cfg.first_projection_dim,
        latent_dimension=cfg.latent_dim,
        initialization_std=cfg.init_std,
    ).to(DEVICE)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[MODEL] users={number_of_users} items={number_of_items} "
        f"parameters={parameter_count:,} "
        f"user_input_dim={number_of_items} item_input_dim={number_of_users}"
    )

    model, history, best_checkpoint = train_dmf_community(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        validation_candidate_sets=validation_candidate_sets,
        cfg=cfg,
        device=DEVICE,
    )

    print("[EVAL] Precomputing community user/item latent representations...")
    user_vectors = encode_all_users(model, DEVICE, cfg.representation_batch_size)
    item_vectors = encode_all_items(model, DEVICE, cfg.representation_batch_size)

    test_candidate_sets = build_sampled_candidate_sets(
        positive_items=test_items,
        positive_items_all=positive_items_all,
        number_of_items=number_of_items,
        negative_count=cfg.sampled_negative_count,
        seed=cfg.seed,
    )
    sampled_metrics, sampled_debug = evaluate_fixed_sampled_candidates(
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        candidate_sets=test_candidate_sets,
        k_values=cfg.k_values,
        protocol_name=f"sampled_{cfg.sampled_negative_count}_multi_positive",
        negative_count_requested=cfg.sampled_negative_count,
        include_details=True,
    )
    if sampled_metrics.empty:
        raise RuntimeError("No test users were evaluated.")

    sampled_metrics.insert(1, "COMMUNITY_TAG", community_tag)
    sampled_metrics.insert(2, "n_users", number_of_users)
    sampled_metrics.insert(3, "n_items", number_of_items)
    print("\n[TEST METRICS — LEGACY-COMPATIBLE SAMPLED PROTOCOL]")
    print(sampled_metrics)

    all_items_metrics = pd.DataFrame()
    all_items_recommendations = pd.DataFrame()
    if cfg.evaluate_all_community_items:
        all_items_metrics, all_items_recommendations = evaluate_all_community_items(
            user_vectors=user_vectors,
            item_vectors=item_vectors,
            known_items_all=known_items_all,
            test_items=test_items,
            k_values=cfg.all_items_k_values,
        )
        if not all_items_metrics.empty:
            all_items_metrics.insert(1, "COMMUNITY_TAG", community_tag)
            all_items_metrics.insert(2, "n_users", number_of_users)
            all_items_metrics.insert(3, "n_items", number_of_items)
            print("\n[TRUE ALL-COMMUNITY-ITEM METRICS]")
            print(all_items_metrics)

    sampled_debug = add_original_ids(sampled_debug, user_ids, item_ids)
    all_items_recommendations = add_original_ids(
        all_items_recommendations,
        user_ids,
        item_ids,
    )
    all_items_recommendations = add_true_test_ratings(
        all_items_recommendations,
        split_df,
    )

    if cfg.debug_users is not None and not sampled_debug.empty:
        sampled_debug = sampled_debug[
            sampled_debug["userID"].isin(set(cfg.debug_users))
        ].copy()

    split_path = os.path.join(run_dir, f"{cfg.dataset_tag}_split_{version_tag}.csv")
    split_df.to_csv(split_path, index=False, encoding="utf-8-sig")
    user_mapping_path, item_mapping_path = save_mappings(run_dir, user_ids, item_ids)

    recommendations_path: Optional[str] = None
    if cfg.save_per_user_recommendations and not all_items_recommendations.empty:
        recommendations_path = os.path.join(
            run_dir,
            f"{cfg.dataset_tag}_all_items_top{max(cfg.all_items_k_values)}_"
            f"{version_tag}.csv",
        )
        all_items_recommendations.to_csv(
            recommendations_path,
            index=False,
            encoding="utf-8-sig",
        )

    debug_path: Optional[str] = None
    if cfg.save_sampled_debug_scores and not sampled_debug.empty:
        debug_path = os.path.join(
            run_dir,
            f"{cfg.dataset_tag}_sampled_{cfg.sampled_negative_count}_debug_"
            f"{version_tag}.csv",
        )
        sampled_debug.to_csv(debug_path, index=False, encoding="utf-8-sig")

    model_path = os.path.join(
        run_dir,
        f"{cfg.dataset_tag}_CanonicalDMF_model_{version_tag}.pt",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "community_tag": community_tag,
            "version_tag": version_tag,
            "number_of_users": number_of_users,
            "number_of_items": number_of_items,
            "best_checkpoint": best_checkpoint,
        },
        model_path,
    )

    combined_metrics = pd.concat(
        [frame for frame in [sampled_metrics, all_items_metrics] if not frame.empty],
        ignore_index=True,
    )

    metadata = {
        **asdict(cfg),
        "community_tag": community_tag,
        "community_file": ratings_path,
        "version_tag": version_tag,
        "device": DEVICE,
        "number_of_users": number_of_users,
        "number_of_items": number_of_items,
        "number_of_rows": int(len(encoded)),
        "model_parameter_count": int(parameter_count),
        "validation_fallback_used": bool(validation_fallback),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "architecture": {
            "user_input": "CC community training-matrix row Y_k[u, :]",
            "item_input": "CC community training-matrix column Y_k[:, i]",
            "user_network": "two-layer canonical DMF tower",
            "item_network": "two-layer canonical DMF tower",
            "score": "cosine similarity",
            "loss": "pointwise BCE; observed positive=1, sampled alternative=0",
        },
        "legacy_compatible_settings": {
            "test_fraction": cfg.test_fraction,
            "train_positive_threshold": cfg.train_positive_threshold,
            "test_positive_threshold": cfg.test_positive_threshold,
            "negative_ratio": cfg.negative_ratio,
            "first_projection_dim": cfg.first_projection_dim,
            "latent_dim": cfg.latent_dim,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            "sampled_negative_count": cfg.sampled_negative_count,
            "k_values": list(cfg.k_values),
        },
        "checkpoint_selection": {
            "primary": f"sampled validation NDCG@{cfg.checkpoint_k}",
            "tie_break_1": f"sampled validation Recall@{cfg.checkpoint_k}",
            "tie_break_2": "validation BCE loss",
            "test_metrics_used_for_selection": False,
            **best_checkpoint,
        },
        "reporting_note": (
            "sampled_20_multi_positive is the primary protocol matched to DC-DMF V7; "
            "all_community_items_multi_positive is a separate stricter diagnostic."
        ),
        "artifacts": {
            "split_csv": split_path,
            "user_mapping_csv": user_mapping_path,
            "item_mapping_csv": item_mapping_path,
            "all_items_recommendations_csv": recommendations_path,
            "sampled_debug_csv": debug_path,
            "model_state_dict": model_path,
        },
    }

    metadata_path = os.path.join(
        run_dir,
        f"{cfg.dataset_tag}_meta_{version_tag}.json",
    )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    excel_path = os.path.join(
        run_dir,
        f"{cfg.dataset_tag}_CanonicalDMF_{version_tag}.xlsx",
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        sampled_metrics.to_excel(writer, index=False, sheet_name="sampled_mean")
        if not all_items_metrics.empty:
            all_items_metrics.to_excel(writer, index=False, sheet_name="all_items_mean")
        pd.DataFrame(history).to_excel(
            writer,
            index=False,
            sheet_name="training_history",
        )

    print("\n[SAVED COMMUNITY]")
    print(f"Metadata: {metadata_path}")
    print(f"Metrics:  {excel_path}")
    print(f"Split:    {split_path}")
    print(f"Model:    {model_path}")
    if recommendations_path:
        print(f"Recs:     {recommendations_path}")
    if debug_path:
        print(f"Debug:    {debug_path}")

    return combined_metrics


# =============================================================================
# 13) SUMMARY ACROSS COMMUNITIES
# =============================================================================


def build_aggregate_summary(per_community: pd.DataFrame) -> pd.DataFrame:
    if per_community.empty:
        return pd.DataFrame()

    metric_columns = ["HR", "NDCG", "Recall", "Precision", "F1", "MAP"]
    rows: List[Dict[str, float | int | str]] = []

    for (protocol, k), group in per_community.groupby(
        ["EvalProtocol", "k"],
        sort=True,
    ):
        macro = {column: float(group[column].mean()) for column in metric_columns}
        weights = group["users_eval"].astype(float).to_numpy()
        if weights.sum() > 0:
            weighted = {
                column: float(
                    np.average(group[column].astype(float), weights=weights)
                )
                for column in metric_columns
            }
        else:
            weighted = macro.copy()

        common = {
            "EvalProtocol": str(protocol),
            "k": int(k),
            "communities": int(group["COMMUNITY_TAG"].nunique()),
            "users_weight_sum": int(group["users_eval"].sum()),
        }
        rows.append(
            {
                "Aggregate": "macro_community_mean",
                **common,
                **macro,
            }
        )
        rows.append(
            {
                "Aggregate": "user_weighted_mean",
                **common,
                **weighted,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    validate_config(CFG)
    ensure_dir(CFG.output_root)
    base_version_tag = build_base_version_tag(CFG)

    print(f"[DEVICE] {DEVICE}")
    print(f"[BASE VERSION] {base_version_tag}")
    print(f"[CC COMMUNITIES] {len(CFG.community_files)}")

    results: List[pd.DataFrame] = []
    failures: List[Dict[str, str]] = []

    for ratings_path in CFG.community_files:
        try:
            result = run_one_community(
                ratings_path=ratings_path,
                cfg=CFG,
                base_version_tag=base_version_tag,
            )
            if not result.empty:
                results.append(result)
        except Exception as error:
            print("\n" + "-" * 100)
            print(f"[ERROR] community file failed: {ratings_path}")
            print(f"[ERROR] {type(error).__name__}: {error}")
            print("-" * 100)
            failures.append(
                {
                    "community_file": ratings_path,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    aggregate = build_aggregate_summary(combined)

    summary_path = os.path.join(
        CFG.output_root,
        f"Canonical_CC_DMF_Douban_COMMUNITIES_summary_{base_version_tag}.xlsx",
    )
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        if not combined.empty:
            combined.to_excel(writer, index=False, sheet_name="per_community")
        if not aggregate.empty:
            aggregate.to_excel(writer, index=False, sheet_name="aggregate")
        if failures:
            pd.DataFrame(failures).to_excel(
                writer,
                index=False,
                sheet_name="failures",
            )
        if combined.empty and not failures:
            pd.DataFrame([{"status": "No results were produced."}]).to_excel(
                writer,
                index=False,
                sheet_name="status",
            )

    print("\n" + "=" * 100)
    print("[FINAL SUMMARY]")
    if not aggregate.empty:
        print(aggregate)
    print(f"[SUMMARY SAVED] {summary_path}")
    print(f"[SUCCESSFUL COMMUNITIES] {len(results)}")
    print(f"[FAILED COMMUNITIES] {len(failures)}")


if __name__ == "__main__":
    main()
