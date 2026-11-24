"""Sparse interaction matrices with the user/item IDs of the official splits."""

from collections import Counter
from dataclasses import dataclass
from random import Random

import numpy as np
from scipy.sparse import csr_matrix

from prepare import DATASETS


@dataclass(frozen=True)
class Interactions:
    train: csr_matrix
    valid: csr_matrix
    test: csr_matrix
    inputs: csr_matrix


def temporal_weights(users, minimum):
    totals = Counter(users)
    seen = Counter()
    weights = np.empty(len(users), dtype=np.float32)
    for index, user in enumerate(users):
        # The official files store chronological order, with no timestamp column.
        weights[index] = minimum + (1 - minimum) * seen[user] / max(totals[user] - 1, 1)
        seen[user] += 1
    return weights


def load_data(name, time_min=None):
    rows = []
    for split in ("train", "valid", "test"):
        path = DATASETS / name / f"{split}_list.npy"
        if not path.is_file():
            raise ValueError(f"Arquivo ausente: {path}. Execute python prepare.py {name}.")
        pairs = np.load(path, allow_pickle=False)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu" or len(pairs) == 0 or pairs.min() < 0:
            raise ValueError(f"Pares de interação inválidos em {path}.")
        rows.append(pairs)

    shape = tuple(rows[0].max(axis=0) + 1)
    matrices = [csr_matrix((np.ones(len(pairs), dtype=np.float32), pairs.T), shape=shape) for pairs in rows]
    train, valid, test = matrices
    if any(matrix.nnz != len(pairs) for matrix, pairs in zip(matrices, rows)):
        raise ValueError(f"Pares duplicados em {name}.")
    # Yelp noisy ships with overlapping pairs. Keep the authors' splits and masks.
    inputs = train
    if time_min is not None:
        inputs = csr_matrix((temporal_weights(rows[0][:, 0], time_min), rows[0].T), shape=shape)
    return Interactions(train, valid, test, inputs)


def add_random_noise(data, ratio, seed):
    """Inject uniform unseen items into train and validation at the given ratio."""
    rng = Random(seed)
    observed = data.train + data.valid + data.test
    n_users, n_items = observed.shape
    available = n_items - observed.getnnz(axis=1)
    noisy_train = np.minimum((data.train.getnnz(axis=1) * ratio).astype(int), available)
    noisy_valid = np.minimum((data.valid.getnnz(axis=1) * ratio).astype(int), available - noisy_train)
    train_columns, valid_columns = [], []
    for user in range(n_users):
        train_count, valid_count = int(noisy_train[user]), int(noisy_valid[user])
        if train_count + valid_count == 0:
            continue
        begin, end = observed.indptr[user:user + 2]
        forbidden = np.sort(observed.indices[begin:end])
        allowed_offsets = rng.sample(range(n_items - len(forbidden)), train_count + valid_count)
        # Convert offsets in the unseen catalog to original item IDs.
        gaps = forbidden - np.arange(len(forbidden))
        sampled = allowed_offsets + np.searchsorted(gaps, allowed_offsets, side="right")
        train_columns.extend(sampled[:train_count])
        valid_columns.extend(sampled[train_count:])

    train_rows = np.repeat(np.arange(n_users), noisy_train)
    valid_rows = np.repeat(np.arange(n_users), noisy_valid)
    train = data.train + csr_matrix((np.ones(len(train_rows), np.float32),
                                    (train_rows, train_columns)), shape=observed.shape)
    valid = data.valid + csr_matrix((np.ones(len(valid_rows), np.float32),
                                    (valid_rows, valid_columns)), shape=observed.shape)
    return Interactions(train, valid, data.test, train)
