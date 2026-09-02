from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from contrail_benchmark.segmentation.data import make_fold_splits

DATA_ROOT = os.environ.get("CONTRAIL_DATA_ROOT")
EXPECTED_TRAIN_HASH = "56c9223b160000465d6adcdc5ea1a81207803fbfbb8c2c151730c9b3aff311b3"
EXPECTED_VALIDATION_HASH = (
    "67bb9ee5c1007b792767cba6215b013ec893102cce2a8845523b736b9d5df066"
)


def _stem_hash(pool: list[tuple[Path, Path]], indices) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(f"{pool[int(index)][0].stem}\n".encode())
    return digest.hexdigest()


@pytest.mark.skipif(
    not DATA_ROOT or not Path(DATA_ROOT).is_dir(),
    reason="set CONTRAIL_DATA_ROOT to run the provider-controlled split test",
)
def test_fold5_matches_recovered_training_split() -> None:
    pool, splits, _ = make_fold_splits(DATA_ROOT, folds=5, seed=3407)
    training, validation = splits[4]
    assert len(pool) == 16_080
    assert len(training) == 12_864
    assert len(validation) == 3_216
    assert _stem_hash(pool, training) == EXPECTED_TRAIN_HASH
    assert _stem_hash(pool, validation) == EXPECTED_VALIDATION_HASH
