"""Participant identifiers never go into a message, which can reach run logs (CLAUDE.md section 3)."""
import os

import pandas as pd
import pytest

from ml_helpers import make_cfg, synthetic_dataset


def test_unknown_participants_are_counted_not_named(tmp_path):
    from genostore.datamodule import SNPStoreDataModule, dataset_dir
    data_dir, _ = synthetic_dataset(tmp_path, n=300, snps=60)
    cfg = make_cfg(data_dir, "multi")
    path = os.path.join(dataset_dir(cfg), "samples_val.tsv")
    df = pd.read_csv(path, sep="\t", dtype={"IID": str})
    unknown = ["not-a-participant-1", "not-a-participant-2", "not-a-participant-3", "not-a-participant-4"]
    extra = pd.concat([df.iloc[:1]] * len(unknown), ignore_index=True).assign(IID=unknown)
    pd.concat([df, extra], ignore_index=True).to_csv(path, sep="\t", index=False)
    with pytest.raises(KeyError) as err:
        SNPStoreDataModule(cfg)
    message = str(err.value)
    assert f"samples_val.tsv: {len(unknown)} of {len(df) + len(unknown)} IDs are not in the store" in message
    assert not any(s in message for s in unknown + list(df["IID"]))
