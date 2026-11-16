"""Participant identifiers and values never go into a message, which can reach run logs, and
per-person files never go into a repository or a run directory (CLAUDE.md section 3)."""
import os
import traceback

import numpy as np
import pandas as pd
import pytest

from ml_helpers import make_cfg, synthetic_dataset


def _dataset(tmp_path):
    from genostore.datamodule import dataset_dir
    data_dir, _ = synthetic_dataset(tmp_path, n=300, snps=60)
    cfg = make_cfg(data_dir, "multi")
    path = os.path.join(dataset_dir(cfg), "samples_val.tsv")
    return cfg, path, pd.read_csv(path, sep="\t", dtype={"IID": str})


def _shown(err):
    """Everything a log would show of the exception, chained exceptions included."""
    return "".join(traceback.format_exception(type(err), err, err.__traceback__))


def test_unknown_participants_are_counted_not_named(tmp_path):
    from genostore.datamodule import SNPStoreDataModule
    cfg, path, df = _dataset(tmp_path)
    unknown = ["not-a-participant-1", "not-a-participant-2", "not-a-participant-3", "not-a-participant-4"]
    extra = pd.concat([df.iloc[:1]] * len(unknown), ignore_index=True).assign(IID=unknown)
    pd.concat([df, extra], ignore_index=True).to_csv(path, sep="\t", index=False)
    with pytest.raises(KeyError) as err:
        SNPStoreDataModule(cfg)
    assert f"samples_val.tsv: {len(unknown)} of {len(df) + len(unknown)} IDs are not in the store" in str(err.value)
    assert not any(s in _shown(err.value) for s in unknown + list(df["IID"]))


@pytest.mark.parametrize("column", ["label", "covariate"])
def test_a_cell_that_is_not_a_number_is_not_quoted(tmp_path, column):
    """A shifted row puts text, here a participant's ID, where a number belongs."""
    from genostore.datamodule import SNPStoreDataModule
    cfg, path, df = _dataset(tmp_path)
    dm = SNPStoreDataModule(cfg)
    names = dm.label_names if column == "label" else dm.cov_names
    marker = df["IID"].iloc[7] + "MARKER"
    df[names[0]] = df[names[0]].astype(object)
    df.loc[7, names[0]] = marker
    df.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="is not a number") as err:
        SNPStoreDataModule(cfg)
    assert marker not in _shown(err.value) and df["IID"].iloc[7] not in _shown(err.value)


def test_one_label_s_other_values_are_counted_not_listed():
    from types import SimpleNamespace
    from genostore.datamodule import SNPStoreDataModule
    one_label = SimpleNamespace(cont=False, multilabel=False)
    SNPStoreDataModule._check_01(one_label, "train", np.array([0, 1, 1, 0]))
    with pytest.raises(ValueError, match="2 are not") as err:
        SNPStoreDataModule._check_01(one_label, "train", np.array([0, 1, 1234.5, 7.25]))
    assert "1234.5" not in str(err.value) and "7.25" not in str(err.value)


def test_per_person_scores_stay_out_of_repositories_and_run_directories(tmp_path):
    from score_snapshots import refuse_shared_place
    repo, run, data = tmp_path / "repo", tmp_path / "runs" / "r", tmp_path / "data"
    (repo / ".git").mkdir(parents=True)
    run.mkdir(parents=True)
    for bad in (repo / "scores" / "t" / "s", run / "scores"):
        with pytest.raises(SystemExit, match="per-person scores belong under data/"):
            refuse_shared_place(str(bad), str(run))
    refuse_shared_place(str(data / "scores" / "t" / "s"), str(run))
