import importlib

from app.data import paths


def teardown_function():
    # _DATA_ROOT is computed at import time; make sure no test leaves a
    # stale env var or reloaded module state behind for the next one.
    import os

    os.environ.pop("MARKET_DATA_ROOT", None)
    importlib.reload(paths)


def test_data_root_defaults_to_repo_relative_path_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("MARKET_DATA_ROOT", raising=False)
    importlib.reload(paths)

    assert paths._DATA_ROOT.name == "data"
    assert paths.RAW_DATA_DIR == paths._DATA_ROOT / "raw"


def test_market_data_root_env_var_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_DATA_ROOT", str(tmp_path))
    importlib.reload(paths)

    assert paths._DATA_ROOT == tmp_path
    assert paths.RAW_DATA_DIR == tmp_path / "raw"
    assert paths.MODELS_DATA_DIR == tmp_path / "models"
