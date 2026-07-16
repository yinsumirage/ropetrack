from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments" / "clean_baseline.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_dataset_config(dataset: str) -> dict[str, Any]:
    cfg = load_yaml(REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml")
    cfg.setdefault("adapter", dataset)
    return cfg


def build_run_args(
    dataset: str,
    method: str | None = None,
    config: Path = DEFAULT_CONFIG,
    **overrides,
) -> SimpleNamespace:
    exp_cfg = load_yaml(config)
    data_cfg = load_dataset_config(dataset)
    defaults = exp_cfg.get("defaults", {})
    methods = exp_cfg.get("methods", {})
    method_name = method or defaults.get("method")
    if method_name not in methods:
        raise ValueError(f"unknown method {method_name!r} in {config}")
    method_cfg = methods[method_name]
    protocol = data_cfg.get("protocol", {})
    adapter = data_cfg.get("adapter", dataset)
    out_dir = overrides.get("out_dir") or Path(defaults["out_dir"].format(
        experiment=exp_cfg["name"],
        dataset=dataset,
        method=method_name,
    ))

    def value(name: str, default=None):
        return overrides[name] if name in overrides and overrides[name] is not None else default

    common = {
        "dataset": dataset,
        "adapter": adapter,
        "method": method_name,
        "out_dir": out_dir,
        "split": value("split", "evaluation"),
        "limit": value("limit", defaults.get("limit", 0)),
        "mode": value("mode", protocol.get("benchmark_mode") or defaults.get("mode")),
        "backend": value("backend", method_cfg["backend"]),
        "units": value("units", protocol.get("units") or defaults.get("units")),
        "device": value("device", defaults.get("device")),
        "batch_size": value("batch_size", defaults.get("batch_size")),
        "detector_batch_size": value("detector_batch_size", defaults.get("detector_batch_size")),
        "num_workers": value("num_workers", defaults.get("num_workers")),
        "joint_source": value("joint_source", protocol.get("joint_source") or defaults.get("joint_source")),
        "protocol_check_samples": value("protocol_check_samples", defaults.get("protocol_check_samples")),
        "protocol_tolerance_m": value("protocol_tolerance_m", defaults.get("protocol_tolerance_m")),
        "eval_num_workers": value("eval_num_workers", defaults.get("eval_num_workers")),
        "run_eval": bool(value("run_eval", False)),
        "save_mano_cache": bool(value("save_mano_cache", False)),
        "joint_only_output": bool(value("joint_only_output", False)),
        "wilor_ckpt": repo_path(value("wilor_ckpt", method_cfg.get("wilor_ckpt"))),
        "wilor_cfg": repo_path(value("wilor_cfg", method_cfg.get("wilor_cfg"))),
        "hamer_ckpt": repo_path(value("hamer_ckpt", method_cfg.get("hamer_ckpt"))),
    }
    if adapter == "freihand":
        common["freihand_root"] = value("freihand_root", Path(data_cfg["remote_root"]))
    elif adapter == "ho3d":
        common["ho3d_root"] = value("ho3d_root", Path(data_cfg["remote_root"]))
    else:
        common["root"] = value("root", Path(data_cfg["remote_root"]))
    return SimpleNamespace(**common)
