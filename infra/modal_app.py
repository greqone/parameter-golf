from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal

from infra.modal_utils import ExperimentSpec, REPO_ROOT, ensure_run_id, execute_training_run


APP_NAME = os.environ.get("PG_MODAL_APP_NAME", "parameter-golf-autoresearch")
REMOTE_REPO_ROOT = "/root/parameter-golf"
REMOTE_DATA_ROOT = "/pg-data"
REMOTE_CHECKPOINT_ROOT = "/pg-checkpoints"
REMOTE_RESULTS_ROOT = "/pg-results"
DATA_VOLUME_NAME = os.environ.get("PG_MODAL_DATA_VOLUME", "pg-data")
CHECKPOINT_VOLUME_NAME = os.environ.get("PG_MODAL_CHECKPOINT_VOLUME", "pg-checkpoints")
RESULTS_VOLUME_NAME = os.environ.get("PG_MODAL_RESULTS_VOLUME", "pg-results")


def _ignore_local_path(path: Path) -> bool:
    text = path.as_posix()
    normalized = f"/{text.lstrip('./')}"
    return any(
        fragment in normalized
        for fragment in (
            "/.git",
            "/.venv",
            "/.venv311",
            "/__pycache__",
            "/data/datasets",
            "/logs",
            "/experiments/runs",
            "/experiments/benchmarks",
        )
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"))
    .pip_install("modal")
    .add_local_dir(REPO_ROOT, REMOTE_REPO_ROOT, copy=True, ignore=_ignore_local_path)
)
app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
volume_mounts = {
    REMOTE_DATA_ROOT: data_volume,
    REMOTE_CHECKPOINT_ROOT: checkpoint_volume,
    REMOTE_RESULTS_ROOT: results_volume,
}


def _run_remote(spec_dict: dict[str, Any], *, runtime: str, gpu_type: str, gpu_count: int) -> dict[str, Any]:
    spec = ExperimentSpec.from_dict(spec_dict)
    ensure_run_id(spec)
    result = execute_training_run(
        spec=spec,
        repo_root=Path(REMOTE_REPO_ROOT),
        data_root=Path(REMOTE_DATA_ROOT),
        runs_root=Path(REMOTE_RESULTS_ROOT) / "runs",
        checkpoint_root=Path(REMOTE_CHECKPOINT_ROOT),
        runtime=runtime,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        python_executable=None,
        torchrun_executable="torchrun",
    )
    data_volume.commit()
    checkpoint_volume.commit()
    results_volume.commit()
    return result


@app.function(
    name="pg-proxy",
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=7200,
    retries=modal.Retries(max_retries=2, initial_delay=10.0, max_delay=60.0),
    startup_timeout=1800,
    volumes=volume_mounts,
)
def run_proxy(spec_dict: dict[str, Any]) -> dict[str, Any]:
    return _run_remote(spec_dict, runtime="modal:proxy", gpu_type="H100", gpu_count=1)


@app.function(
    name="pg-promote-b200",
    gpu="B200+:8",
    cpu=32,
    memory=131072,
    timeout=14400,
    retries=modal.Retries(max_retries=2, initial_delay=30.0, max_delay=60.0),
    startup_timeout=2400,
    volumes=volume_mounts,
)
def run_promote_b200(spec_dict: dict[str, Any]) -> dict[str, Any]:
    return _run_remote(spec_dict, runtime="modal:promote", gpu_type="B200+", gpu_count=8)


@app.function(
    name="pg-promote-hopper",
    gpu="H200:8",
    cpu=32,
    memory=131072,
    timeout=14400,
    retries=modal.Retries(max_retries=2, initial_delay=30.0, max_delay=60.0),
    startup_timeout=2400,
    volumes=volume_mounts,
)
def run_promote_hopper(spec_dict: dict[str, Any]) -> dict[str, Any]:
    return _run_remote(spec_dict, runtime="modal:promote-fallback", gpu_type="H200", gpu_count=8)


@app.function(
    name="pg-verify-h100",
    gpu="H100!:8",
    cpu=32,
    memory=131072,
    timeout=14400,
    retries=modal.Retries(max_retries=2, initial_delay=30.0, max_delay=60.0),
    startup_timeout=2400,
    volumes=volume_mounts,
)
def run_verify_h100(spec_dict: dict[str, Any]) -> dict[str, Any]:
    return _run_remote(spec_dict, runtime="modal:verify", gpu_type="H100!", gpu_count=8)


@app.local_entrypoint()
def launch(config: str, stage: str = "proxy") -> None:
    config_path = Path(config).resolve()
    spec = ExperimentSpec.from_path(config_path)
    ensure_run_id(spec)
    with app.run():
        if stage == "proxy":
            result = run_proxy.remote(spec.to_dict())
        elif stage == "promote":
            result = run_promote_b200.remote(spec.to_dict())
        elif stage == "promote-hopper":
            result = run_promote_hopper.remote(spec.to_dict())
        elif stage == "verify":
            result = run_verify_h100.remote(spec.to_dict())
        else:
            raise ValueError(f"Unsupported stage {stage!r}")
    print(json.dumps({k: v for k, v in result.items() if k != "log_text"}, indent=2, sort_keys=True))
