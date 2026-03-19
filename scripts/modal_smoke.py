from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.modal_app import app, run_proxy
from infra.modal_utils import (
    EXPERIMENTS_ROOT,
    REPO_ROOT,
    ExperimentSpec,
    ensure_experiment_layout,
    ensure_run_id,
    execute_training_run,
    modal_auth_configured,
    record_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Parameter Golf smoke test locally or on Modal.")
    parser.add_argument(
        "--config",
        default=str(EXPERIMENTS_ROOT / "configs" / "proxy_smoke_baseline.json"),
        help="Path to a JSON experiment config.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "local", "remote"),
        default="auto",
        help="Execution mode. Auto prefers Modal when auth is present.",
    )
    return parser.parse_args()


def run_local(spec: ExperimentSpec) -> dict:
    return execute_training_run(
        spec=spec,
        repo_root=REPO_ROOT,
        data_root=REPO_ROOT / "data",
        runs_root=EXPERIMENTS_ROOT / "runs",
        checkpoint_root=EXPERIMENTS_ROOT / "checkpoints",
        runtime="local:proxy",
        gpu_type="RTX4090",
        gpu_count=1,
    )


def run_remote(spec: ExperimentSpec) -> dict:
    with app.run():
        return run_proxy.remote(spec.to_dict())


def main() -> None:
    args = parse_args()
    ensure_experiment_layout()
    config_path = Path(args.config).resolve()
    spec = ExperimentSpec.from_path(config_path)
    ensure_run_id(spec)

    if args.mode == "local" or (args.mode == "auto" and not modal_auth_configured()):
        result = run_local(spec)
    else:
        result = run_remote(spec)

    run_dir = record_result(result, config_path=config_path)
    print(json.dumps({k: v for k, v in result.items() if k != "log_text"}, indent=2, sort_keys=True))
    print(f"local_run_dir={run_dir}")


if __name__ == "__main__":
    main()
