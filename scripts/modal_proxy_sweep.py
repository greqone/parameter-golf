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


DEFAULT_CONFIGS = [
    EXPERIMENTS_ROOT / "configs" / "proxy_smoke_baseline.json",
    EXPERIMENTS_ROOT / "configs" / "proxy_fp16_embed_wd3600.json",
    EXPERIMENTS_ROOT / "configs" / "proxy_long_context_seq2048.json",
    EXPERIMENTS_ROOT / "configs" / "proxy_sliding_window_eval.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch of proxy experiments.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[str(path) for path in DEFAULT_CONFIGS],
        help="JSON config files to launch.",
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


def failed_result(spec: ExperimentSpec, error: BaseException) -> dict:
    return {
        "timestamp_utc": "",
        "run_id": spec.run_id,
        "name": spec.name,
        "stage": spec.stage,
        "status": "failed",
        "runtime": "modal:proxy",
        "commit": spec.commit,
        "script_path": spec.script_path,
        "data_variant": spec.data_variant,
        "seed": spec.seed,
        "gpu_type": "H100",
        "gpu_count": 1,
        "wallclock_seconds": "",
        "logged_train_time_ms": "",
        "train_shards": spec.train_shards,
        "final_exact_val_loss": "",
        "final_exact_val_bpb": "",
        "compressed_bytes": "",
        "code_bytes": "",
        "total_bytes": "",
        "peak_memory_mib": "",
        "step": "",
        "iterations": "",
        "description": f"{spec.description} | error={error}",
        "log_text": "",
    }


def main() -> None:
    args = parse_args()
    ensure_experiment_layout()
    config_paths = [Path(path).resolve() for path in args.configs]
    specs = [ExperimentSpec.from_path(path) for path in config_paths]
    for spec in specs:
        ensure_run_id(spec)

    results: list[dict] = []
    if args.mode == "local" or (args.mode == "auto" and not modal_auth_configured()):
        for spec in specs:
            results.append(run_local(spec))
    else:
        with app.run():
            remote_results = list(
                run_proxy.map(
                    [spec.to_dict() for spec in specs],
                    return_exceptions=True,
                    wrap_returned_exceptions=False,
                )
            )
        for spec, remote_result in zip(specs, remote_results, strict=True):
            if isinstance(remote_result, BaseException):
                results.append(failed_result(spec, remote_result))
            else:
                results.append(remote_result)

    summaries = []
    for config_path, result in zip(config_paths, results, strict=True):
        run_dir = record_result(result, config_path=config_path)
        summaries.append(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "final_exact_val_bpb": result.get("final_exact_val_bpb"),
                "wallclock_seconds": result.get("wallclock_seconds"),
                "local_run_dir": str(run_dir),
            }
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
