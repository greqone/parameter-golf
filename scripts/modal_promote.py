from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.modal_app import app, run_promote_b200, run_promote_hopper
from infra.modal_utils import ExperimentSpec, ensure_experiment_layout, ensure_run_id, modal_auth_configured, record_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote one candidate to an 8-GPU Modal run.")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config.")
    parser.add_argument(
        "--hopper-fallback",
        action="store_true",
        help="Use H200:8 instead of B200+:8 for safer Hopper-only execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_experiment_layout()
    if not modal_auth_configured():
        raise RuntimeError("Modal auth is not configured.")

    config_path = Path(args.config).resolve()
    spec = ExperimentSpec.from_path(config_path)
    spec.stage = "promote"
    ensure_run_id(spec)

    with app.run():
        if args.hopper_fallback:
            result = run_promote_hopper.remote(spec.to_dict())
        else:
            result = run_promote_b200.remote(spec.to_dict())

    run_dir = record_result(result, config_path=config_path)
    print(json.dumps({k: v for k, v in result.items() if k != "log_text"}, indent=2, sort_keys=True))
    print(f"local_run_dir={run_dir}")


if __name__ == "__main__":
    main()
