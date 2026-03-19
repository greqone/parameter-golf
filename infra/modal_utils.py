from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
RUNS_ROOT = EXPERIMENTS_ROOT / "runs"
RESULTS_TSV_PATH = EXPERIMENTS_ROOT / "results.tsv"
BEST_CANDIDATES_PATH = EXPERIMENTS_ROOT / "best_candidates.json"
CONFIGS_ROOT = EXPERIMENTS_ROOT / "configs"
MODAL_CONFIG_PATH = Path.home() / ".modal.toml"
DEFAULT_FINEWEB_REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
DEFAULT_FINEWEB_REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
RESULT_COLUMNS = [
    "timestamp_utc",
    "run_id",
    "name",
    "stage",
    "status",
    "runtime",
    "commit",
    "script_path",
    "data_variant",
    "seed",
    "gpu_type",
    "gpu_count",
    "wallclock_seconds",
    "logged_train_time_ms",
    "train_shards",
    "final_exact_val_loss",
    "final_exact_val_bpb",
    "compressed_bytes",
    "code_bytes",
    "total_bytes",
    "peak_memory_mib",
    "step",
    "iterations",
    "description",
]


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    description: str
    stage: str
    script_path: str = "train_gpt.py"
    data_variant: str = "sp1024"
    train_shards: int = 1
    seed: int = 1337
    env: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    checkpoint_every: int = 0
    run_id: str = ""
    commit: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentSpec":
        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            stage=str(raw.get("stage", "proxy")),
            script_path=str(raw.get("script_path", "train_gpt.py")),
            data_variant=str(raw.get("data_variant", "sp1024")),
            train_shards=int(raw.get("train_shards", 1)),
            seed=int(raw.get("seed", 1337)),
            env={str(k): str(v) for k, v in dict(raw.get("env", {})).items()},
            tags=[str(x) for x in raw.get("tags", [])],
            checkpoint_every=int(raw.get("checkpoint_every", 0)),
            run_id=str(raw.get("run_id", "")),
            commit=str(raw.get("commit", "")),
            notes=str(raw.get("notes", "")),
        )

    @classmethod
    def from_path(cls, path: Path) -> "ExperimentSpec":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def normalized_env(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self.env.items()}

    def resolved_script_path(self, repo_root: Path) -> Path:
        path = Path(self.script_path)
        return path if path.is_absolute() else (repo_root / path).resolve()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:64] or "run"


def git_executable() -> str | None:
    candidates = [
        os.environ.get("GIT_EXE"),
        shutil.which("git"),
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
    ]
    for candidate in candidates:
        if candidate and (candidate == shutil.which("git") or Path(candidate).exists()):
            return candidate
    return None


def current_commit(repo_root: Path = REPO_ROOT) -> str:
    git = git_executable()
    if git is None or not (repo_root / ".git").exists():
        return ""
    proc = subprocess.run(
        [git, "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def ensure_experiment_layout() -> None:
    CONFIGS_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV_PATH.exists():
        with RESULTS_TSV_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t")
            writer.writeheader()
    if not BEST_CANDIDATES_PATH.exists():
        write_json(
            BEST_CANDIDATES_PATH,
            {"updated_at": None, "best_overall": [], "by_stage": {}},
        )


def ensure_run_id(spec: ExperimentSpec) -> str:
    if not spec.run_id:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        spec.run_id = f"{timestamp}-{slugify(spec.name)}"
    if not spec.commit:
        spec.commit = current_commit()
    return spec.run_id


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def modal_auth_configured() -> bool:
    if not MODAL_CONFIG_PATH.exists():
        return False
    try:
        data = tomllib.loads(MODAL_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return any(
        isinstance(value, dict) and value.get("token_id") and value.get("token_secret")
        for value in data.values()
    )


def dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"Unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")


def local_path_for_remote(target_root: Path, relative_path: str, remote_root_prefix: str) -> Path:
    remote_path = Path(relative_path)
    if remote_root_prefix and remote_path.parts[:1] == (remote_root_prefix,):
        remote_path = remote_path.relative_to(remote_root_prefix)
    if remote_path.parts[:1] == ("datasets",):
        return target_root / "datasets" / Path(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return target_root / "tokenizers" / Path(*remote_path.parts[1:])
    return target_root / remote_path


def _download_artifact(
    *,
    target_root: Path,
    relative_path: str,
    repo_id: str,
    remote_root_prefix: str,
) -> Path:
    destination = local_path_for_remote(target_root, relative_path, remote_root_prefix)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote_path = Path(relative_path)
    cached_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type="dataset",
        )
    )
    cached_source = cached_path.resolve(strict=True)
    try:
        os.link(cached_source, destination)
    except OSError:
        shutil.copy2(cached_source, destination)
    return destination


def _manifest_path(target_root: Path) -> Path:
    return target_root / "manifest.json"


def download_challenge_data(
    *,
    target_root: Path,
    variant: str,
    train_shards: int,
    repo_id: str = DEFAULT_FINEWEB_REPO_ID,
    remote_root_prefix: str = DEFAULT_FINEWEB_REMOTE_ROOT_PREFIX,
) -> dict[str, Any]:
    target_root.mkdir(parents=True, exist_ok=True)
    manifest_local_path = _manifest_path(target_root)
    if not manifest_local_path.exists():
        source_manifest = _download_artifact(
            target_root=target_root,
            relative_path=f"{remote_root_prefix}/manifest.json",
            repo_id=repo_id,
            remote_root_prefix=remote_root_prefix,
        )
        if source_manifest != manifest_local_path:
            shutil.copy2(source_manifest, manifest_local_path)
    manifest = json.loads(manifest_local_path.read_text(encoding="utf-8"))
    dataset_name = dataset_dir_for_variant(variant)
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_name), None)
    if dataset_entry is None:
        raise ValueError(f"Dataset {dataset_name} not found in manifest")
    max_train_shards = int((dataset_entry.get("stats") or {}).get("files_train", 0))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val", 0))
    if train_shards < 0 or train_shards > max_train_shards:
        raise ValueError(f"{variant} exposes {max_train_shards} train shards, requested {train_shards}")
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if tokenizer_entry is None:
        raise ValueError(f"Tokenizer {tokenizer_name} not found in manifest")

    dataset_prefix = f"{remote_root_prefix}/datasets/{dataset_name}"
    for i in range(val_shards):
        _download_artifact(
            target_root=target_root,
            relative_path=f"{dataset_prefix}/fineweb_val_{i:06d}.bin",
            repo_id=repo_id,
            remote_root_prefix=remote_root_prefix,
        )
    for i in range(train_shards):
        _download_artifact(
            target_root=target_root,
            relative_path=f"{dataset_prefix}/fineweb_train_{i:06d}.bin",
            repo_id=repo_id,
            remote_root_prefix=remote_root_prefix,
        )

    tokenizer_artifacts = [
        tokenizer_entry.get(key)
        for key in ("model_path", "vocab_path", "path")
        if tokenizer_entry.get(key)
    ]
    if not tokenizer_artifacts:
        raise ValueError(f"Tokenizer entry missing artifacts: {tokenizer_entry}")
    local_tokenizer_paths = [
        _download_artifact(
            target_root=target_root,
            relative_path=f"{remote_root_prefix}/{artifact_path}",
            repo_id=repo_id,
            remote_root_prefix=remote_root_prefix,
        )
        for artifact_path in tokenizer_artifacts
    ]
    tokenizer_model_path = next((path for path in local_tokenizer_paths if path.suffix == ".model"), local_tokenizer_paths[0])
    data_path = target_root / "datasets" / dataset_name
    return {
        "data_path": str(data_path),
        "dataset_name": dataset_name,
        "tokenizer_model_path": str(tokenizer_model_path),
        "vocab_size": int(dataset_entry.get("vocab_size", 0)),
        "tokenizer_name": tokenizer_name,
        "val_shards": val_shards,
    }


def prepare_dataset_view(
    *,
    source_data_path: Path,
    target_root: Path,
    dataset_name: str,
    train_shards: int,
) -> Path:
    if train_shards <= 0:
        return source_data_path

    source_train_files = sorted(source_data_path.glob("fineweb_train_*.bin"))
    source_val_files = sorted(source_data_path.glob("fineweb_val_*.bin"))
    if len(source_train_files) < train_shards:
        raise ValueError(
            f"{source_data_path} exposes {len(source_train_files)} train shards, requested {train_shards}"
        )

    view_path = target_root / "views" / f"{dataset_name}_train{train_shards:03d}"
    marker_path = view_path / ".ready"
    desired_names = {path.name for path in source_train_files[:train_shards]}
    desired_names.update(path.name for path in source_val_files)
    if marker_path.exists():
        existing_names = {path.name for path in view_path.glob("*.bin")}
        if existing_names == desired_names:
            return view_path

    view_path.mkdir(parents=True, exist_ok=True)
    for path in view_path.glob("*.bin"):
        if path.name not in desired_names:
            path.unlink()

    def link_into_view(source_path: Path) -> None:
        target_path = view_path / source_path.name
        if target_path.exists():
            return
        try:
            os.link(source_path, target_path)
        except OSError:
            try:
                os.symlink(source_path, target_path)
            except OSError:
                shutil.copy2(source_path, target_path)

    for source_path in source_train_files[:train_shards]:
        link_into_view(source_path)
    for source_path in source_val_files:
        link_into_view(source_path)

    marker_path.write_text(json.dumps({"dataset_name": dataset_name, "train_shards": train_shards}) + "\n", encoding="utf-8")
    return view_path


def _extract_last_match(pattern: str, text: str) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    return matches[-1] if matches else None


def parse_train_metrics(log_text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "logged_train_time_ms": None,
        "final_exact_val_loss": None,
        "final_exact_val_bpb": None,
        "compressed_bytes": None,
        "code_bytes": None,
        "total_bytes": None,
        "peak_memory_mib": None,
        "step": None,
        "iterations": None,
    }
    exact_match = _extract_last_match(
        r"final[^\n]*?_exact[^\n]*?val_loss:(?P<loss>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+)",
        log_text,
    )
    if exact_match is not None:
        metrics["final_exact_val_loss"] = float(exact_match.group("loss"))
        metrics["final_exact_val_bpb"] = float(exact_match.group("bpb"))

    compressed_match = _extract_last_match(r"Serialized model[^\n]*: (?P<bytes>\d+) bytes", log_text)
    if compressed_match is not None:
        metrics["compressed_bytes"] = int(compressed_match.group("bytes"))

    code_match = _extract_last_match(r"Code size: (?P<bytes>\d+) bytes", log_text)
    if code_match is not None:
        metrics["code_bytes"] = int(code_match.group("bytes"))

    total_match = _extract_last_match(r"Total submission size[^\n]*: (?P<bytes>\d+) bytes", log_text)
    if total_match is not None:
        metrics["total_bytes"] = int(total_match.group("bytes"))

    peak_memory_match = _extract_last_match(
        r"peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB",
        log_text,
    )
    if peak_memory_match is not None:
        metrics["peak_memory_mib"] = int(peak_memory_match.group("allocated"))

    train_time_match = _extract_last_match(r"train_time:(?P<ms>[0-9.]+)ms", log_text)
    if train_time_match is not None:
        metrics["logged_train_time_ms"] = int(float(train_time_match.group("ms")))

    step_match = _extract_last_match(
        r"(?:stopping_early:[^\n]* )?step:(?P<step>\d+)/(?P<iterations>\d+)",
        log_text,
    )
    if step_match is not None:
        metrics["step"] = int(step_match.group("step"))
        metrics["iterations"] = int(step_match.group("iterations"))

    return metrics


def stream_subprocess(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        proc.wait()
    return proc.returncode, log_path.read_text(encoding="utf-8", errors="replace")


def default_python_executable(repo_root: Path) -> str:
    if sys.platform == "win32":
        candidate = repo_root / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = repo_root / ".venv" / "bin" / "python"
    return str(candidate if candidate.exists() else Path(sys.executable))


def default_torchrun_executable(repo_root: Path) -> str:
    if sys.platform == "win32":
        candidate = repo_root / ".venv" / "Scripts" / "torchrun.exe"
    else:
        candidate = repo_root / ".venv" / "bin" / "torchrun"
    return str(candidate if candidate.exists() else "torchrun")


def execute_training_run(
    *,
    spec: ExperimentSpec,
    repo_root: Path,
    data_root: Path,
    runs_root: Path,
    checkpoint_root: Path,
    runtime: str,
    gpu_type: str,
    gpu_count: int,
    python_executable: str | None = None,
    torchrun_executable: str | None = None,
) -> dict[str, Any]:
    ensure_run_id(spec)
    runs_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "spec.json", spec.to_dict())

    artifacts = download_challenge_data(
        target_root=data_root,
        variant=spec.data_variant,
        train_shards=spec.train_shards,
    )
    data_path = prepare_dataset_view(
        source_data_path=Path(artifacts["data_path"]),
        target_root=data_root,
        dataset_name=str(artifacts["dataset_name"]),
        train_shards=spec.train_shards,
    )
    python_exe = python_executable or default_python_executable(repo_root)
    torchrun_exe = torchrun_executable or default_torchrun_executable(repo_root)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RUN_ID": spec.run_id,
            "DATA_PATH": str(data_path),
            "TOKENIZER_PATH": artifacts["tokenizer_model_path"],
            "VOCAB_SIZE": str(artifacts["vocab_size"]),
            "SEED": str(spec.seed),
            "CHECKPOINT_PATH": str(checkpoint_root / spec.run_id / "checkpoint.pt"),
            "CHECKPOINT_EVERY": str(spec.checkpoint_every),
            "RESUME_FROM_CHECKPOINT": "1",
        }
    )
    if runtime.startswith("local"):
        env.setdefault("ENABLE_TORCH_COMPILE", "0")
        env.setdefault("SDP_BACKEND", "math")
    else:
        env.setdefault("ENABLE_TORCH_COMPILE", "1")
        env.setdefault("SDP_BACKEND", "flash")
    env.update(spec.normalized_env())

    script_path = spec.resolved_script_path(repo_root)
    if gpu_count > 1:
        command = [torchrun_exe, "--standalone", f"--nproc_per_node={gpu_count}", str(script_path)]
    else:
        command = [python_exe, str(script_path)]

    started = time.perf_counter()
    returncode, log_text = stream_subprocess(
        command=command,
        cwd=run_dir,
        env=env,
        log_path=run_dir / "train.log",
    )
    wallclock_seconds = round(time.perf_counter() - started, 3)
    parsed = parse_train_metrics(log_text)
    result = {
        "timestamp_utc": utc_now_iso(),
        "run_id": spec.run_id,
        "name": spec.name,
        "stage": spec.stage,
        "status": "ok" if returncode == 0 and parsed["final_exact_val_bpb"] is not None else "failed",
        "runtime": runtime,
        "commit": spec.commit,
        "script_path": str(script_path.relative_to(repo_root)),
        "data_variant": spec.data_variant,
        "seed": spec.seed,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "wallclock_seconds": wallclock_seconds,
        "train_shards": spec.train_shards,
        "description": spec.description,
        "returncode": returncode,
        "run_dir": str(run_dir),
        "checkpoint_path": env["CHECKPOINT_PATH"],
        "env_overrides": spec.normalized_env(),
        "command": command,
        "log_text": log_text,
    }
    result.update(parsed)
    write_json(run_dir / "result.json", {k: v for k, v in result.items() if k != "log_text"})
    return result


def append_results_row(result: dict[str, Any], path: Path = RESULTS_TSV_PATH) -> None:
    ensure_experiment_layout()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t")
        writer.writerow({column: result.get(column, "") for column in RESULT_COLUMNS})


def update_best_candidates(result: dict[str, Any], path: Path = BEST_CANDIDATES_PATH) -> None:
    ensure_experiment_layout()
    if result.get("status") != "ok" or result.get("final_exact_val_bpb") is None:
        return
    doc = json.loads(path.read_text(encoding="utf-8"))
    stage = str(result["stage"])
    entry = {
        "run_id": result["run_id"],
        "name": result["name"],
        "stage": stage,
        "runtime": result["runtime"],
        "final_exact_val_bpb": result["final_exact_val_bpb"],
        "final_exact_val_loss": result["final_exact_val_loss"],
        "gpu_type": result["gpu_type"],
        "gpu_count": result["gpu_count"],
        "wallclock_seconds": result["wallclock_seconds"],
        "commit": result["commit"],
        "description": result["description"],
    }
    by_stage = doc.setdefault("by_stage", {})
    stage_entries = [row for row in by_stage.get(stage, []) if row.get("run_id") != result["run_id"]]
    stage_entries.append(entry)
    stage_entries.sort(key=lambda row: row["final_exact_val_bpb"])
    by_stage[stage] = stage_entries[:5]

    overall: list[dict[str, Any]] = []
    for rows in by_stage.values():
        overall.extend(rows)
    overall.sort(key=lambda row: row["final_exact_val_bpb"])
    doc["best_overall"] = overall[:10]
    doc["updated_at"] = utc_now_iso()
    write_json(path, doc)


def materialize_local_run_copy(result: dict[str, Any], *, config_path: Path | None = None) -> Path:
    ensure_experiment_layout()
    run_dir = RUNS_ROOT / str(result["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copy2(config_path, run_dir / "config.json")
    if "log_text" in result:
        (run_dir / "train.log").write_text(result["log_text"], encoding="utf-8", errors="replace")
    write_json(run_dir / "result.json", {k: v for k, v in result.items() if k != "log_text"})
    return run_dir


def record_result(result: dict[str, Any], *, config_path: Path | None = None) -> Path:
    run_dir = materialize_local_run_copy(result, config_path=config_path)
    append_results_row(result)
    update_best_candidates(result)
    return run_dir


def load_best_candidate(stage: str) -> dict[str, Any] | None:
    if not BEST_CANDIDATES_PATH.exists():
        return None
    doc = json.loads(BEST_CANDIDATES_PATH.read_text(encoding="utf-8"))
    stage_rows = doc.get("by_stage", {}).get(stage, [])
    return stage_rows[0] if stage_rows else None
