from __future__ import annotations

from pathlib import Path
from typing import Any


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def init_wandb_run(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    name: str | None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Any | None:
    if not enabled:
        return None

    _try_load_dotenv()
    try:
        import wandb
    except Exception as e:
        raise RuntimeError(
            "W&B is enabled but wandb is not available. Install with `pip install wandb`."
        ) from e

    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config or {},
        tags=tags or [],
    )


def log_metrics(run: Any | None, metrics: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    run.log(metrics, step=step)


def log_dir_artifact(
    run: Any | None,
    *,
    name: str,
    artifact_type: str,
    dir_path: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if run is None:
        return
    if not Path(dir_path).exists():
        return

    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
    artifact.add_dir(dir_path)
    run.log_artifact(artifact)


def log_file_artifact(
    run: Any | None,
    *,
    name: str,
    artifact_type: str,
    file_path: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if run is None:
        return
    if not Path(file_path).exists():
        return

    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
    artifact.add_file(file_path)
    run.log_artifact(artifact)


def finish_wandb_run(run: Any | None) -> None:
    if run is None:
        return
    run.finish()
