"""Persistent log of job start/finish events.

Events are appended as JSON lines to an XDG state directory so that the log
survives across ``ysu watch`` sessions.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .models import Job


@dataclass
class JobEvent:
    kind: str  # "started" | "finished"
    job_id: str
    name: str
    user: str
    partition: str
    gpu_label: str
    timestamp: str  # ISO 8601

    @property
    def when(self) -> datetime:
        try:
            return datetime.fromisoformat(self.timestamp)
        except ValueError:
            return datetime.now()


def log_path() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state"
    )
    path = Path(base) / "yale-slurm-utils"
    path.mkdir(parents=True, exist_ok=True)
    return path / "events.jsonl"


def make_event(kind: str, job: Job, when: datetime | None = None) -> JobEvent:
    return JobEvent(
        kind=kind,
        job_id=job.job_id,
        name=job.name or "(unnamed)",
        user=job.user,
        partition=job.partition,
        gpu_label=job.gpu_label,
        timestamp=(when or datetime.now()).isoformat(timespec="seconds"),
    )


def append_events(events: list[JobEvent], path: Path | None = None) -> None:
    if not events:
        return
    path = path or log_path()
    with path.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(asdict(event)) + "\n")


def load_events(limit: int | None = None, path: Path | None = None) -> list[JobEvent]:
    path = path or log_path()
    if not path.exists():
        return []
    events: list[JobEvent] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                events.append(JobEvent(**data))
            except (json.JSONDecodeError, TypeError):
                continue
    if limit is not None:
        return events[-limit:]
    return events


def clear_events(path: Path | None = None) -> None:
    path = path or log_path()
    if path.exists():
        path.unlink()
