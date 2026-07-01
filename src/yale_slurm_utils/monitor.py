"""Job-transition detection shared by the watch UI.

The live dashboard itself lives in :mod:`yale_slurm_utils.tui` (a Textual app).
This module keeps the pure, easily-tested pieces: snapshotting the queue and
diffing two snapshots into start/finish events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .events import JobEvent, make_event
from .models import Job


@dataclass
class _Snapshot:
    states: dict[str, str]  # job_id -> state
    jobs: dict[str, Job]  # job_id -> job


def _snapshot(jobs: list[Job]) -> _Snapshot:
    return _Snapshot(
        states={j.job_id: j.state.upper() for j in jobs},
        jobs={j.job_id: j for j in jobs},
    )


def detect_transitions(prev: _Snapshot | None, curr: _Snapshot) -> list[JobEvent]:
    """Compare two snapshots and emit started/finished events.

    * started  -> a job is RUNNING now and either is new or was PENDING before.
    * finished -> a job that we previously tracked is no longer in the queue
      (SLURM drops completed jobs from ``squeue``).
    """
    if prev is None:
        return []
    now = datetime.now()
    events: list[JobEvent] = []

    for job_id, state in curr.states.items():
        was = prev.states.get(job_id)
        if state == "RUNNING" and was != "RUNNING":
            events.append(make_event("started", curr.jobs[job_id], now))

    for job_id, was in prev.states.items():
        if job_id not in curr.states:
            events.append(make_event("finished", prev.jobs[job_id], now))

    return events


def can_cancel(job: Job, me: str | None) -> bool:
    """You may only cancel your own jobs."""
    if me is None:
        return True
    return not job.user or job.user == me
