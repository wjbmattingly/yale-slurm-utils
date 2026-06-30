"""Smoke tests for the Rich renderables (they must not raise)."""

from __future__ import annotations

import io

from rich.console import Console

from yale_slurm_utils import render
from yale_slurm_utils.models import Job


def _job(job_id: str, user: str = "me", state: str = "RUNNING") -> Job:
    return Job(
        job_id=job_id, partition="gpu_h200", name="train", user=user, state=state,
        num_cpus=8, mem_mb=64 * 1024, gpu_total=1, gpu_types={"h200": 1},
        nodelist="c01n04", time_limit_s=21600, elapsed_s=3600,
    )


def _to_text(renderable, width: int = 200) -> str:
    console = Console(width=width, record=True, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


def test_jobs_selectable_marks_selection():
    jobs = [_job("1"), _job("2"), _job("3")]
    out = _to_text(render.jobs_selectable(jobs, selected=1))
    assert "❯" in out  # the cursor is drawn
    assert "64G" in out  # memory column surfaces


def test_jobs_selectable_empty():
    out = _to_text(render.jobs_selectable([], selected=0))
    assert "No jobs" in out


def test_jobs_table_has_cpu_and_memory():
    out = _to_text(render.jobs_table([_job("1")]))
    assert "64G" in out  # memory value
    assert "c01n04" in out  # node still renders
