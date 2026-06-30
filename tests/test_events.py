"""Tests for job event detection and the persistent log."""

from __future__ import annotations

from yale_slurm_utils.events import append_events, load_events, make_event
from yale_slurm_utils.models import Job
from yale_slurm_utils.monitor import _snapshot, can_cancel, detect_transitions


def _job(job_id: str, state: str) -> Job:
    return Job(
        job_id=job_id, partition="gpu_h200", name="t", user="me", state=state,
        num_cpus=8, mem_mb=180 * 1024,
    )


def test_pending_to_running_emits_started():
    prev = _snapshot([_job("1", "PENDING")])
    curr = _snapshot([_job("1", "RUNNING")])
    events = detect_transitions(prev, curr)
    assert [e.kind for e in events] == ["started"]
    assert events[0].job_id == "1"


def test_disappearing_job_emits_finished():
    prev = _snapshot([_job("1", "RUNNING")])
    curr = _snapshot([])
    events = detect_transitions(prev, curr)
    assert [e.kind for e in events] == ["finished"]


def test_new_running_job_emits_started():
    prev = _snapshot([])
    curr = _snapshot([_job("2", "RUNNING")])
    assert [e.kind for e in detect_transitions(prev, curr)] == ["started"]


def test_no_change_no_events():
    snap = _snapshot([_job("1", "RUNNING")])
    assert detect_transitions(snap, snap) == []


def test_first_snapshot_is_silent():
    # No prior state -> never spam events on startup.
    assert detect_transitions(None, _snapshot([_job("1", "RUNNING")])) == []


def test_can_cancel_only_your_own_jobs():
    mine = Job(job_id="1", partition="gpu", name="t", user="me", state="RUNNING")
    theirs = Job(job_id="2", partition="gpu", name="t", user="someone", state="RUNNING")
    assert can_cancel(mine, "me") is True
    assert can_cancel(theirs, "me") is False
    # Unknown "me" (e.g. couldn't resolve user) doesn't block.
    assert can_cancel(theirs, None) is True


def test_log_roundtrip(tmp_path):
    path = tmp_path / "events.jsonl"
    event = make_event("started", _job("9", "RUNNING"))
    append_events([event], path=path)
    loaded = load_events(path=path)
    assert len(loaded) == 1
    assert loaded[0].job_id == "9"
    assert loaded[0].kind == "started"
    assert loaded[0].cpus == 8
    assert loaded[0].mem_mb == 180 * 1024


def test_log_loads_legacy_lines_without_cpu_mem(tmp_path):
    # Lines written before cpus/mem were tracked must still load.
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"kind": "started", "job_id": "7", "name": "old", "user": "me", '
        '"partition": "gpu", "gpu_label": "1x gpu", "timestamp": "2026-01-01T00:00:00"}\n'
    )
    loaded = load_events(path=path)
    assert len(loaded) == 1
    assert loaded[0].job_id == "7"
    assert loaded[0].cpus == 0
    assert loaded[0].mem_mb is None
