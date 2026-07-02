"""Smoke tests for the Textual watch app using the headless pilot."""

from __future__ import annotations

import asyncio

from yale_slurm_utils import tui
from yale_slurm_utils.models import Job


def _sample_jobs():
    return [
        Job(job_id="1", partition="gpu", name="a", user="me", state="RUNNING",
            num_cpus=8, mem_mb=64 * 1024, gpu_total=1, gpu_types={"h200": 1},
            nodelist="c01", time_limit_s=3600, elapsed_s=600),
        Job(job_id="2", partition="gpu", name="b", user="me", state="RUNNING",
            num_cpus=4, mem_mb=32 * 1024, gpu_total=1, gpu_types={"h200": 1},
            nodelist="c02", time_limit_s=3600, elapsed_s=60),
        Job(job_id="3", partition="gpu", name="c", user="other", state="RUNNING",
            num_cpus=2, mem_mb=16 * 1024),
    ]


def _patch(monkeypatch, jobs):
    monkeypatch.setattr(tui, "get_jobs", lambda **kw: list(jobs))
    monkeypatch.setattr(tui, "gpu_inventory", lambda p: [])
    monkeypatch.setattr(tui, "gpu_usage_by_user", lambda p: {})
    monkeypatch.setattr(tui, "current_user", lambda: "me")


def test_interactive_selection_moves(monkeypatch):
    _patch(monkeypatch, _sample_jobs())
    app = tui.WatchApp(user="me", partition=None,
                       bell=False, interactive=True)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.set_focus(app.query_one("#right-scroll"))
            assert app.sel_idx == 0
            await pilot.press("down")
            assert app.sel_idx == 1
            await pilot.press("j")
            assert app.sel_idx == 2
            await pilot.press("up")
            assert app.sel_idx == 1

    asyncio.run(scenario())


def test_cancel_others_job_is_blocked(monkeypatch):
    jobs = _sample_jobs()
    _patch(monkeypatch, jobs)
    cancelled = []
    monkeypatch.setattr(tui, "cancel_job", lambda jid: cancelled.append(jid))
    app = tui.WatchApp(user="*", partition=None,
                       bell=False, interactive=True)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            # Select the third job (owned by "other").
            app.set_focus(app.query_one("#right-scroll"))
            app.sel_idx = 2
            app.sel_id = "3"
            app._request_cancel()
            await pilot.pause()
            assert app._status is not None and app._status[0] == "err"
            assert cancelled == []

    asyncio.run(scenario())


def test_no_auto_refresh_but_manual_refresh_works(monkeypatch):
    calls = {"n": 0}
    jobs = _sample_jobs()

    def _get_jobs(**kw):
        calls["n"] += 1
        return list(jobs)

    monkeypatch.setattr(tui, "get_jobs", _get_jobs)
    monkeypatch.setattr(tui, "gpu_inventory", lambda p: [])
    monkeypatch.setattr(tui, "gpu_usage_by_user", lambda p: {})
    monkeypatch.setattr(tui, "current_user", lambda: "me")
    app = tui.WatchApp(user="me", partition=None, bell=False, interactive=False)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            # One initial snapshot, no timer-driven polls.
            assert calls["n"] == 1
            await pilot.press("r")
            await pilot.pause()
            await pilot.pause()
            assert calls["n"] == 2

    asyncio.run(scenario())


def test_readonly_mode_ignores_selection_keys(monkeypatch):
    _patch(monkeypatch, _sample_jobs())
    app = tui.WatchApp(user="me", partition=None,
                       bell=False, interactive=False)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.set_focus(app.query_one("#right-scroll"))
            await pilot.press("down")
            # No selection cursor in read-only mode.
            assert app.sel_idx == 0

    asyncio.run(scenario())
