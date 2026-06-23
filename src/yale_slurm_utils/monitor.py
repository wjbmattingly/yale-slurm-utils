"""Real-time ``ysu watch`` dashboard.

Polls SLURM on an interval, renders a live Rich dashboard, and detects job
start/finish transitions for the monitored user(s) -- ringing the terminal
bell and appending to the persistent event log on every transition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from . import render
from .events import JobEvent, append_events, load_events, make_event
from .models import Job
from .slurm import SlurmError, get_jobs, get_partitions, gpu_inventory, gpu_usage_by_user


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


def _build_layout() -> Layout:
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="pool", size=6),
        Layout(name="body", ratio=1),
        Layout(name="events", size=14),
        Layout(name="footer", size=1),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    return layout


def _footer(interval: int, monitored: str, bell: bool) -> Text:
    text = Text(justify="center")
    text.append(" monitoring: ", style="dim")
    text.append(monitored, style="cyan")
    text.append(f"   ·   refresh {interval}s", style="dim")
    text.append("   ·   bell ", style="dim")
    text.append("on" if bell else "off", style="green" if bell else "red dim")
    text.append("   ·   press ", style="dim")
    text.append("Ctrl-C", style="bold")
    text.append(" to quit ", style="dim")
    return text


def run_dashboard(
    *,
    user: str | None,
    partition: str | None,
    interval: int = 10,
    bell: bool = True,
    console: Console | None = None,
) -> None:
    console = console or Console()
    monitored = "all users" if user == "*" else (user or "you")
    if partition:
        monitored += f" · {partition}"

    layout = _build_layout()
    prev: _Snapshot | None = None
    error: str | None = None

    with Live(
        layout,
        console=console,
        screen=True,
        refresh_per_second=4,
        redirect_stderr=False,
    ) as live:
        while True:
            now = datetime.now()
            try:
                my_jobs = get_jobs(user=user, partition=partition)
                gpu_classes = gpu_inventory(partition)
                usage = gpu_usage_by_user(partition)
                error = None
            except SlurmError as exc:
                error = str(exc)
                my_jobs, gpu_classes, usage = [], [], {}

            curr = _snapshot(my_jobs)
            new_events = detect_transitions(prev, curr)
            if new_events:
                append_events(new_events)
                if bell:
                    for _ in new_events:
                        console.bell()
            prev = curr

            subtitle = f"watching {monitored}"
            layout["header"].update(render.header(subtitle))
            layout["pool"].update(render.gpu_summary(gpu_classes))

            if error:
                layout["left"].update(
                    Panel(Text(error, style="red"), title="SLURM error",
                          border_style="red")
                )
                layout["right"].update(Panel(Text("")))
            else:
                me = None if user == "*" else (user or _whoami())
                layout["left"].update(render.gpu_users_table(usage, me=me))
                if user == "*":
                    layout["right"].update(render.jobs_table(my_jobs, now))
                else:
                    layout["right"].update(
                        Panel(
                            render.jobs_detail(my_jobs, now),
                            title="Your jobs",
                            title_align="left",
                            border_style="grey37",
                        )
                    )

            layout["events"].update(render.events_table(load_events(limit=12)))
            layout["footer"].update(_footer(interval, monitored, bell))

            live.refresh()
            try:
                time.sleep(interval)
            except KeyboardInterrupt:  # pragma: no cover
                break


def _whoami() -> str:
    from .slurm import current_user

    return current_user()
