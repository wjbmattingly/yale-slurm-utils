"""Real-time ``ysu watch`` dashboard.

Polls SLURM on an interval, renders a live Rich dashboard, and detects job
start/finish transitions for the monitored user(s) -- ringing the terminal
bell and appending to the persistent event log on every transition.

With ``--interactive`` it also reads the keyboard (raw mode) so you can scroll
through your jobs and cancel the highlighted one.
"""

from __future__ import annotations

import os
import select
import sys
import time
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from . import render
from .events import JobEvent, append_events, load_events, make_event
from .models import Job
from .slurm import (
    SlurmError,
    cancel_job,
    current_user,
    get_jobs,
    gpu_inventory,
    gpu_usage_by_user,
)

try:  # raw keyboard input is POSIX-only (fine for HPC login nodes)
    import termios
    import tty

    _RAW_AVAILABLE = True
except ImportError:  # pragma: no cover - non-POSIX
    _RAW_AVAILABLE = False


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


# --------------------------------------------------------------------------- #
# Keyboard input (raw mode)
# --------------------------------------------------------------------------- #
class _RawInput:
    """Read single keypresses without waiting for Enter (cbreak mode).

    ``cbreak`` (not raw) keeps signals working, so Ctrl-C still quits.
    """

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self._old: list | None = None

    def open(self) -> None:
        self._old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def close(self) -> None:
        if self._old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            self._old = None

    def read_key(self, timeout: float) -> str | None:
        """Return a key name, or ``None`` if nothing was pressed in ``timeout``.

        Reads from the raw fd (not Python's buffered ``sys.stdin``) so that
        arrow-key escape sequences (``ESC [ A``) are decoded reliably rather
        than being mistaken for a lone ``ESC``.
        """
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        ch = os.read(self.fd, 1)
        if not ch:  # EOF
            return None
        if ch != b"\x1b":
            return ch.decode(errors="ignore")

        # Escape sequence — pull the rest of it (CSI / SS3 arrows, etc.).
        seq = b""
        while len(seq) < 5:
            ready, _, _ = select.select([self.fd], [], [], 0.05)
            if not ready:
                break
            nxt = os.read(self.fd, 1)
            if not nxt:
                break
            seq += nxt
            if nxt.isalpha() or nxt == b"~":
                break
        if seq[:1] in (b"[", b"O"):
            return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(
                seq[1:2].decode(errors="ignore"), "esc"
            )
        return "esc"


# --------------------------------------------------------------------------- #
# Layout / footer
# --------------------------------------------------------------------------- #
def _build_layout(interactive: bool = False) -> Layout:
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="pool", size=6),
        Layout(name="body", ratio=1),
        Layout(name="events", size=12 if interactive else 14),
        Layout(name="footer", size=2 if interactive else 1),
    )
    if not interactive:
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


def _interactive_footer(
    interval: int,
    monitored: str,
    bell: bool,
    status: tuple[str, str] | None,
    confirm: Job | None,
) -> Group:
    if confirm is not None:
        top = Text(justify="center")
        top.append("Cancel job ", style="bold yellow")
        top.append(confirm.job_id, style="bold")
        top.append(f"  ({confirm.name or 'unnamed'})", style="yellow")
        top.append("?   press ", style="bold yellow")
        top.append("y", style="bold green")
        top.append(" to confirm · any other key to abort", style="bold yellow")
    elif status is not None:
        kind, message = status
        style = {"ok": "green", "err": "bold red", "dim": "dim"}.get(kind, "white")
        top = Text(message, style=style, justify="center")
    else:
        top = Text("monitoring ", style="dim", justify="center")
        top.append(monitored, style="cyan")
        top.append(f"  ·  refresh {interval}s  ·  bell ", style="dim")
        top.append("on" if bell else "off", style="green" if bell else "red dim")

    keys = Text(justify="center")
    for key, label in (
        ("↑/↓ or j/k", "select"),
        ("g/G", "top/bottom"),
        ("c", "cancel"),
        ("q", "quit"),
    ):
        if len(keys):
            keys.append("   ·   ", style="dim")
        keys.append(key, style="bold")
        keys.append(f" {label}", style="dim")
    return Group(top, keys)


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
def run_dashboard(
    *,
    user: str | None,
    partition: str | None,
    interval: int = 10,
    bell: bool = True,
    console: Console | None = None,
    interactive: bool = False,
) -> None:
    console = console or Console()
    monitored = "all users" if user == "*" else (user or "you")
    if partition:
        monitored += f" · {partition}"

    can_interact = interactive and _RAW_AVAILABLE and sys.stdin.isatty()
    if interactive and not can_interact:
        console.print(
            "[yellow]Interactive mode needs a real terminal; "
            "falling back to read-only watch.[/]"
        )

    me = current_user() or None
    layout = _build_layout(interactive=can_interact)
    prev: _Snapshot | None = None
    error: str | None = None
    my_jobs: list[Job] = []
    gpu_classes: list = []
    usage: dict = {}

    sel_idx = 0
    sel_id: str | None = None
    status: tuple[str, str] | None = None
    confirm: Job | None = None

    def _fetch() -> None:
        nonlocal prev, error, my_jobs, gpu_classes, usage
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

    def _render() -> None:
        now = datetime.now()
        subtitle = f"watching {monitored}"
        if can_interact:
            subtitle += " · interactive"
        layout["header"].update(render.header(subtitle))
        layout["pool"].update(render.gpu_summary(gpu_classes))

        if can_interact:
            if error:
                layout["body"].update(
                    Panel(Text(error, style="red"), title="SLURM error",
                          border_style="red")
                )
            else:
                title = "Your jobs" if user != "*" else "Jobs"
                layout["body"].update(
                    render.jobs_selectable(my_jobs, sel_idx, now, me=me, title=title)
                )
            layout["footer"].update(
                _interactive_footer(interval, monitored, bell, status, confirm)
            )
        else:
            if error:
                layout["left"].update(
                    Panel(Text(error, style="red"), title="SLURM error",
                          border_style="red")
                )
                layout["right"].update(Panel(Text("")))
            else:
                users_me = None if user == "*" else (user or me)
                layout["left"].update(render.gpu_users_table(usage, me=users_me))
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
            layout["footer"].update(_footer(interval, monitored, bell))

        layout["events"].update(
            render.events_table(load_events(limit=10 if can_interact else 12))
        )

    raw = _RawInput() if can_interact else None
    with Live(
        layout,
        console=console,
        screen=True,
        refresh_per_second=4,
        auto_refresh=not can_interact,
        redirect_stderr=False,
    ) as live:
        if raw is not None:
            raw.open()
        try:
            next_refresh = 0.0
            while True:
                if time.monotonic() >= next_refresh:
                    _fetch()
                    next_refresh = time.monotonic() + interval

                ids = [j.job_id for j in my_jobs]
                if can_interact:
                    if sel_id in ids:
                        sel_idx = ids.index(sel_id)
                    else:
                        sel_idx = min(sel_idx, len(ids) - 1) if ids else 0
                    sel_id = ids[sel_idx] if ids else None

                _render()
                live.refresh()

                if not can_interact:
                    try:
                        time.sleep(interval)
                    except KeyboardInterrupt:
                        break
                    continue

                timeout = max(0.0, next_refresh - time.monotonic())
                key = raw.read_key(min(timeout, 1.0))  # cap so the clock ticks
                if key is None:
                    continue

                if confirm is not None:
                    if key in ("y", "Y"):
                        try:
                            cancel_job(confirm.job_id)
                            status = ("ok", f"Cancelled job {confirm.job_id}.")
                        except SlurmError as exc:
                            status = ("err", str(exc))
                        next_refresh = 0.0  # reflect the change immediately
                    else:
                        status = ("dim", "Cancel aborted.")
                    confirm = None
                    continue

                if key in ("q", "Q"):
                    break
                elif key in ("down", "j"):
                    if ids:
                        sel_idx = min(sel_idx + 1, len(ids) - 1)
                        sel_id = ids[sel_idx]
                        status = None
                elif key in ("up", "k"):
                    if ids:
                        sel_idx = max(sel_idx - 1, 0)
                        sel_id = ids[sel_idx]
                        status = None
                elif key == "g":
                    if ids:
                        sel_idx, sel_id = 0, ids[0]
                elif key == "G":
                    if ids:
                        sel_idx = len(ids) - 1
                        sel_id = ids[sel_idx]
                elif key in ("c", "x", "d"):
                    if not ids:
                        status = ("dim", "No jobs to cancel.")
                    else:
                        job = my_jobs[sel_idx]
                        if not can_cancel(job, me):
                            status = (
                                "err",
                                f"You can only cancel your own jobs (not {job.user}'s).",
                            )
                        else:
                            confirm = job
        except KeyboardInterrupt:  # pragma: no cover
            pass
        finally:
            if raw is not None:
                raw.close()
