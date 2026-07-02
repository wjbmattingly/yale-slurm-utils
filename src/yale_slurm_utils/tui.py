"""Textual UI for ``ysu watch`` (both read-only and interactive).

Everything is still rendered with the existing Rich renderables from
:mod:`render`; Textual just wraps them in scrollable, focusable panes so you can
Tab between the tables, scroll each one, and (with ``--interactive``) move a
cursor through your jobs and cancel the selected one.
"""

from __future__ import annotations

from datetime import datetime

from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from . import render, theme
from .events import append_events, load_events
from .models import Job
from .monitor import _snapshot, can_cancel, detect_transitions
from .slurm import (
    SlurmError,
    cancel_job,
    current_user,
    get_jobs,
    gpu_inventory,
    gpu_usage_by_user,
)

# Lines a Rich table prints before its first data row (title, top border,
# header, header separator). Used to keep the selected row scrolled into view.
_TABLE_HEADER_LINES = 4


class ConfirmCancelScreen(ModalScreen[bool]):
    """A small modal to confirm cancelling a job (keyboard or mouse)."""

    BINDINGS = [
        Binding("y", "yes", "Yes"),
        Binding("n", "no", "No"),
        Binding("escape", "no", "No"),
    ]

    def __init__(self, job: Job) -> None:
        super().__init__()
        self._job = job

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label(
                f"Cancel job {self._job.job_id} "
                f"({self._job.name or 'unnamed'})?",
                id="confirm-label",
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes, cancel", variant="error", id="yes")
                yield Button("No", variant="primary", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class WatchApp(App):
    """Live SLURM dashboard."""

    CSS = """
    Screen { layout: vertical; }
    #header { height: 3; }
    #pool { height: 6; }
    #body { height: 1fr; }
    #left-scroll, #right-scroll { width: 1fr; }
    #right-scroll { height: 1fr; }
    #body #right-scroll { border-left: solid $panel; }
    #events-scroll { height: 14; }
    #footer { height: 2; }
    VerticalScroll:focus { border-left: solid $accent; }
    #confirm-box {
        width: auto; height: auto; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #confirm-buttons { height: auto; align: center middle; margin-top: 1; }
    #confirm-buttons Button { margin: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("tab", "focus_next", "Next table", show=False),
        Binding("shift+tab", "focus_previous", "Prev table", show=False),
    ]

    def __init__(
        self,
        *,
        user: str | None,
        partition: str | None,
        bell: bool = True,
        interactive: bool = False,
    ) -> None:
        super().__init__()
        self.user = user
        self.partition = partition
        self._bell = bell
        self.interactive = interactive
        self.me = current_user() or None
        self.updated_at: datetime | None = None

        self.jobs: list[Job] = []
        self.gpus: list = []
        self.usage: dict = {}
        self.err: str | None = None
        self._prev = None
        self.sel_idx = 0
        self.sel_id: str | None = None
        self._status: tuple[str, str] | None = None

    # -- layout ------------------------------------------------------------ #
    def compose(self) -> ComposeResult:
        yield Static(id="header")
        yield Static(id="pool")
        if self.interactive:
            # Interactive is focused on managing your own jobs — no users table.
            yield VerticalScroll(Static(id="right"), id="right-scroll")
        else:
            with Horizontal(id="body"):
                yield VerticalScroll(Static(id="left"), id="left-scroll")
                yield VerticalScroll(Static(id="right"), id="right-scroll")
        yield VerticalScroll(Static(id="events"), id="events-scroll")
        yield Static(id="footer")

    def on_mount(self) -> None:
        # Match Textual's chrome (panel/surface backgrounds) to our theme mode.
        self.theme = (
            "textual-light" if theme.active().mode == "light" else "textual-dark"
        )
        # Single snapshot only. We deliberately do NOT poll on a timer: sampling
        # SLURM (sinfo/squeue) on an interval loads the scheduler for everyone.
        # The user refreshes on demand with `r` (or by re-running the command).
        self._poll()

    def action_refresh(self) -> None:
        self._status = ("dim", "Refreshing…")
        self.render_all()
        self._poll()

    # -- data --------------------------------------------------------------- #
    @work(thread=True, exclusive=True)
    def _poll(self) -> None:
        try:
            jobs = get_jobs(user=self.user, partition=self.partition)
            gpus = gpu_inventory(self.partition)
            # The per-user table is only shown in read-only mode.
            usage = {} if self.interactive else gpu_usage_by_user(self.partition)
            err = None
        except SlurmError as exc:
            jobs, gpus, usage, err = [], [], {}, str(exc)
        self.call_from_thread(self._apply, jobs, gpus, usage, err)

    def _apply(self, jobs, gpus, usage, err) -> None:
        curr = _snapshot(jobs)
        new_events = detect_transitions(self._prev, curr)
        if new_events:
            append_events(new_events)
            if self._bell:
                for _ in new_events:
                    self.bell()
        self._prev = curr
        self.jobs, self.gpus, self.usage, self.err = jobs, gpus, usage, err
        self.updated_at = datetime.now()
        self._reconcile_selection()
        self.render_all()

    def _reconcile_selection(self) -> None:
        ids = [j.job_id for j in self.jobs]
        if self.sel_id in ids:
            self.sel_idx = ids.index(self.sel_id)
        else:
            self.sel_idx = min(self.sel_idx, len(ids) - 1) if ids else 0
        self.sel_id = ids[self.sel_idx] if ids else None

    # -- rendering ---------------------------------------------------------- #
    def _subtitle(self) -> str:
        monitored = "all users" if self.user == "*" else (self.user or "you")
        if self.partition:
            monitored += f" · {self.partition}"
        text = f"snapshot · {monitored}"
        if self.interactive:
            text += " · interactive"
        return text

    def _footer_renderable(self) -> Text:
        if self.interactive:
            if self._status is not None:
                kind, message = self._status
                style = {"ok": "green", "err": "bold red", "dim": "dim"}.get(kind, "")
                line = Text(message, style=style, justify="center")
            else:
                line = Text(justify="center")
                for key, label in (
                    ("↑/↓ or j/k", "select"),
                    ("c", "cancel"),
                    ("r", "refresh"),
                    ("Tab", "switch table"),
                    ("q", "quit"),
                ):
                    if len(line):
                        line.append("   ·   ", style="dim")
                    line.append(key, style="bold")
                    line.append(f" {label}", style="dim")
            return line
        stamp = self.updated_at.strftime("%H:%M:%S") if self.updated_at else "—"
        line = Text(justify="center")
        line.append("snapshot ", style="dim")
        line.append(stamp, style="cyan")
        line.append("  ·  no auto-refresh  ·  ", style="dim")
        line.append("r", style="bold")
        line.append(" refresh  ·  ", style="dim")
        line.append("Tab", style="bold")
        line.append(" switch table  ·  ", style="dim")
        line.append("q", style="bold")
        line.append(" quit", style="dim")
        return line

    def render_all(self) -> None:
        if not self.is_mounted:
            return
        now = datetime.now()
        self.query_one("#header", Static).update(render.header(self._subtitle()))
        self.query_one("#pool", Static).update(render.gpu_summary(self.gpus))

        if self.interactive:
            # Just your jobs, full width, with a movable cursor.
            if self.err:
                self.query_one("#right", Static).update(
                    Panel(Text(self.err, style="red"), title="SLURM error",
                          border_style="red")
                )
            else:
                title = "Your jobs" if self.user != "*" else "Jobs"
                self.query_one("#right", Static).update(
                    render.jobs_selectable(
                        self.jobs, self.sel_idx, now, me=self.me, title=title
                    )
                )
        elif self.err:
            self.query_one("#left", Static).update(
                Panel(Text(self.err, style="red"), title="SLURM error",
                      border_style="red")
            )
            self.query_one("#right", Static).update(Text(""))
        else:
            users_me = None if self.user == "*" else (self.user or self.me)
            self.query_one("#left", Static).update(
                render.gpu_users_table(self.usage, me=users_me)
            )
            if self.user == "*":
                self.query_one("#right", Static).update(
                    render.jobs_table(self.jobs, now)
                )
            else:
                self.query_one("#right", Static).update(
                    render.jobs_detail(self.jobs, now)
                )

        self.query_one("#events", Static).update(
            render.events_table(load_events(limit=12))
        )
        self.query_one("#footer", Static).update(self._footer_renderable())

    # -- interaction -------------------------------------------------------- #
    def _ensure_selection_visible(self) -> None:
        try:
            scroller = self.query_one("#right-scroll", VerticalScroll)
        except Exception:
            return
        line = _TABLE_HEADER_LINES + self.sel_idx
        top = scroller.scroll_offset.y
        height = scroller.size.height
        if line < top:
            scroller.scroll_to(y=line, animate=False)
        elif line >= top + height:
            scroller.scroll_to(y=line - height + 1, animate=False)

    def _move(self, delta: int) -> None:
        if not self.jobs:
            return
        self.sel_idx = max(0, min(self.sel_idx + delta, len(self.jobs) - 1))
        self.sel_id = self.jobs[self.sel_idx].job_id
        self._status = None
        self.render_all()
        self._ensure_selection_visible()

    def _request_cancel(self) -> None:
        if not self.jobs:
            self._status = ("dim", "No jobs to cancel.")
            self.render_all()
            return
        job = self.jobs[self.sel_idx]
        if not can_cancel(job, self.me):
            self._status = (
                "err",
                f"You can only cancel your own jobs (not {job.user}'s).",
            )
            self.render_all()
            return

        def _after(confirmed: bool | None) -> None:
            if confirmed:
                self._do_cancel(job)
            else:
                self._status = ("dim", "Cancel aborted.")
                self.render_all()

        self.push_screen(ConfirmCancelScreen(job), _after)

    @work(thread=True)
    def _do_cancel(self, job: Job) -> None:
        try:
            cancel_job(job.job_id)
            result = ("ok", f"Cancelled job {job.job_id}.")
        except SlurmError as exc:
            result = ("err", str(exc))
        self.call_from_thread(self._after_cancel, result)

    def _after_cancel(self, result: tuple[str, str]) -> None:
        self._status = result
        self._poll()

    def on_key(self, event) -> None:
        if not self.interactive:
            return
        try:
            right = self.query_one("#right-scroll", VerticalScroll)
        except Exception:
            return
        if self.focused is not right:
            return
        if event.key in ("down", "j"):
            self._move(1)
            event.stop()
        elif event.key in ("up", "k"):
            self._move(-1)
            event.stop()
        elif event.key in ("c", "x"):
            self._request_cancel()
            event.stop()


def run_dashboard(
    *,
    user: str | None,
    partition: str | None,
    bell: bool = True,
    console=None,  # accepted for API compatibility; Textual manages output
    interactive: bool = False,
) -> None:
    WatchApp(
        user=user,
        partition=partition,
        bell=bell,
        interactive=interactive,
    ).run()
