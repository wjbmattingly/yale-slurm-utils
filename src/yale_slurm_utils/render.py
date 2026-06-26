"""Rich renderables: turn the data models into beautiful console output."""

from __future__ import annotations

from datetime import datetime

from rich.align import Align
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import theme
from .alloc import GpuOption, GpuOptions, ResolvedRequest
from .events import JobEvent
from .models import GpuClass, Job, Partition
from .parsing import humanize_mb, humanize_seconds

CLUSTER_NAME = "Bouchet"


# --------------------------------------------------------------------------- #
# Headers
# --------------------------------------------------------------------------- #
def header(subtitle: str = "") -> Panel:
    title = Text(no_wrap=True)
    title.append("Yale", style=f"bold {theme.YALE_BLUE_BRIGHT}")
    title.append(" SLURM ", style="bold white")
    title.append("Utils", style=f"bold {theme.YALE_BLUE_BRIGHT}")
    title.append(f" · {CLUSTER_NAME}", style="dim")
    if subtitle:
        title.append(f" · {subtitle}", style="italic cyan")

    now = datetime.now().strftime("%a %d %b  %H:%M:%S")
    line = Table.grid(expand=True, padding=(0, 2))
    line.add_column(justify="left", ratio=1, no_wrap=True)
    line.add_column(justify="right", no_wrap=True)
    line.add_row(title, Text(now, style="dim", no_wrap=True))
    return Panel(line, border_style=theme.YALE_BLUE_BRIGHT, padding=(0, 1))


# --------------------------------------------------------------------------- #
# Partitions
# --------------------------------------------------------------------------- #
def partitions_table(partitions: list[Partition], gpu_only: bool = False) -> Table:
    table = Table(
        title="Partitions",
        title_style="bold",
        header_style="bold white",
        border_style="grey37",
        expand=True,
        row_styles=["", "on grey11"],
    )
    table.add_column("Partition", style="bold", no_wrap=True)
    table.add_column("Nodes", justify="right")
    table.add_column("State (idle/mix/alloc/down)", justify="left")
    table.add_column("CPU usage", justify="left")
    table.add_column("CPUs A/T", justify="right", no_wrap=True)
    table.add_column("Memory", justify="right", no_wrap=True)
    table.add_column("GPUs (free/total)", justify="left")

    for part in partitions:
        if gpu_only and not part.has_gpus:
            continue
        counts = part.state_counts()
        state_text = Text()
        for label, key, style in (
            ("idle", "idle", "green"),
            ("mix", "mixed", "yellow"),
            ("alloc", "allocated", "bright_red"),
            ("down", "down", "red dim"),
        ):
            value = counts.get(key, 0)
            if value:
                if len(state_text):
                    state_text.append(" ")
                state_text.append(f"{value}{label[0]}", style=style)
        if not len(state_text):
            state_text = Text("-", style="dim")

        name = Text(part.clean_name, style="bold")
        if part.is_default:
            name.append("  (default)", style="dim italic")

        if part.has_gpus:
            free = part.free_gpus
            total = part.total_gpus
            gpu_cell = Text()
            gpu_cell.append(theme.usage_bar(part.used_gpus, total, width=12))
            gpu_cell.append(
                f" {free}/{total}",
                style=theme.availability_style(free, total),
            )
        else:
            gpu_cell = Text("-", style="dim")

        table.add_row(
            name,
            str(part.node_count),
            state_text,
            theme.usage_bar(part.cpus_alloc, part.cpus_total, width=14),
            f"{part.cpus_alloc}/{part.cpus_total}",
            humanize_mb(part.mem_total_mb),
            gpu_cell,
        )
    return table


# --------------------------------------------------------------------------- #
# GPUs
# --------------------------------------------------------------------------- #
def gpu_table(gpu_classes: list[GpuClass], free_only: bool = False) -> Table:
    title = "Free GPUs" if free_only else "GPU Availability"
    table = Table(
        title=title,
        title_style="bold",
        header_style="bold white",
        border_style="grey37",
        expand=True,
        row_styles=["", "on grey11"],
    )
    table.add_column("Partition", style="bold", no_wrap=True)
    table.add_column("GPU type", no_wrap=True)
    table.add_column("Utilisation", justify="left")
    table.add_column("Used", justify="right", no_wrap=True)
    table.add_column("Free", justify="right", no_wrap=True)
    table.add_column("Total", justify="right", no_wrap=True)
    table.add_column("Nodes", justify="right", no_wrap=True)

    shown = False
    for gc in gpu_classes:
        if free_only and gc.free <= 0:
            continue
        shown = True
        gtype = Text(gc.gpu_type, style=theme.gpu_color(gc.gpu_type))
        free_text = Text(str(gc.free), style=theme.availability_style(gc.free, gc.total))
        bar = Text()
        bar.append(theme.usage_bar(gc.used, gc.total, width=18))
        bar.append(f" {gc.util_pct:>3.0f}%", style=theme.util_style(gc.util_pct))
        nodes_label = str(gc.nodes_total)
        if gc.nodes_unavailable:
            nodes_label += f" ({gc.nodes_unavailable}↓)"
        table.add_row(
            gc.partition,
            gtype,
            bar,
            str(gc.used),
            free_text,
            str(gc.total),
            nodes_label,
        )

    if not shown:
        message = "No free GPUs right now." if free_only else "No GPUs found."
        table.add_row(Text(message, style="dim"), "", "", "", "", "", "")
    return table


def gpu_summary(gpu_classes: list[GpuClass]) -> Panel:
    """A compact totals-by-type panel (free / total across the cluster)."""
    totals: dict[str, list[int]] = {}
    for gc in gpu_classes:
        agg = totals.setdefault(gc.gpu_type, [0, 0])
        agg[0] += gc.free
        agg[1] += gc.total

    cards: list[RenderableType] = []
    for gpu_type, (free, total) in sorted(totals.items()):
        color = theme.gpu_color(gpu_type)
        body = Text(justify="center")
        body.append(f"{free}", style=f"bold {theme.availability_style(free, total)}")
        body.append(f" / {total}\n", style="dim")
        body.append("free", style="dim")
        cards.append(
            Panel(
                body,
                title=Text(gpu_type, style=color),
                border_style=color,
                padding=(0, 1),
                width=22,
            )
        )
    if not cards:
        cards.append(Text("No GPUs found.", style="dim"))
    return Panel(
        Columns(cards, expand=False, equal=False),
        title="GPU pool (free / total)",
        title_align="left",
        border_style="grey37",
    )


def gpu_users_table(usage: dict[str, dict[str, int]], me: str | None = None) -> Table:
    table = Table(
        title="Who's using which GPUs",
        title_style="bold",
        header_style="bold white",
        border_style="grey37",
        expand=True,
        row_styles=["", "on grey11"],
    )
    table.add_column("User", style="bold", no_wrap=True)
    table.add_column("Total", justify="right", no_wrap=True)
    table.add_column("GPUs", justify="left")

    ranked = sorted(usage.items(), key=lambda kv: sum(kv[1].values()), reverse=True)
    if not ranked:
        table.add_row(Text("No GPUs in use.", style="dim"), "", "")
        return table

    for user, types in ranked:
        total = sum(types.values())
        user_text = Text(user, style="bold cyan" if user == me else "bold")
        if user == me:
            user_text.append("  (you)", style="italic green")
        table.add_row(user_text, str(total), theme.gpu_chiplets(types))
    return table


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def jobs_table(jobs: list[Job], now: datetime | None = None) -> Table:
    now = now or datetime.now()
    table = Table(
        title="Jobs",
        title_style="bold",
        header_style="bold white",
        border_style="grey37",
        expand=True,
        row_styles=["", "on grey11"],
    )
    table.add_column("Job ID", style="bold", no_wrap=True)
    table.add_column("Name", no_wrap=True, max_width=22)
    table.add_column("Partition", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("GPUs", no_wrap=True)
    table.add_column("Node", no_wrap=True)
    table.add_column("Progress", justify="left")
    table.add_column("Time left", justify="right", no_wrap=True)

    if not jobs:
        table.add_row(Text("No jobs.", style="dim"), "", "", "", "", "", "", "")
        return table

    for job in jobs:
        progress = _progress_cell(job, now)
        time_left = _time_left_cell(job, now)
        where = job.nodelist or (job.reason or "-")
        table.add_row(
            job.job_id,
            job.name or "(unnamed)",
            job.partition,
            Text(job.state, style=theme.state_style(job.state)),
            theme.gpu_chiplets(job.gpu_types) if job.gpu_total else Text("-", "dim"),
            where,
            progress,
            time_left,
        )
    return table


def _progress_cell(job: Job, now: datetime) -> Text:
    pct = job.percent_elapsed(now)
    if pct is None:
        if job.is_pending:
            return Text("queued", style="bright_yellow")
        return Text("-", style="dim")
    cell = Text()
    cell.append(theme.percent_bar(pct, width=16))
    cell.append(f" {pct:>3.0f}%", style="bold")
    return cell


def _time_left_cell(job: Job, now: datetime) -> Text:
    if job.is_pending:
        return Text("-", style="dim")
    left = job.time_left_s(now)
    if left is None:
        return Text("∞", style="dim")
    style = "bright_red" if left < 1800 else ("yellow" if left < 7200 else "green")
    return Text(humanize_seconds(left), style=style)


def job_panel(job: Job, now: datetime | None = None) -> Panel:
    """A detailed single-job panel including working dir, logs and command."""
    now = now or datetime.now()
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", justify="right", no_wrap=True)
    grid.add_column(overflow="fold")

    grid.add_row("Partition", Text(job.partition, style="white"))
    grid.add_row(
        "State",
        Text(job.state, style=theme.state_style(job.state)),
    )
    if job.gpu_total:
        grid.add_row("GPUs", theme.gpu_chiplets(job.gpu_types))
    grid.add_row("Resources", f"{job.num_nodes} node(s) · {job.num_cpus} CPU(s)")

    if job.is_pending:
        grid.add_row("Reason", Text(job.reason or "-", style="bright_yellow"))
        if job.start_time:
            grid.add_row(
                "Est. start", job.start_time.strftime("%a %d %b %H:%M")
            )
    else:
        grid.add_row("Node(s)", job.nodelist or "-")
        pct = job.percent_elapsed(now)
        if pct is not None:
            bar = Text()
            bar.append(theme.percent_bar(pct, width=24))
            bar.append(f"  {pct:>3.0f}%", style="bold")
            grid.add_row("Elapsed", bar)
        left = job.time_left_s(now)
        if left is not None:
            elapsed = humanize_seconds(job.elapsed_s)
            limit = humanize_seconds(job.time_limit_s)
            style = "bright_red" if left < 1800 else "green"
            tl = Text()
            tl.append(humanize_seconds(left), style=f"bold {style}")
            tl.append(f"  ({elapsed} / {limit})", style="dim")
            grid.add_row("Time left", tl)
        if job.end_time:
            grid.add_row("Ends", job.end_time.strftime("%a %d %b %H:%M"))

    if job.workdir:
        grid.add_row("Work dir", Text(job.workdir, style="cyan"))
    if job.command:
        grid.add_row("Command", Text(job.command, style="white"))
    if job.stdout:
        grid.add_row("Log (out)", Text(job.stdout, style="green"))
    if job.stderr and job.stderr != job.stdout:
        grid.add_row("Log (err)", Text(job.stderr, style="yellow"))

    title = Text()
    title.append(f"#{job.job_id}  ", style="bold")
    title.append(job.name or "(unnamed)", style="bold white")
    border = theme.state_style(job.state)
    return Panel(grid, title=title, title_align="left", border_style=border)


def jobs_detail(jobs: list[Job], now: datetime | None = None) -> RenderableType:
    if not jobs:
        return Panel(Text("No jobs found.", style="dim"), border_style="grey37")
    return Group(*[job_panel(job, now) for job in jobs])


# --------------------------------------------------------------------------- #
# Allocations (ysu grab)
# --------------------------------------------------------------------------- #
def _grab_command(option_partition: str, gpu_type: str) -> Text:
    cmd = Text("ysu grab", style=theme.YALE_BLUE_BRIGHT)
    cmd.append(f" -g {gpu_type}", style="cyan")
    cmd.append(f" -p {option_partition}", style="cyan")
    return cmd


def alloc_options_table(options: GpuOptions) -> Table:
    """Every allocatable (partition, GPU type) pairing + how to grab it."""
    table = Table(
        title="Allocatable GPUs",
        title_style="bold",
        header_style="bold white",
        border_style="grey37",
        expand=True,
        row_styles=["", "on grey11"],
    )
    table.add_column("Partition", style="bold", no_wrap=True)
    table.add_column("GPU type", no_wrap=True)
    table.add_column("Availability", justify="left")
    table.add_column("Free", justify="right", no_wrap=True)
    table.add_column("Total", justify="right", no_wrap=True)
    table.add_column("Grab it with", no_wrap=True)

    items = sorted(options.items, key=lambda it: (-it.free, it.partition, it.gpu_type))
    if not items:
        table.add_row(Text("No GPUs found on this cluster.", style="dim"), "", "", "", "", "")
        return table

    for it in items:
        bar = Text()
        bar.append(theme.usage_bar(it.total - it.free, it.total, width=16))
        free_text = Text(str(it.free), style=theme.availability_style(it.free, it.total))
        table.add_row(
            it.partition,
            Text(it.gpu_type, style=theme.gpu_color(it.gpu_type)),
            bar,
            free_text,
            str(it.total),
            _grab_command(it.partition, it.gpu_type),
        )
    return table


def free_gpu_menu(items: list[GpuOption], title: str = "Free GPUs you can grab") -> Table:
    """A numbered menu of free (partition, GPU type) options to pick from."""
    table = Table(
        title=title,
        title_style="bold",
        header_style="bold white",
        border_style="grey37",
        expand=True,
        row_styles=["", "on grey11"],
    )
    table.add_column("#", justify="right", style="bold", no_wrap=True)
    table.add_column("Partition", style="bold", no_wrap=True)
    table.add_column("GPU type", no_wrap=True)
    table.add_column("Free", justify="right", no_wrap=True)
    table.add_column("Availability", justify="left")

    for i, it in enumerate(items, start=1):
        bar = theme.usage_bar(it.total - it.free, it.total, width=16)
        table.add_row(
            str(i),
            it.partition,
            Text(it.gpu_type, style=theme.gpu_color(it.gpu_type)),
            Text(str(it.free), style=theme.availability_style(it.free, it.total)),
            bar,
        )
    return table


def alloc_suggestions(message: str, options: GpuOptions) -> RenderableType:
    """An error message paired with the real configurations to choose from."""
    error = Text()
    error.append("✗ ", style="bold red")
    error.append(message, style="red")
    body: list[RenderableType] = [error, Text()]
    if options:
        body.append(alloc_options_table(options))
    return Group(*body)


def alloc_preview(resolved: ResolvedRequest, args: list[str]) -> Panel:
    """Show the exact ``salloc`` command and a plain-English summary."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", justify="right", no_wrap=True)
    grid.add_column(overflow="fold")

    gpu_label = Text()
    if resolved.gpu_type:
        gpu_label.append(f"{resolved.count}× ", style="bold")
        gpu_label.append(resolved.gpu_type, style=theme.gpu_color(resolved.gpu_type))
    else:
        gpu_label.append(f"{resolved.count}× ", style="bold")
        gpu_label.append("any GPU", style="bold green")
    grid.add_row("GPUs", gpu_label)

    part = Text(resolved.partition_arg, style="cyan")
    if resolved.interactive_only:
        part.append("  (interactive/devel — use -a for all)", style="dim italic")
    elif resolved.any_partition and len(resolved.partitions) > 1:
        part.append("  (SLURM picks the first free)", style="dim italic")
    grid.add_row("Partition", part)
    grid.add_row("CPUs", f"{resolved.cpus} per task")
    grid.add_row("Memory", "all on node" if resolved.mem == "0" else resolved.mem)
    grid.add_row("Wall time", resolved.time)
    if resolved.account:
        grid.add_row("Account", resolved.account)

    command = Text("salloc " + " ".join(args), style="bold white")
    body = Group(grid, Text(), Text("Command:", style="dim"), command)
    return Panel(
        body,
        title=Text("Interactive GPU allocation", style="bold"),
        title_align="left",
        border_style=theme.YALE_BLUE_BRIGHT,
    )


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
def events_table(events: list[JobEvent], limit: int = 12) -> Table:
    table = Table(
        title="Job event log",
        title_style="bold",
        header_style="bold white",
        border_style="grey37",
        expand=True,
    )
    table.add_column("Time", no_wrap=True, style="dim")
    table.add_column("Event", no_wrap=True)
    table.add_column("Job", no_wrap=True)
    table.add_column("Name", overflow="ellipsis", max_width=24)
    table.add_column("User", no_wrap=True)
    table.add_column("Partition", no_wrap=True)
    table.add_column("GPUs", no_wrap=True)

    recent = events[-limit:]
    if not recent:
        table.add_row(Text("No events recorded yet.", style="dim"), "", "", "", "", "", "")
        return table

    for event in reversed(recent):
        if event.kind == "started":
            badge = Text("▶ started", style="bold green")
        else:
            badge = Text("■ finished", style="bold cyan")
        table.add_row(
            event.when.strftime("%m-%d %H:%M:%S"),
            badge,
            event.job_id,
            event.name,
            event.user,
            event.partition,
            event.gpu_label,
        )
    return table
