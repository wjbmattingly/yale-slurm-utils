"""Command-line interface for Yale SLURM Utils (``ysu`` / ``yale-slurm``)."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console, Group
from rich.text import Text

from . import __version__, alloc, render
from .alloc import AllocError, AllocRequest
from .events import clear_events, load_events
from .monitor import run_dashboard
from .slurm import (
    SlurmError,
    current_user,
    exec_salloc,
    get_jobs,
    get_partition_names,
    get_partitions,
    gpu_inventory,
    gpu_usage_by_user,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    rich_markup_mode="rich",
    help="[bold]Yale SLURM Utils[/] — beautifully formatted, real-time SLURM stats.",
)

console = Console()

PartitionOpt = typer.Option(
    None, "--partition", "-p", help="Restrict to a partition, e.g. gpu_h200.",
    metavar="NAME",
)


def _guard(fn):
    """Run a render function, turning SlurmError into a clean exit."""
    try:
        fn()
    except SlurmError as exc:
        console.print(
            Text("✗ ", style="bold red") + Text(str(exc), style="red")
        )
        raise typer.Exit(code=1)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"yale-slurm-utils {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Show a quick overview when invoked without a subcommand."""
    if ctx.invoked_subcommand is not None:
        return

    def _run() -> None:
        partitions = get_partitions()
        gpu_classes = gpu_inventory()
        jobs = get_jobs()
        console.print(render.header("overview"))
        console.print(render.gpu_summary(gpu_classes))
        console.print(render.partitions_table(partitions))
        console.print(render.jobs_table(jobs))
        console.print(
            Text(
                "Tip: `ysu grab` to grab a GPU · `ysu watch` for a live "
                "dashboard · `ysu gpus --free` for open GPUs · `ysu jobs` for "
                "your job details.",
                style="dim italic",
            )
        )

    _guard(_run)


@app.command("partitions")
def partitions_cmd(
    partition: Optional[str] = PartitionOpt,
    gpu: bool = typer.Option(False, "--gpu", "-g", help="Only GPU partitions."),
) -> None:
    """Show the breakdown of partitions: nodes, CPU, memory and GPUs."""

    def _run() -> None:
        parts = get_partitions(partition)
        console.print(render.partitions_table(parts, gpu_only=gpu))

    _guard(_run)


@app.command("list-partitions")
def list_partitions_cmd(
    plain: bool = typer.Option(
        False, "--plain", help="Print one name per line (script friendly)."
    ),
) -> None:
    """List available partition names."""

    def _run() -> None:
        names = get_partition_names()
        if plain:
            for name in names:
                print(name)
            return
        text = Text()
        for i, name in enumerate(names):
            if i:
                text.append("  ")
            text.append(name, style="cyan")
        console.print(render.header("partitions"))
        console.print(text)

    _guard(_run)


@app.command("gpus")
def gpus_cmd(
    partition: Optional[str] = PartitionOpt,
    free: bool = typer.Option(False, "--free", "-f", help="Only show free GPUs."),
    users: bool = typer.Option(
        False, "--users", "-u", help="Show which users are using which GPUs."
    ),
) -> None:
    """Show GPU availability per partition and type."""

    def _run() -> None:
        gpu_classes = gpu_inventory(partition)
        renderables = [render.gpu_summary(gpu_classes), render.gpu_table(gpu_classes, free_only=free)]
        if users:
            usage = gpu_usage_by_user(partition)
            renderables.append(render.gpu_users_table(usage, me=current_user()))
        console.print(Group(*renderables))

    _guard(_run)


@app.command("free")
def free_cmd(partition: Optional[str] = PartitionOpt) -> None:
    """Shortcut: show only the free GPUs."""

    def _run() -> None:
        gpu_classes = gpu_inventory(partition)
        console.print(render.gpu_summary(gpu_classes))
        console.print(render.gpu_table(gpu_classes, free_only=True))

    _guard(_run)


@app.command("grab")
def grab_cmd(
    gpu: Optional[str] = typer.Option(
        None, "--gpu", "-g",
        help="GPU model, e.g. h200. Default: any available GPU.",
        metavar="TYPE",
    ),
    num: int = typer.Option(
        alloc.DEFAULT_COUNT, "--num", "-n", min=1, help="How many GPUs to request."
    ),
    partition: Optional[str] = typer.Option(
        None, "--partition", "-p",
        help="Pin to one partition. Default: any that has the GPU.",
        metavar="NAME",
    ),
    time: str = typer.Option(
        alloc.DEFAULT_TIME, "--time", "-t",
        help="Wall time: 6, 2h, 30m, 6:00:00 or 1-00:00:00.",
    ),
    cpus: int = typer.Option(
        alloc.DEFAULT_CPUS, "--cpus", "-c", min=1, help="CPUs per task."
    ),
    mem: str = typer.Option(
        alloc.DEFAULT_MEM, "--mem", "-m",
        help="Memory: 32G, 64G, 500M, 1T, or 0 for all on the node.",
    ),
    account: Optional[str] = typer.Option(
        None, "--account", "-A", help="Charge the allocation to this account."
    ),
    all_partitions: bool = typer.Option(
        False, "--all-partitions", "-a",
        help="Search every partition, not just interactive (devel) ones.",
    ),
    free: bool = typer.Option(
        False, "--free", "-f",
        help="List the free GPUs and let you pick one to grab.",
    ),
    list_options: bool = typer.Option(
        False, "--list", "-l",
        help="List every allocatable GPU configuration and exit.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the salloc command without running it."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
) -> None:
    """Grab an interactive GPU with [bold]salloc[/] — validated, with suggestions.

    Examples:

    [cyan]ysu grab[/]                 grab any free GPU for 6h, 8 CPUs, 32G

    [cyan]ysu grab -g h200 -n 2[/]    two H200s

    [cyan]ysu grab -p gpu_devel -t 2h -m 64G[/]

    [cyan]ysu grab --free[/]          list the free GPUs and pick one

    [cyan]ysu grab --list[/]          see everything you could ask for
    """

    def _resolve(req: AllocRequest, options) -> "alloc.ResolvedRequest | None":
        try:
            return alloc.resolve_request(req, options)
        except AllocError as exc:
            console.print(render.alloc_suggestions(str(exc), exc.options or options))
            raise typer.Exit(code=1)

    def _launch(resolved: "alloc.ResolvedRequest") -> None:
        args = alloc.build_salloc_args(resolved)
        console.print(render.alloc_preview(resolved, args))
        if dry_run:
            console.print(Text("Dry run — nothing launched.", style="dim italic"))
            return
        if not yes and not typer.confirm("Launch this allocation?", default=True):
            console.print("[dim]Cancelled.[/]")
            return
        console.print(Text("Requesting allocation… (Ctrl-C to give up)", style="dim"))
        exec_salloc(args)

    def _pick_free(options) -> "alloc.GpuOption | None":
        # Validate any pre-filters so a typo still gets a helpful suggestion.
        if gpu and gpu not in options.gpu_types:
            raise AllocError(
                f"Unknown GPU type {gpu!r}. "
                f"Available types: {', '.join(options.gpu_types)}.",
                options,
            )
        if partition and partition not in options.partitions:
            raise AllocError(
                f"Unknown GPU partition {partition!r}. "
                f"GPU partitions: {', '.join(options.partitions)}.",
                options,
            )
        items = alloc.free_options(
            options, gpu_type=gpu, partition=partition, all_partitions=all_partitions,
        )
        if partition:
            scope = f"partition {partition}"
        elif all_partitions:
            scope = "any partition"
        else:
            scope = "the interactive (devel) partitions"
        if not items:
            console.print(
                Text(f"No free GPUs on {scope} right now.", style="yellow")
            )
            console.print(
                Text(
                    "Try `ysu grab --list` to see everything, or `-a` to include "
                    "all partitions.",
                    style="dim italic",
                )
            )
            return None

        console.print(render.header("grab"))
        console.print(render.free_gpu_menu(items, title=f"Free GPUs on {scope}"))

        if yes:
            return items[0]  # non-interactive: take the most-free option
        while True:
            raw = typer.prompt(
                f"Which GPU? [1-{len(items)}, q to cancel]", default="1"
            ).strip().lower()
            if raw in {"q", "quit", "cancel"}:
                console.print("[dim]Cancelled.[/]")
                return None
            try:
                idx = int(raw)
            except ValueError:
                console.print("[red]Please enter a number.[/]")
                continue
            if 1 <= idx <= len(items):
                return items[idx - 1]
            console.print(f"[red]Pick a number between 1 and {len(items)}.[/]")

    def _run() -> None:
        options = alloc.gpu_options(gpu_inventory())

        if list_options:
            console.print(render.header("grab"))
            console.print(render.alloc_options_table(options))
            console.print(
                Text(
                    "Tip: `ysu grab` alone grabs any free GPU · `ysu grab --free` "
                    "to pick from what's free · add `-g <type>`, `-n <count>`, "
                    "`-t <time>`, `-m <mem>` to narrow it down.",
                    style="dim italic",
                )
            )
            return

        if free:
            try:
                choice = _pick_free(options)
            except AllocError as exc:
                console.print(render.alloc_suggestions(str(exc), exc.options or options))
                raise typer.Exit(code=1)
            if choice is None:
                return
            request = AllocRequest(
                partition=choice.partition,
                gpu_type=choice.gpu_type,
                count=num,
                time=time,
                cpus=cpus,
                mem=mem,
                account=account,
            )
            _launch(_resolve(request, options))
            return

        request = AllocRequest(
            partition=partition,
            gpu_type=gpu,
            count=num,
            time=time,
            cpus=cpus,
            mem=mem,
            account=account,
            all_partitions=all_partitions,
        )
        _launch(_resolve(request, options))

    _guard(_run)


@app.command("users")
def users_cmd(partition: Optional[str] = PartitionOpt) -> None:
    """Show which users are using which GPUs."""

    def _run() -> None:
        usage = gpu_usage_by_user(partition)
        console.print(render.gpu_users_table(usage, me=current_user()))

    _guard(_run)


@app.command("jobs")
def jobs_cmd(
    user: Optional[str] = typer.Option(
        None, "--user", "-u", help="Show another user's jobs (default: you)."
    ),
    all_users: bool = typer.Option(
        False, "--all", "-a", help="Show jobs from all users."
    ),
    partition: Optional[str] = PartitionOpt,
    table: bool = typer.Option(
        False, "--table", "-t", help="Compact table instead of detailed panels."
    ),
) -> None:
    """Show your jobs: GPUs, source dirs, log files and time remaining."""
    target = "*" if all_users else user

    def _run() -> None:
        jobs = get_jobs(user=target, partition=partition)
        console.print(render.header("jobs"))
        if table or all_users:
            console.print(render.jobs_table(jobs))
        else:
            console.print(render.jobs_detail(jobs))

    _guard(_run)


@app.command("watch")
def watch_cmd(
    interval: int = typer.Option(
        10, "--interval", "-i", min=2, help="Refresh interval in seconds."
    ),
    user: Optional[str] = typer.Option(
        None, "--user", "-u", help="Monitor another user (default: you)."
    ),
    all_users: bool = typer.Option(
        False, "--all", "-a", help="Monitor all users' jobs for start/finish."
    ),
    partition: Optional[str] = PartitionOpt,
    no_bell: bool = typer.Option(
        False, "--no-bell", help="Disable the terminal bell on job events."
    ),
) -> None:
    """Live dashboard with a [bold]bell[/] on job start/finish + an event log."""
    target = "*" if all_users else user
    try:
        run_dashboard(
            user=target,
            partition=partition,
            interval=interval,
            bell=not no_bell,
            console=console,
        )
    except KeyboardInterrupt:
        console.print("[dim]Stopped watching.[/]")
    except SlurmError as exc:
        console.print(Text("✗ ", style="bold red") + Text(str(exc), style="red"))
        raise typer.Exit(code=1)


@app.command("log")
def log_cmd(
    limit: int = typer.Option(20, "--limit", "-n", help="How many events to show."),
    clear: bool = typer.Option(False, "--clear", help="Erase the event log."),
) -> None:
    """Show the persistent log of job start/finish events."""
    if clear:
        clear_events()
        console.print("[green]Event log cleared.[/]")
        return
    console.print(render.events_table(load_events(limit=limit), limit=limit))


# `ysu alloc` is an alias for `ysu grab`.
app.command("alloc", hidden=True)(grab_cmd)


if __name__ == "__main__":  # pragma: no cover
    app()
