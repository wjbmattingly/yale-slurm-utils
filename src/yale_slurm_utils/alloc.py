"""Build and validate interactive GPU allocations (``salloc``).

This module is deliberately free of any SLURM I/O so the request-building and
validation logic can be unit tested against synthetic GPU inventories. The CLI
fetches the live inventory (:func:`~yale_slurm_utils.slurm.gpu_inventory`) and
hands the resulting :class:`~yale_slurm_utils.models.GpuClass` list to
:func:`gpu_options`.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .models import GpuClass
from .slurm import SlurmError
from .parsing import parse_slurm_time

# Sensible defaults for "just give me a GPU to work on for a while".
DEFAULT_TIME = "6:00:00"
DEFAULT_CPUS = 8
DEFAULT_MEM = "32G"
DEFAULT_COUNT = 1

# Interactive `salloc` jobs are usually only permitted on the "devel"
# partitions (e.g. ``gpu_devel``). Batch-only partitions like ``scavenge`` or
# ``priority_gpu`` reject interactive grabs with a QOS policy error, so when the
# user doesn't pin a partition we default to the interactive ones.
INTERACTIVE_PARTITION_HINTS = ("devel",)


def is_interactive_partition(name: str) -> bool:
    low = name.lower()
    return any(hint in low for hint in INTERACTIVE_PARTITION_HINTS)


class AllocError(SlurmError):
    """A request that doesn't match any real cluster configuration.

    Carries the live :class:`GpuOptions` so the CLI can show the user the set
    of configurations they *could* have asked for.
    """

    def __init__(self, message: str, options: "GpuOptions | None" = None) -> None:
        super().__init__(message)
        self.options = options


# --------------------------------------------------------------------------- #
# Live cluster options
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GpuOption:
    """One allocatable (partition, GPU model) pairing with availability."""

    partition: str
    gpu_type: str
    free: int
    total: int


class GpuOptions:
    """The set of GPU configurations a user could actually request."""

    def __init__(self, items: list[GpuOption]) -> None:
        self.items = items

    def __bool__(self) -> bool:
        return bool(self.items)

    @staticmethod
    def _ranked(agg: dict[str, tuple[int, int]]) -> list[str]:
        # Most free first, then alphabetical — so suggestions lead with what
        # the user can actually grab right now.
        return [k for k, _ in sorted(agg.items(), key=lambda kv: (-kv[1][0], kv[0]))]

    @property
    def gpu_types(self) -> list[str]:
        agg: dict[str, tuple[int, int]] = {}
        for it in self.items:
            free, total = agg.get(it.gpu_type, (0, 0))
            agg[it.gpu_type] = (free + it.free, total + it.total)
        return self._ranked(agg)

    @property
    def partitions(self) -> list[str]:
        agg: dict[str, tuple[int, int]] = {}
        for it in self.items:
            free, total = agg.get(it.partition, (0, 0))
            agg[it.partition] = (free + it.free, total + it.total)
        return self._ranked(agg)

    def partitions_for_type(self, gpu_type: str) -> list[str]:
        rows = [it for it in self.items if it.gpu_type == gpu_type]
        rows.sort(key=lambda it: (-it.free, it.partition))
        return [it.partition for it in rows]

    def types_for_partition(self, partition: str) -> list[str]:
        rows = [it for it in self.items if it.partition == partition]
        rows.sort(key=lambda it: (-it.free, it.gpu_type))
        return [it.gpu_type for it in rows]


def gpu_options(gpu_classes: list[GpuClass]) -> GpuOptions:
    """Build :class:`GpuOptions` from a GPU inventory."""
    items = [
        GpuOption(gc.partition, gc.gpu_type, gc.free, gc.total)
        for gc in gpu_classes
        if gc.total > 0
    ]
    return GpuOptions(items)


def free_options(
    options: GpuOptions,
    *,
    gpu_type: str | None = None,
    partition: str | None = None,
    all_partitions: bool = False,
) -> list[GpuOption]:
    """The (partition, GPU type) pairings that have a free GPU *right now*.

    Sorted most-free first. By default the result is restricted to the
    interactive (devel) partitions — the ones an interactive ``salloc`` is
    actually allowed on — unless a specific ``partition`` is given or
    ``all_partitions`` is set.
    """
    items = [it for it in options.items if it.free > 0]
    if gpu_type:
        items = [it for it in items if it.gpu_type == gpu_type]
    if partition:
        items = [it for it in items if it.partition == partition]
    elif not all_partitions:
        items = [it for it in items if is_interactive_partition(it.partition)]
    items.sort(key=lambda it: (-it.free, it.partition, it.gpu_type))
    return items


# --------------------------------------------------------------------------- #
# Input normalisation
# --------------------------------------------------------------------------- #
def _seconds_to_slurm(total: int) -> str:
    days, rem = divmod(max(total, 0), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def normalize_walltime(value: str) -> str:
    """Accept friendly durations and return a SLURM ``--time`` string.

    Understands ``6`` (hours), ``2h``, ``30m``, ``3d``, as well as native
    SLURM forms like ``6:00:00`` and ``1-00:00:00``.
    """
    raw = (value or "").strip()
    if not raw:
        raise AllocError("Please provide a --time, e.g. 6, 2h, 30m or 6:00:00.")
    low = raw.lower()

    match = re.fullmatch(r"(\d+)\s*([dhm])", low)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        factor = {"d": 86400, "h": 3600, "m": 60}[unit]
        if amount <= 0:
            raise AllocError(f"--time {value!r} must be greater than zero.")
        return _seconds_to_slurm(amount * factor)

    if re.fullmatch(r"\d+", low):  # a bare number means hours
        hours = int(low)
        if hours <= 0:
            raise AllocError("--time must be greater than zero.")
        return _seconds_to_slurm(hours * 3600)

    if parse_slurm_time(raw) is None:
        raise AllocError(
            f"Couldn't understand --time {value!r}. "
            "Try 6, 2h, 30m, 6:00:00 or 1-00:00:00."
        )
    return raw


def normalize_mem(value: str) -> str:
    """Accept friendly memory sizes and return a SLURM ``--mem`` string.

    Understands ``32`` / ``32G`` / ``500M`` / ``1T``, and ``0`` (or ``all``)
    meaning "all memory on the node".
    """
    raw = (value or "").strip()
    low = raw.lower()
    if low in {"0", "all"}:
        return "0"
    if raw.isdigit():  # bare number defaults to gigabytes
        return f"{raw}G"
    match = re.fullmatch(r"(?i)(\d+)\s*([mgt])b?", raw)
    if not match:
        raise AllocError(
            f"Couldn't understand --mem {value!r}. "
            "Try 32G, 64G, 500M, 1T, or 0 for all memory on the node."
        )
    return f"{match.group(1)}{match.group(2).upper()}"


# --------------------------------------------------------------------------- #
# Request -> validated request -> salloc args
# --------------------------------------------------------------------------- #
@dataclass
class AllocRequest:
    """What the user asked for (pre-validation, pre-defaults)."""

    partition: str | None = None
    gpu_type: str | None = None
    count: int = DEFAULT_COUNT
    time: str = DEFAULT_TIME
    cpus: int = DEFAULT_CPUS
    mem: str = DEFAULT_MEM
    account: str | None = None
    # Search every matching partition, not just the interactive (devel) ones.
    all_partitions: bool = False


@dataclass
class ResolvedRequest:
    """A fully validated request, ready to turn into ``salloc`` flags."""

    partitions: list[str]
    gpu_type: str | None
    count: int
    time: str
    cpus: int
    mem: str
    account: str | None = None
    # True when the user didn't pin a partition and we fanned out across the
    # partitions that can satisfy the request ("grab any GPU").
    any_partition: bool = False
    # True when that fan-out was restricted to the interactive (devel)
    # partitions (the default for an unpinned grab).
    interactive_only: bool = False

    @property
    def partition_arg(self) -> str:
        return ",".join(self.partitions)

    @property
    def gres(self) -> str:
        if self.gpu_type:
            return f"gpu:{self.gpu_type}:{self.count}"
        return f"gpu:{self.count}"


def _suggest(value: str, choices: list[str]) -> str:
    close = difflib.get_close_matches(value, choices, n=1, cutoff=0.4)
    return f" Did you mean {close[0]!r}?" if close else ""


def resolve_request(req: AllocRequest, options: GpuOptions) -> ResolvedRequest:
    """Validate ``req`` against the live ``options`` and fill in defaults.

    Raises :class:`AllocError` (carrying ``options``) when the partition or GPU
    type doesn't exist, or when the two are mutually incompatible.
    """
    if not options:
        raise AllocError("No GPUs are visible on this cluster.", options)

    if req.count < 1:
        raise AllocError("--num must be at least 1.", options)

    time = normalize_walltime(req.time)
    mem = normalize_mem(req.mem)
    if req.cpus < 1:
        raise AllocError("--cpus must be at least 1.", options)

    gpu_type = req.gpu_type
    if gpu_type and gpu_type not in options.gpu_types:
        raise AllocError(
            f"Unknown GPU type {gpu_type!r}.{_suggest(gpu_type, options.gpu_types)} "
            f"Available types: {', '.join(options.gpu_types)}.",
            options,
        )

    partition = req.partition
    if partition:
        if partition not in options.partitions:
            raise AllocError(
                f"Unknown GPU partition {partition!r}."
                f"{_suggest(partition, options.partitions)} "
                f"GPU partitions: {', '.join(options.partitions)}.",
                options,
            )
        if gpu_type and gpu_type not in options.types_for_partition(partition):
            offered = ", ".join(options.types_for_partition(partition))
            where = ", ".join(options.partitions_for_type(gpu_type))
            raise AllocError(
                f"Partition {partition!r} has no {gpu_type!r} GPUs. "
                f"It offers: {offered}. "
                f"{gpu_type!r} lives in: {where}.",
                options,
            )
        partitions = [partition]
        any_partition = False
        interactive_only = False
    else:
        # No partition pinned: fan out across the partitions that can satisfy
        # the request so SLURM grabs whatever is free first. By default we
        # restrict this to the interactive (devel) partitions, since batch-only
        # partitions reject interactive grabs with a QOS error.
        if gpu_type:
            candidates = options.partitions_for_type(gpu_type)
        else:
            candidates = options.partitions
        interactive_only = False
        if not req.all_partitions:
            devel = [p for p in candidates if is_interactive_partition(p)]
            if devel:
                candidates = devel
                interactive_only = True
        partitions = candidates
        any_partition = True

    return ResolvedRequest(
        partitions=partitions,
        gpu_type=gpu_type,
        count=req.count,
        time=time,
        cpus=req.cpus,
        mem=mem,
        account=req.account,
        any_partition=any_partition,
        interactive_only=interactive_only,
    )


def build_salloc_args(resolved: ResolvedRequest) -> list[str]:
    """Turn a :class:`ResolvedRequest` into ``salloc`` command-line flags."""
    args = [
        f"--partition={resolved.partition_arg}",
        f"--gres={resolved.gres}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={resolved.cpus}",
        f"--mem={resolved.mem}",
        f"--time={resolved.time}",
    ]
    if resolved.account:
        args.append(f"--account={resolved.account}")
    return args
