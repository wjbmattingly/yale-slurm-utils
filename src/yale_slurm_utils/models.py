"""Dataclasses describing the SLURM world: nodes, partitions and jobs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Node:
    """A single compute node as seen from one partition.

    ``sinfo -N`` reports a node once per partition it belongs to, so the same
    physical host can appear in several :class:`Node` instances.
    """

    name: str
    partition: str
    state: str
    cpus_alloc: int = 0
    cpus_idle: int = 0
    cpus_other: int = 0
    cpus_total: int = 0
    mem_total_mb: int | None = None
    mem_free_mb: int | None = None
    gpus_total: dict[str, int] = field(default_factory=dict)
    gpus_used: dict[str, int] = field(default_factory=dict)

    @property
    def gpus_free(self) -> dict[str, int]:
        free: dict[str, int] = {}
        for gpu_type, total in self.gpus_total.items():
            free[gpu_type] = max(total - self.gpus_used.get(gpu_type, 0), 0)
        return free

    @property
    def total_gpus(self) -> int:
        return sum(self.gpus_total.values())

    @property
    def used_gpus(self) -> int:
        return sum(self.gpus_used.values())

    @property
    def free_gpus(self) -> int:
        return sum(self.gpus_free.values())

    @property
    def is_down(self) -> bool:
        s = self.state.lower().rstrip("*~#$@+-")
        return any(
            token in s
            for token in ("down", "drain", "fail", "maint", "unk", "reserved")
        )


@dataclass
class GpuClass:
    """Aggregated availability for a single GPU model within a partition."""

    gpu_type: str
    partition: str
    total: int = 0
    used: int = 0
    nodes_total: int = 0
    nodes_unavailable: int = 0

    @property
    def free(self) -> int:
        return max(self.total - self.used, 0)

    @property
    def util_pct(self) -> float:
        return (self.used / self.total * 100.0) if self.total else 0.0


@dataclass
class Partition:
    """Aggregate view of a partition built from its nodes."""

    name: str
    nodes: list[Node] = field(default_factory=list)

    @property
    def is_default(self) -> bool:
        return self.name.endswith("*")

    @property
    def clean_name(self) -> str:
        return self.name.rstrip("*")

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def cpus_total(self) -> int:
        return sum(n.cpus_total for n in self.nodes)

    @property
    def cpus_alloc(self) -> int:
        return sum(n.cpus_alloc for n in self.nodes)

    @property
    def cpus_idle(self) -> int:
        return sum(n.cpus_idle for n in self.nodes)

    @property
    def cpu_util_pct(self) -> float:
        return (self.cpus_alloc / self.cpus_total * 100.0) if self.cpus_total else 0.0

    @property
    def mem_total_mb(self) -> int:
        return sum(n.mem_total_mb or 0 for n in self.nodes)

    @property
    def gpus_total(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for n in self.nodes:
            for gpu_type, count in n.gpus_total.items():
                out[gpu_type] += count
        return dict(out)

    @property
    def gpus_used(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for n in self.nodes:
            for gpu_type, count in n.gpus_used.items():
                out[gpu_type] += count
        return dict(out)

    @property
    def total_gpus(self) -> int:
        return sum(self.gpus_total.values())

    @property
    def used_gpus(self) -> int:
        return sum(self.gpus_used.values())

    @property
    def free_gpus(self) -> int:
        return max(self.total_gpus - self.used_gpus, 0)

    @property
    def has_gpus(self) -> bool:
        return self.total_gpus > 0

    def state_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for n in self.nodes:
            counts[_simple_state(n.state)] += 1
        return dict(counts)


def _simple_state(state: str) -> str:
    s = state.lower().rstrip("*~#$@+-")
    if "idle" in s:
        return "idle"
    if "mix" in s:
        return "mixed"
    if "alloc" in s:
        return "allocated"
    if any(t in s for t in ("down", "fail")):
        return "down"
    if "drain" in s:
        return "drain"
    if "maint" in s:
        return "maint"
    if "resv" in s or "reserved" in s:
        return "reserved"
    return s or "unknown"


@dataclass
class Job:
    """A SLURM job (running or pending)."""

    job_id: str
    partition: str
    name: str
    user: str
    state: str
    num_nodes: int = 1
    num_cpus: int = 1
    nodelist: str | None = None
    reason: str | None = None
    submit_time: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    time_limit_s: int | None = None
    elapsed_s: int | None = None
    gpu_total: int = 0
    gpu_types: dict[str, int] = field(default_factory=dict)
    workdir: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    command: str | None = None

    @property
    def is_running(self) -> bool:
        return self.state.upper() == "RUNNING"

    @property
    def is_pending(self) -> bool:
        return self.state.upper() == "PENDING"

    @property
    def gpu_label(self) -> str:
        if not self.gpu_total:
            return "-"
        if self.gpu_types and set(self.gpu_types) != {"gpu"}:
            return ", ".join(
                f"{count}x {gpu_type}" for gpu_type, count in self.gpu_types.items()
            )
        return f"{self.gpu_total}x gpu"

    def time_left_s(self, now: datetime | None = None) -> int | None:
        """Seconds remaining before the wall-clock limit is hit."""
        now = now or datetime.now()
        if self.end_time is not None and self.is_running:
            return max(int((self.end_time - now).total_seconds()), 0)
        if self.elapsed_s is not None and self.time_limit_s is not None:
            return max(self.time_limit_s - self.elapsed_s, 0)
        return None

    def percent_elapsed(self, now: datetime | None = None) -> float | None:
        """Fraction (0-100) of the wall-clock limit consumed."""
        if self.time_limit_s in (None, 0):
            return None
        if self.elapsed_s is not None:
            return min(self.elapsed_s / self.time_limit_s * 100.0, 100.0)
        if self.start_time is not None and self.is_running:
            now = now or datetime.now()
            elapsed = (now - self.start_time).total_seconds()
            return min(max(elapsed, 0) / self.time_limit_s * 100.0, 100.0)
        return None
