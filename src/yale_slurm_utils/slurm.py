"""Thin, well-typed wrappers around the SLURM client commands.

Everything funnels through :func:`_run`, which shells out to ``sinfo`` /
``squeue`` with a ``|``-delimited output format that is robust to the wide,
free-form values SLURM likes to emit (node lists, working directories, ...).
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
from collections import defaultdict

from .models import GpuClass, Job, Node, Partition
from .parsing import (
    clean,
    parse_cpu_state,
    parse_gres,
    parse_mem_mb,
    parse_slurm_time,
    parse_timestamp,
    parse_tres_gpus,
)

# Field is "<Name>:|" -> SLURM appends a literal "|" after each value, and we
# separate fields with ",". This avoids width truncation entirely.
_SEP = "|"

_SINFO_FIELDS = [
    "Partition",
    "NodeHost",
    "StateLong",
    "CPUsState",
    "Memory",
    "FreeMem",
    "Gres",
    "GresUsed",
]

_SQUEUE_FIELDS = [
    "JobID",
    "Partition",
    "Name",
    "UserName",
    "State",
    "TimeUsed",
    "TimeLimit",
    "NumNodes",
    "NumCPUs",
    "NodeList",
    "Reason",
    "SubmitTime",
    "StartTime",
    "EndTime",
    "tres-alloc",
    "WorkDir",
    "STDOUT",
    "STDERR",
    "Command",
]

# Lightweight field set for cluster-wide scans (thousands of jobs).
_SQUEUE_LITE_FIELDS = [
    "JobID",
    "UserName",
    "Partition",
    "State",
    "NodeList",
    "tres-alloc",
]


class SlurmError(RuntimeError):
    """Raised when a SLURM command is missing or fails."""


def current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - extremely unusual
        return os.environ.get("USER", "")


def _format(fields: list[str]) -> str:
    return ",".join(f"{name}:{_SEP}" for name in fields)


def _run(args: list[str]) -> str:
    exe = args[0]
    if shutil.which(exe) is None:
        raise SlurmError(
            f"`{exe}` was not found on PATH. Yale SLURM Utils must run on a "
            "machine with the SLURM client tools installed (e.g. a Bouchet "
            "login node)."
        )
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - env dependent
        raise SlurmError(f"`{exe}` timed out after 60s.") from exc
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
        if "Unable to contact slurm controller" in message:
            raise SlurmError(
                "Unable to contact the SLURM controller. Are you on a cluster "
                "login node with the scheduler reachable?"
            )
        raise SlurmError(f"`{' '.join(args)}` failed: {message}")
    return proc.stdout


def _split_rows(output: str, n_fields: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        # Each value is suffixed by "|"; strip a single trailing separator.
        parts = line.split(_SEP)
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if len(parts) < n_fields:
            parts += [""] * (n_fields - len(parts))
        rows.append(parts[:n_fields])
    return rows


# --------------------------------------------------------------------------- #
# Nodes / partitions
# --------------------------------------------------------------------------- #
def get_nodes(partition: str | None = None) -> list[Node]:
    """Return one :class:`Node` per (node, partition) pairing."""
    args = ["sinfo", "-h", "-N", "-O", _format(_SINFO_FIELDS)]
    if partition:
        args += ["-p", partition]
    output = _run(args)

    nodes: list[Node] = []
    for row in _split_rows(output, len(_SINFO_FIELDS)):
        part, host, state, cpu_state, mem, free_mem, gres, gres_used = row
        alloc, idle, other, total = parse_cpu_state(cpu_state)
        nodes.append(
            Node(
                name=host.strip(),
                partition=part.strip(),
                state=state.strip(),
                cpus_alloc=alloc,
                cpus_idle=idle,
                cpus_other=other,
                cpus_total=total,
                mem_total_mb=parse_mem_mb(mem),
                mem_free_mb=parse_mem_mb(free_mem),
                gpus_total=parse_gres(gres),
                gpus_used=parse_gres(gres_used),
            )
        )
    return nodes


def get_partitions(partition: str | None = None) -> list[Partition]:
    """Group nodes into :class:`Partition` aggregates, preserving sinfo order."""
    grouped: dict[str, Partition] = {}
    order: list[str] = []
    for node in get_nodes(partition):
        if node.partition not in grouped:
            grouped[node.partition] = Partition(name=node.partition)
            order.append(node.partition)
        grouped[node.partition].nodes.append(node)
    return [grouped[name] for name in order]


def get_partition_names(include_default_marker: bool = False) -> list[str]:
    """Return the sorted list of partition names."""
    output = _run(["sinfo", "-h", "-o", "%P"])
    names = []
    for line in output.splitlines():
        name = line.strip()
        if not name:
            continue
        if not include_default_marker:
            name = name.rstrip("*")
        names.append(name)
    return sorted(set(names))


def gpu_inventory(partition: str | None = None) -> list[GpuClass]:
    """Aggregate GPU availability per (partition, gpu_type)."""
    classes: dict[tuple[str, str], GpuClass] = {}
    for node in get_nodes(partition):
        unavailable = node.is_down
        for gpu_type, total in node.gpus_total.items():
            key = (node.partition, gpu_type)
            gc = classes.get(key)
            if gc is None:
                gc = GpuClass(gpu_type=gpu_type, partition=node.partition)
                classes[key] = gc
            gc.total += total
            gc.used += node.gpus_used.get(gpu_type, 0)
            gc.nodes_total += 1
            if unavailable:
                gc.nodes_unavailable += 1
    return sorted(classes.values(), key=lambda g: (g.partition, g.gpu_type))


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def _build_job(row: list[str]) -> Job:
    (
        job_id,
        part,
        name,
        user,
        state,
        time_used,
        time_limit,
        num_nodes,
        num_cpus,
        nodelist,
        reason,
        submit,
        start,
        end,
        tres,
        workdir,
        stdout,
        stderr,
        command,
    ) = row
    gpu_total, gpu_types = parse_tres_gpus(tres)

    def _int(value: str, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    return Job(
        job_id=job_id.strip(),
        partition=part.strip(),
        name=name.strip(),
        user=user.strip(),
        state=state.strip(),
        num_nodes=_int(num_nodes, 1),
        num_cpus=_int(num_cpus, 1),
        nodelist=clean(nodelist),
        reason=clean(reason),
        submit_time=parse_timestamp(submit),
        start_time=parse_timestamp(start),
        end_time=parse_timestamp(end),
        time_limit_s=parse_slurm_time(time_limit),
        elapsed_s=parse_slurm_time(time_used),
        gpu_total=gpu_total,
        gpu_types=gpu_types,
        workdir=clean(workdir),
        stdout=clean(stdout),
        stderr=clean(stderr),
        command=clean(command),
    )


def get_jobs(
    user: str | None = None,
    partition: str | None = None,
    states: list[str] | None = None,
) -> list[Job]:
    """Return detailed jobs (with workdir / log paths / GPU allocation)."""
    args = ["squeue", "-h", "-O", _format(_SQUEUE_FIELDS)]
    if user == "*":
        pass
    elif user:
        args += ["-u", user]
    else:
        args += ["--me"]
    if partition:
        args += ["-p", partition]
    if states:
        args += ["-t", ",".join(states)]
    output = _run(args)
    return [_build_job(row) for row in _split_rows(output, len(_SQUEUE_FIELDS))]


def gpu_jobs(partition: str | None = None) -> list[Job]:
    """Cluster-wide running GPU jobs (lightweight: id/user/partition/node/gpus)."""
    args = ["squeue", "-h", "-t", "RUNNING", "-O", _format(_SQUEUE_LITE_FIELDS)]
    if partition:
        args += ["-p", partition]
    output = _run(args)

    jobs: list[Job] = []
    for row in _split_rows(output, len(_SQUEUE_LITE_FIELDS)):
        job_id, user, part, state, nodelist, tres = row
        gpu_total, gpu_types = parse_tres_gpus(tres)
        if gpu_total <= 0:
            continue
        jobs.append(
            Job(
                job_id=job_id.strip(),
                partition=part.strip(),
                name="",
                user=user.strip(),
                state=state.strip(),
                nodelist=clean(nodelist),
                gpu_total=gpu_total,
                gpu_types=gpu_types,
            )
        )
    return jobs


def gpu_usage_by_user(partition: str | None = None) -> dict[str, dict[str, int]]:
    """Map ``user -> {gpu_type: count}`` across all running GPU jobs."""
    usage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for job in gpu_jobs(partition):
        types = job.gpu_types or {"gpu": job.gpu_total}
        for gpu_type, count in types.items():
            usage[job.user][gpu_type] += count
    return {user: dict(types) for user, types in usage.items()}
