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
    expand_slurm_filename,
    iter_scontrol_nodes,
    parse_slurm_time,
    parse_timestamp,
    parse_tres_gpus,
    parse_tres_mem,
    state_is_down,
)

# Field is "<Name>:|" -> SLURM appends a literal "|" after each value, and we
# separate fields with ",". This avoids width truncation entirely.
_SEP = "|"

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


def cancel_job(job_id: str) -> None:
    """Cancel a job with ``scancel``. Raises :class:`SlurmError` on failure."""
    job_id = (job_id or "").strip()
    if not job_id:
        raise SlurmError("No job id to cancel.")
    _run(["scancel", job_id])


def exec_salloc(args: list[str]) -> None:  # pragma: no cover - replaces process
    """Launch an interactive ``salloc`` allocation.

    This *replaces* the current process (``execvp``) so the resulting
    interactive shell behaves exactly as if the user had typed ``salloc ...``
    themselves — job control, the terminal and Ctrl-C all work natively.
    Therefore this never returns on success.
    """
    if shutil.which("salloc") is None:
        raise SlurmError(
            "`salloc` was not found on PATH. Run this on a cluster login node "
            "with the SLURM client tools installed (e.g. a Bouchet login node)."
        )
    os.execvp("salloc", ["salloc", *args])


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
def _scontrol_nodes() -> list[dict]:
    """Authoritative per-physical-node inventory via ``scontrol``.

    ``scontrol -d -o show node`` lists each node exactly once (with its
    ``CfgTRES``/``AllocTRES`` and the set of partitions it belongs to), so we
    can tally CPUs/memory/GPUs without the double-counting that plagues
    per-partition ``sinfo`` output when partitions overlap.
    """
    output = _run(["scontrol", "-d", "-o", "show", "node"])
    return iter_scontrol_nodes(output)


def _node_from_scontrol(pn: dict, partition: str) -> Node:
    total = pn["cpus_total"]
    alloc = pn["cpus_alloc"]
    mem_total = pn["mem_total_mb"]
    mem_alloc = pn["mem_alloc_mb"] or 0
    mem_free = (mem_total - mem_alloc) if mem_total is not None else None
    return Node(
        name=pn["name"],
        partition=partition,
        state=pn["state"],
        cpus_alloc=alloc,
        cpus_idle=max(total - alloc, 0),
        cpus_other=0,
        cpus_total=total,
        mem_total_mb=mem_total,
        mem_free_mb=mem_free,
        gpus_total=dict(pn["gpus_total"]),
        gpus_used=dict(pn["gpus_used"]),
    )


def get_nodes(partition: str | None = None) -> list[Node]:
    """Return one :class:`Node` per (node, partition) pairing.

    A physical node is reported once for each partition it belongs to (so the
    per-partition views are complete), but every instance carries that node's
    real, node-level resource counts.
    """
    nodes: list[Node] = []
    for pn in _scontrol_nodes():
        for part in pn["partitions"]:
            if partition and part != partition:
                continue
            nodes.append(_node_from_scontrol(pn, part))
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


def _accumulate(gc: GpuClass, total: int, used: int, unavailable: bool) -> None:
    gc.total += total
    gc.used += used
    gc.nodes_total += 1
    if unavailable:
        gc.nodes_unavailable += 1
        gc.unavailable_gpus += max(total - used, 0)


def gpu_inventory(partition: str | None = None) -> list[GpuClass]:
    """Aggregate GPU availability per (partition, gpu_type).

    Each node is counted once *within* each partition it belongs to, so a given
    row (e.g. ``gpu_b200`` / ``b200``) reflects the true capacity of that
    partition. Note that summing rows across partitions would double-count
    physical GPUs — use :func:`gpu_pool_inventory` for cluster-wide totals.
    """
    classes: dict[tuple[str, str], GpuClass] = {}
    for node in get_nodes(partition):
        unavailable = node.is_down
        for gpu_type, total in node.gpus_total.items():
            key = (node.partition, gpu_type)
            gc = classes.get(key)
            if gc is None:
                gc = GpuClass(gpu_type=gpu_type, partition=node.partition)
                classes[key] = gc
            _accumulate(gc, total, node.gpus_used.get(gpu_type, 0), unavailable)
    return sorted(classes.values(), key=lambda g: (g.partition, g.gpu_type))


def gpu_pool_inventory(partition: str | None = None) -> list[GpuClass]:
    """Cluster-wide GPU totals per type, counting each physical node once.

    Unlike :func:`gpu_inventory`, this deduplicates nodes that live in several
    overlapping partitions, so the "GPU pool" totals reflect the real number of
    physical cards. Restrict to a ``partition`` to pool just that partition's
    nodes.
    """
    classes: dict[str, GpuClass] = {}
    for pn in _scontrol_nodes():
        if partition and partition not in pn["partitions"]:
            continue
        unavailable = state_is_down(pn["state"])
        for gpu_type, total in pn["gpus_total"].items():
            gc = classes.get(gpu_type)
            if gc is None:
                gc = GpuClass(gpu_type=gpu_type, partition=partition or "*")
                classes[gpu_type] = gc
            _accumulate(gc, total, pn["gpus_used"].get(gpu_type, 0), unavailable)
    return sorted(classes.values(), key=lambda g: g.gpu_type)


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

    job_id = job_id.strip()
    name = name.strip()
    user = user.strip()
    nodes = clean(nodelist)

    def _resolve(path: str) -> str | None:
        return expand_slurm_filename(
            path, job_id=job_id, job_name=name, user=user, nodelist=nodes
        )

    return Job(
        job_id=job_id,
        partition=part.strip(),
        name=name,
        user=user,
        state=state.strip(),
        num_nodes=_int(num_nodes, 1),
        num_cpus=_int(num_cpus, 1),
        mem_mb=parse_tres_mem(tres),
        nodelist=nodes,
        reason=clean(reason),
        submit_time=parse_timestamp(submit),
        start_time=parse_timestamp(start),
        end_time=parse_timestamp(end),
        time_limit_s=parse_slurm_time(time_limit),
        elapsed_s=parse_slurm_time(time_used),
        gpu_total=gpu_total,
        gpu_types=gpu_types,
        workdir=clean(workdir),
        stdout=_resolve(stdout),
        stderr=_resolve(stderr),
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
