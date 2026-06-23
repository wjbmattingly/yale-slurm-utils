"""Pure parsing helpers for SLURM command output.

These functions are intentionally free of any I/O so they can be unit tested
in isolation against captured ``sinfo`` / ``squeue`` strings.
"""

from __future__ import annotations

import re
from datetime import datetime

# ``(null)``, ``N/A`` and empty strings all mean "no value" across SLURM tools.
_EMPTY = {"", "(null)", "n/a", "none", "(none)"}


def is_empty(value: str | None) -> bool:
    return value is None or value.strip().lower() in _EMPTY


def clean(value: str | None) -> str | None:
    """Return ``None`` for SLURM's various "no value" sentinels."""
    if is_empty(value):
        return None
    return value.strip()


def parse_slurm_time(value: str | None) -> int | None:
    """Parse a SLURM duration (e.g. ``1-02:03:04``, ``02:03:04``, ``3:04``).

    Returns the number of seconds, or ``None`` for unknown/unlimited values.
    """
    if is_empty(value):
        return None
    value = value.strip()
    if value.lower() in {"unlimited", "infinite", "invalid"}:
        return None

    days = 0
    if "-" in value:
        day_part, _, value = value.partition("-")
        try:
            days = int(day_part)
        except ValueError:
            return None

    parts = value.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None

    if len(nums) == 3:
        hours, minutes, seconds = nums
    elif len(nums) == 2:
        hours, minutes, seconds = 0, nums[0], nums[1]
    elif len(nums) == 1:
        hours, minutes, seconds = 0, 0, nums[0]
    else:
        return None

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse a SLURM ISO-ish timestamp such as ``2026-06-23T15:06:49``."""
    if is_empty(value):
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def parse_mem_mb(value: str | None) -> int | None:
    """Parse a SLURM memory value into megabytes.

    ``sinfo`` ``Memory`` is already in MB (a bare integer); TRES values look
    like ``768G`` / ``180000M`` / ``2T``.
    """
    if is_empty(value):
        return None
    value = value.strip()
    match = re.fullmatch(r"(?i)\s*([\d.]+)\s*([kmgtp]?)b?\s*", value)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2).lower()
    factor = {"": 1, "k": 1 / 1024, "m": 1, "g": 1024, "t": 1024**2, "p": 1024**3}
    return int(number * factor[unit])


# ``gpu:rtx_pro_6000_blackwell:8(IDX:1-5,7)`` -> ("rtx_pro_6000_blackwell", 8)
# ``gpu:8`` (untyped) -> ("gpu", 8)
_GRES_RE = re.compile(
    r"gpu:(?:(?P<type>[a-zA-Z0-9_.]+):)?(?P<count>\d+)(?:\([^)]*\))?"
)


def parse_gres(value: str | None) -> dict[str, int]:
    """Parse a ``Gres``/``GresUsed`` string into ``{gpu_type: count}``.

    Only GPU GRES is tracked (the thing people actually fight over). Non-GPU
    generic resources are ignored.
    """
    result: dict[str, int] = {}
    if is_empty(value):
        return result
    for match in _GRES_RE.finditer(value):
        gpu_type = match.group("type") or "gpu"
        count = int(match.group("count"))
        result[gpu_type] = result.get(gpu_type, 0) + count
    return result


# ``cpu=1,mem=180G,node=1,billing=15,gres/gpu=1,gres/gpu:rtx_5000_ada=1``
_TYPED_GPU_RE = re.compile(r"gres/gpu:(?P<type>[a-zA-Z0-9_.]+)=(?P<count>\d+)")
_TOTAL_GPU_RE = re.compile(r"gres/gpu=(?P<count>\d+)")


def parse_tres_gpus(tres: str | None) -> tuple[int, dict[str, int]]:
    """Parse an ``AllocTRES``/``tres-alloc`` string for GPU usage.

    Returns ``(total_gpus, {gpu_type: count})``. The typed entries are
    preferred; the untyped ``gres/gpu=N`` total is used as a fallback when no
    GPU type is recorded.
    """
    typed: dict[str, int] = {}
    if not is_empty(tres):
        for match in _TYPED_GPU_RE.finditer(tres):
            typed[match.group("type")] = int(match.group("count"))

    total = sum(typed.values())
    if total == 0 and not is_empty(tres):
        match = _TOTAL_GPU_RE.search(tres)
        if match:
            total = int(match.group("count"))
            if total:
                typed = {"gpu": total}
    return total, typed


def parse_cpu_state(value: str | None) -> tuple[int, int, int, int]:
    """Parse ``CPUsState`` (``A/I/O/T``) into ``(alloc, idle, other, total)``."""
    if is_empty(value):
        return (0, 0, 0, 0)
    parts = value.strip().split("/")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return (0, 0, 0, 0)
    while len(nums) < 4:
        nums.append(0)
    return (nums[0], nums[1], nums[2], nums[3])


def humanize_seconds(seconds: int | None) -> str:
    """Render a duration like ``1d 02:03`` / ``02:03:04`` / ``∞``."""
    if seconds is None:
        return "∞"
    if seconds < 0:
        seconds = 0
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}"
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def humanize_mb(mb: int | None) -> str:
    """Render a memory amount in MB as a human friendly GiB/TiB string."""
    if mb is None:
        return "-"
    if mb >= 1024 * 1024:
        return f"{mb / 1024 / 1024:.1f}T"
    if mb >= 1024:
        return f"{mb / 1024:.0f}G"
    return f"{mb}M"
