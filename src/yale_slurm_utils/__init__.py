"""Yale SLURM Utils.

A beautifully formatted, real-time toolkit for inspecting SLURM clusters
(built for Yale's Bouchet cluster, but works with any SLURM deployment).

The package exposes both a CLI (``ysu`` / ``yale-slurm``) and a small Python
framework for querying partitions, GPUs and jobs.
"""

from __future__ import annotations

__version__ = "0.1.3"

from .models import GpuClass, Job, Node, Partition  # noqa: E402
from .slurm import (  # noqa: E402
    SlurmError,
    get_jobs,
    get_nodes,
    get_partition_names,
    gpu_inventory,
)

__all__ = [
    "__version__",
    "GpuClass",
    "Job",
    "Node",
    "Partition",
    "SlurmError",
    "get_jobs",
    "get_nodes",
    "get_partition_names",
    "gpu_inventory",
]
