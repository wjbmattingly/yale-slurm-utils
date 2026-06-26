"""Unit tests for the GPU allocation request builder (no SLURM required)."""

from __future__ import annotations

import pytest

from yale_slurm_utils.alloc import (
    AllocError,
    AllocRequest,
    build_salloc_args,
    free_options,
    gpu_options,
    normalize_mem,
    normalize_walltime,
    resolve_request,
)
from yale_slurm_utils.models import GpuClass


def _inventory() -> list[GpuClass]:
    return [
        GpuClass(gpu_type="h200", partition="gpu_h200", total=8, used=6),
        GpuClass(gpu_type="h200", partition="gpu_devel", total=4, used=0),
        GpuClass(gpu_type="b200", partition="gpu_b200", total=4, used=4),
        GpuClass(gpu_type="l40s", partition="gpu", total=2, used=1),
    ]


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def test_normalize_walltime_friendly_forms():
    assert normalize_walltime("6") == "06:00:00"
    assert normalize_walltime("2h") == "02:00:00"
    assert normalize_walltime("30m") == "00:30:00"
    assert normalize_walltime("3d") == "3-00:00:00"
    # Native SLURM forms pass through unchanged.
    assert normalize_walltime("6:00:00") == "6:00:00"
    assert normalize_walltime("1-00:00:00") == "1-00:00:00"


def test_normalize_walltime_rejects_garbage():
    with pytest.raises(AllocError):
        normalize_walltime("soon")
    with pytest.raises(AllocError):
        normalize_walltime("0")


def test_normalize_mem():
    assert normalize_mem("32") == "32G"
    assert normalize_mem("64G") == "64G"
    assert normalize_mem("500m") == "500M"
    assert normalize_mem("1t") == "1T"
    assert normalize_mem("0") == "0"
    assert normalize_mem("all") == "0"
    with pytest.raises(AllocError):
        normalize_mem("lots")


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #
def test_gpu_options_ranked_by_free():
    options = gpu_options(_inventory())
    # gpu_devel has 4 free h200, the most of anything.
    assert options.partitions[0] == "gpu_devel"
    # h200 has the most free across the cluster (4) -> first type.
    assert options.gpu_types[0] == "h200"
    assert set(options.partitions_for_type("h200")) == {"gpu_devel", "gpu_h200"}
    assert options.partitions_for_type("h200")[0] == "gpu_devel"  # most free first


# --------------------------------------------------------------------------- #
# resolution + defaults
# --------------------------------------------------------------------------- #
def test_resolve_any_gpu_defaults_to_interactive_partitions():
    options = gpu_options(_inventory())
    resolved = resolve_request(AllocRequest(), options)
    assert resolved.gpu_type is None
    assert resolved.any_partition is True
    # Default grab is restricted to the interactive (devel) partition only.
    assert resolved.interactive_only is True
    assert resolved.partitions == ["gpu_devel"]
    assert resolved.gres == "gpu:1"
    assert resolved.time == "6:00:00"
    assert resolved.cpus == 8
    assert resolved.mem == "32G"


def test_resolve_all_partitions_fans_out_everywhere():
    options = gpu_options(_inventory())
    resolved = resolve_request(AllocRequest(all_partitions=True), options)
    assert resolved.interactive_only is False
    assert resolved.partitions[0] == "gpu_devel"  # most free first
    assert set(resolved.partitions) == {"gpu_devel", "gpu_h200", "gpu_b200", "gpu"}


def test_resolve_typed_gpu_prefers_interactive_partition():
    options = gpu_options(_inventory())
    resolved = resolve_request(AllocRequest(gpu_type="h200", count=2), options)
    assert resolved.gres == "gpu:h200:2"
    # h200 exists in gpu_devel and gpu_h200; default prefers the devel one.
    assert resolved.partitions == ["gpu_devel"]
    assert resolved.interactive_only is True


def test_resolve_typed_gpu_all_partitions():
    options = gpu_options(_inventory())
    resolved = resolve_request(
        AllocRequest(gpu_type="h200", all_partitions=True), options
    )
    assert set(resolved.partitions) == {"gpu_devel", "gpu_h200"}
    assert resolved.interactive_only is False


def test_resolve_falls_back_when_type_has_no_interactive_partition():
    options = gpu_options(_inventory())
    # b200 only lives in gpu_b200 (not a devel partition) -> fall back to it.
    resolved = resolve_request(AllocRequest(gpu_type="b200"), options)
    assert resolved.partitions == ["gpu_b200"]
    assert resolved.interactive_only is False


def test_resolve_pinned_partition():
    options = gpu_options(_inventory())
    resolved = resolve_request(
        AllocRequest(partition="gpu_devel", gpu_type="h200"), options
    )
    assert resolved.partitions == ["gpu_devel"]
    assert resolved.any_partition is False


def test_unknown_gpu_type_raises_with_options():
    options = gpu_options(_inventory())
    with pytest.raises(AllocError) as exc:
        resolve_request(AllocRequest(gpu_type="h2000"), options)
    assert exc.value.options is options
    # The suggestion machinery should point at the real type.
    assert "h200" in str(exc.value)


def test_unknown_partition_raises():
    options = gpu_options(_inventory())
    with pytest.raises(AllocError):
        resolve_request(AllocRequest(partition="nope"), options)


def test_partition_without_requested_type_raises():
    options = gpu_options(_inventory())
    with pytest.raises(AllocError) as exc:
        resolve_request(
            AllocRequest(partition="gpu_b200", gpu_type="h200"), options
        )
    assert "gpu_b200" in str(exc.value)
    assert "h200" in str(exc.value)


# --------------------------------------------------------------------------- #
# free options (the --free picker)
# --------------------------------------------------------------------------- #
def test_free_options_defaults_to_interactive_with_free_gpus():
    options = gpu_options(_inventory())
    items = free_options(options)
    # Only gpu_devel (interactive) has a free GPU; gpu_b200 is full, gpu/gpu_h200
    # aren't interactive.
    assert [(it.partition, it.gpu_type) for it in items] == [("gpu_devel", "h200")]


def test_free_options_all_partitions_includes_non_devel():
    options = gpu_options(_inventory())
    items = free_options(options, all_partitions=True)
    parts = {it.partition for it in items}
    # gpu_devel (4 free), gpu_h200 (2 free), gpu (1 free); gpu_b200 is full.
    assert parts == {"gpu_devel", "gpu_h200", "gpu"}
    # Ranked most-free first.
    assert items[0].partition == "gpu_devel"


def test_free_options_excludes_full_partitions():
    options = gpu_options(_inventory())
    items = free_options(options, partition="gpu_b200")
    assert items == []  # b200 is fully allocated


def test_free_options_filtered_by_type():
    options = gpu_options(_inventory())
    items = free_options(options, gpu_type="h200", all_partitions=True)
    assert {it.partition for it in items} == {"gpu_devel", "gpu_h200"}


# --------------------------------------------------------------------------- #
# command construction
# --------------------------------------------------------------------------- #
def test_build_salloc_args():
    options = gpu_options(_inventory())
    resolved = resolve_request(
        AllocRequest(
            partition="gpu_devel", gpu_type="h200", count=2,
            time="2h", cpus=16, mem="64G", account="pi_rs2668",
        ),
        options,
    )
    args = build_salloc_args(resolved)
    assert args == [
        "--partition=gpu_devel",
        "--gres=gpu:h200:2",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=16",
        "--mem=64G",
        "--time=02:00:00",
        "--account=pi_rs2668",
    ]


def test_no_gpus_raises():
    with pytest.raises(AllocError):
        resolve_request(AllocRequest(), gpu_options([]))
