"""Unit tests for the pure parsing helpers (no SLURM required)."""

from __future__ import annotations

from datetime import datetime

from yale_slurm_utils.parsing import (
    humanize_seconds,
    parse_cpu_state,
    parse_gres,
    parse_mem_mb,
    parse_slurm_time,
    parse_timestamp,
    parse_tres_gpus,
)


def test_parse_slurm_time():
    assert parse_slurm_time("1-02:03:04") == 86400 + 2 * 3600 + 3 * 60 + 4
    assert parse_slurm_time("02:03:04") == 2 * 3600 + 3 * 60 + 4
    assert parse_slurm_time("3:04") == 3 * 60 + 4
    assert parse_slurm_time("0:00") == 0
    assert parse_slurm_time("UNLIMITED") is None
    assert parse_slurm_time("N/A") is None


def test_parse_timestamp():
    assert parse_timestamp("2026-06-23T15:06:49") == datetime(2026, 6, 23, 15, 6, 49)
    assert parse_timestamp("N/A") is None


def test_parse_mem_mb():
    assert parse_mem_mb("768G") == 768 * 1024
    assert parse_mem_mb("2T") == 2 * 1024 * 1024
    assert parse_mem_mb("180000") == 180000
    assert parse_mem_mb("(null)") is None


def test_parse_gres():
    total = parse_gres("gpu:rtx_pro_6000_blackwell:8(S:0-1)")
    assert total == {"rtx_pro_6000_blackwell": 8}
    used = parse_gres("gpu:rtx_pro_6000_blackwell:6(IDX:1-5,7)")
    assert used == {"rtx_pro_6000_blackwell": 6}
    assert parse_gres("(null)") == {}
    assert parse_gres("gpu:8") == {"gpu": 8}


def test_parse_tres_gpus():
    total, typed = parse_tres_gpus(
        "cpu=1,mem=180G,node=1,billing=15,gres/gpu=1,gres/gpu:rtx_5000_ada=1"
    )
    assert total == 1
    assert typed == {"rtx_5000_ada": 1}

    total, typed = parse_tres_gpus("cpu=6,mem=768G,node=1,billing=51")
    assert total == 0
    assert typed == {}

    total, typed = parse_tres_gpus("cpu=2,gres/gpu=4")
    assert total == 4
    assert typed == {"gpu": 4}


def test_parse_cpu_state():
    assert parse_cpu_state("35/93/0/128") == (35, 93, 0, 128)
    assert parse_cpu_state("") == (0, 0, 0, 0)


def test_humanize_seconds():
    assert humanize_seconds(None) == "∞"
    assert humanize_seconds(0) == "00:00"
    assert humanize_seconds(90061) == "1d 01:01"
