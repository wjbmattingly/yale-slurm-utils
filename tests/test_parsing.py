"""Unit tests for the pure parsing helpers (no SLURM required)."""

from __future__ import annotations

from datetime import datetime

from yale_slurm_utils.parsing import (
    expand_slurm_filename,
    first_node,
    humanize_seconds,
    iter_scontrol_nodes,
    parse_cpu_state,
    parse_gres,
    parse_mem_mb,
    parse_slurm_time,
    parse_timestamp,
    parse_tres_cpu,
    parse_tres_gpus,
    parse_tres_mem,
    state_is_down,
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


def test_parse_tres_mem():
    assert parse_tres_mem("cpu=1,mem=180G,node=1,billing=15") == 180 * 1024
    assert parse_tres_mem("cpu=6,mem=64000M,node=1") == 64000
    assert parse_tres_mem("cpu=2,node=1") is None
    assert parse_tres_mem("(null)") is None


def test_first_node():
    assert first_node("c01n04") == "c01n04"
    assert first_node("gpu[01-03]") == "gpu01"
    assert first_node("c[01,05,09]") == "c01"
    assert first_node("a01,a02") == "a01"
    assert first_node("(null)") is None


def test_expand_slurm_filename():
    # Array job: %A -> master, %a -> task, resolved from a "12345_6" id.
    assert (
        expand_slurm_filename(
            "transcribe_%A_%a.out", job_id="12345_6", job_name="transcribe"
        )
        == "transcribe_12345_6.out"
    )
    # Plain job with %j and %x.
    assert (
        expand_slurm_filename("%x-%j.log", job_id="9999", job_name="train")
        == "train-9999.log"
    )
    # %N resolves from the nodelist; %% stays a literal percent.
    assert (
        expand_slurm_filename("o_%N_%%.txt", job_id="5", nodelist="gpu[02-04]")
        == "o_gpu02_%.txt"
    )
    # Unknown values leave the token in place rather than inventing data.
    assert expand_slurm_filename("j_%u.out", job_id="5") == "j_%u.out"
    assert expand_slurm_filename("(null)") is None


def test_parse_cpu_state():
    assert parse_cpu_state("35/93/0/128") == (35, 93, 0, 128)
    assert parse_cpu_state("") == (0, 0, 0, 0)


def test_humanize_seconds():
    assert humanize_seconds(None) == "∞"
    assert humanize_seconds(0) == "00:00"
    assert humanize_seconds(90061) == "1d 01:01"


def test_parse_tres_cpu():
    assert parse_tres_cpu("cpu=64,mem=1000G,gres/gpu=8") == 64
    assert parse_tres_cpu("") == 0
    assert parse_tres_cpu("mem=8G") == 0


def test_state_is_down():
    assert state_is_down("DOWN")
    assert state_is_down("MIXED+DRAIN")
    assert state_is_down("MAINT")
    assert not state_is_down("IDLE")
    assert not state_is_down("ALLOCATED")
    assert not state_is_down("MIXED")


# A trimmed `scontrol -d -o show node` sample: one fully-allocated b200 node in
# three overlapping partitions, and one b200 node in maintenance (nothing
# allocated). AllocTRES only carries an *untyped* gres/gpu total.
_SCONTROL_SAMPLE = (
    "NodeName=c01n01 CPUAlloc=64 CPUTot=64 State=ALLOCATED "
    "Partitions=gpu,gpu_b200,gpu_devel "
    "CfgTRES=cpu=64,mem=1000G,billing=64,gres/gpu=8,gres/gpu:b200=8 "
    "AllocTRES=cpu=64,mem=800G,gres/gpu=8\n"
    "NodeName=c01n08 CPUAlloc=0 CPUTot=64 State=MAINT "
    "Partitions=gpu,gpu_b200 "
    "CfgTRES=cpu=64,mem=1000G,billing=64,gres/gpu=8,gres/gpu:b200=8 "
    "AllocTRES=\n"
)


def test_iter_scontrol_nodes():
    nodes = iter_scontrol_nodes(_SCONTROL_SAMPLE)
    assert len(nodes) == 2

    n1 = nodes[0]
    assert n1["name"] == "c01n01"
    assert n1["partitions"] == ["gpu", "gpu_b200", "gpu_devel"]
    assert n1["cpus_total"] == 64 and n1["cpus_alloc"] == 64
    assert n1["gpus_total"] == {"b200": 8}
    # Untyped AllocTRES total is mapped onto the node's single GPU model.
    assert n1["gpus_used"] == {"b200": 8}

    n2 = nodes[1]
    assert n2["name"] == "c01n08"
    assert n2["state"] == "MAINT"
    assert n2["gpus_total"] == {"b200": 8}
    assert n2["gpus_used"] == {}  # empty AllocTRES

    assert iter_scontrol_nodes("") == []
