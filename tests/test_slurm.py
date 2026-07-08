"""Tests for GPU tallying: overlapping partitions and down nodes.

These exercise the fix (thanks to Charles Sindelar, YCRC) that counts each
physical node once via `scontrol` instead of double-counting across the many
overlapping partitions a node belongs to.
"""

from __future__ import annotations

from yale_slurm_utils import slurm


def _fake_nodes():
    # 7 fully-allocated b200 nodes + 1 in maintenance (nothing allocated).
    # Every node belongs to BOTH `gpu` and `gpu_b200` (overlapping partitions).
    nodes = []
    for i in range(7):
        nodes.append(
            {
                "name": f"c01n0{i + 1}",
                "state": "ALLOCATED",
                "partitions": ["gpu", "gpu_b200"],
                "cpus_total": 64,
                "cpus_alloc": 64,
                "mem_total_mb": 1000 * 1024,
                "mem_alloc_mb": 800 * 1024,
                "gpus_total": {"b200": 8},
                "gpus_used": {"b200": 8},
            }
        )
    nodes.append(
        {
            "name": "c01n08",
            "state": "MAINT",
            "partitions": ["gpu", "gpu_b200"],
            "cpus_total": 64,
            "cpus_alloc": 0,
            "mem_total_mb": 1000 * 1024,
            "mem_alloc_mb": 0,
            "gpus_total": {"b200": 8},
            "gpus_used": {},
        }
    )
    return nodes


def test_pool_inventory_counts_each_node_once(monkeypatch):
    monkeypatch.setattr(slurm, "_scontrol_nodes", _fake_nodes)

    pool = slurm.gpu_pool_inventory()
    assert len(pool) == 1
    b200 = pool[0]
    # 8 nodes x 8 = 64 physical cards, NOT 128 (despite 2 overlapping partitions).
    assert b200.total == 64
    assert b200.used == 56
    # The maintenance node's 8 cards are stranded, so nothing is actually free.
    assert b200.unavailable_gpus == 8
    assert b200.free == 0
    assert b200.nodes_total == 8
    assert b200.nodes_unavailable == 1

    # Restricting to a partition pools just that partition's nodes (still 64).
    assert slurm.gpu_pool_inventory("gpu_b200")[0].total == 64


def test_per_partition_inventory_is_accurate_per_partition(monkeypatch):
    monkeypatch.setattr(slurm, "_scontrol_nodes", _fake_nodes)

    inv = slurm.gpu_inventory()
    # One (partition, type) row per partition the nodes belong to.
    by_part = {gc.partition: gc for gc in inv}
    assert set(by_part) == {"gpu", "gpu_b200"}
    for gc in inv:
        assert gc.total == 64  # each partition genuinely fronts all 64 cards
        assert gc.used == 56
        assert gc.free == 0


def test_get_nodes_expands_over_partitions(monkeypatch):
    monkeypatch.setattr(slurm, "_scontrol_nodes", _fake_nodes)
    all_nodes = slurm.get_nodes()
    # 8 physical nodes x 2 partitions = 16 (node, partition) rows.
    assert len(all_nodes) == 16
    only_b200 = slurm.get_nodes("gpu_b200")
    assert len(only_b200) == 8
    assert all(n.partition == "gpu_b200" for n in only_b200)
