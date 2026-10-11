"""A1 graph invariants; no dataset or training is required."""

import pytest
import torch

from dsu_research.compartment_ablation import base_topology
from dsu_research.pointsparse_topology import graph_hash, make_topology, structural_metrics
from dsu_research.sparse_compartment_controls import PointSparse, paired_values


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_paired_topologies(seed):
    graphs = {}
    metadata = {}
    metrics = {}
    for arm in ("A0", "A1", "A2", "A3"):
        graph, info = make_topology(arm, seed)
        repeated, again = make_topology(arm, seed)
        assert torch.equal(graph, repeated) and info == again
        assert graph.shape == (128, 64) and graph.dtype == torch.int64
        assert all(len(set(row.tolist())) == 64 for row in graph)
        assert graph_hash(graph) == info["final_hash"]
        graphs[arm], metadata[arm], metrics[arm] = graph, info, structural_metrics(graph)
        model = PointSparse(graph)
        before = [p.detach().clone() for p in model.parameters()]
        make_topology("A0", seed)
        assert all(torch.equal(x, y) for x, y in zip(before, model.parameters()))
        model.initialize(*paired_values(seed))
        assert model.counts()["parameters"] == 9610
        assert model.counts()["main_macs_per_sample"] == 9472
        assert model.initialize(*paired_values(seed))["g64_sha256"] == info["final_hash"]
    assert torch.equal(graphs["A0"], base_topology(seed))
    assert len({metadata[a]["base_hash"] for a in ("A1", "A2", "A3")}) == 1
    base_degree = torch.bincount(graphs["A1"].flatten(), minlength=784)
    assert int((base_degree == 11).sum()) == 352
    assert int((base_degree == 10).sum()) == 432
    for arm in ("A1", "A2", "A3"):
        assert torch.equal(torch.bincount(graphs[arm].flatten(), minlength=784), base_degree)
        assert metrics[arm]["coverage"] == 784 and metrics[arm]["connected"]
    assert metrics["A1"]["mean_jaccard"] - metrics["A2"]["mean_jaccard"] >= .00015
    assert metrics["A3"]["mean_jaccard"] - metrics["A1"]["mean_jaccard"] >= .001
    assert metrics["A1"]["max_jaccard"] - metrics["A2"]["max_jaccard"] >= .02
    assert metrics["A3"]["max_jaccard"] - metrics["A1"]["max_jaccard"] >= .20
    assert metadata["A2"]["objective_swaps"] > 0
    assert metadata["A3"]["objective_swaps"] > 0


def test_invalid_seed_and_graph():
    with pytest.raises(ValueError):
        make_topology("A1", -1)
    with pytest.raises(ValueError):
        make_topology("other", 1)
    graph, _ = make_topology("A1", 1)
    bad = graph.numpy().copy()
    bad[0, 1] = bad[0, 0]
    with pytest.raises(ValueError):
        structural_metrics(bad)


def test_seed_one_hashes_are_frozen():
    expected = {
        "A0": "fd7f2469a457bad1daa506df4df5bc4604dfc5ff39b6fc2f21b285a45acaaf1d",
        "A1": "f79c95ee69e2c25d13dc274ccda8f0e86e8e767ff346df70ddf65c776a4862bb",
        "A2": "f009d0404bb275ef6814188958f8bb3656f20d89d3cbdcccd6d6602a4f01333e",
        "A3": "4292c28b9aa8cad1f8a1554581072b2c1fc7a66505bb458f49e4e9dbadb4336f",
    }
    assert {arm: make_topology(arm, 1)[1]["final_hash"] for arm in expected} == expected
