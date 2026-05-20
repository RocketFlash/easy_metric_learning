import os

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
mp = pytest.importorskip("torch.multiprocessing")

import torch.nn.functional as F

from src.model.margin.partialfc import DistributedPartialFCArcMarginProduct
from src.model.margin import partialfc as partialfc_module


def deterministic_weight(out_features=6, in_features=4):
    return (
        torch.arange(out_features * in_features, dtype=torch.float32).view(
            out_features,
            in_features,
        )
        + 1.0
    )


def partialfc_worker(rank, world_size, init_file, queue):
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        torch.manual_seed(100 + rank)
        margin = DistributedPartialFCArcMarginProduct(
            in_features=4,
            out_features=6,
            s=1.0,
            m=0.0,
            sample_rate=1.0,
            min_sample_classes=1,
            easy_margin=False,
            ls_eps=0.0,
        )
        full_weight = deterministic_weight()
        with torch.no_grad():
            margin.weight.copy_(full_weight[margin.class_start : margin.class_end])

        x = torch.arange(8, dtype=torch.float32).view(2, 4) + 1.0 + rank * 11.0
        x.requires_grad_(True)
        labels = torch.tensor([rank, rank + 3], dtype=torch.long)

        output = margin(x, labels)
        expected = F.linear(F.normalize(x), F.normalize(full_weight))
        loss = output.sum()
        loss.backward()

        queue.put(
            {
                "rank": rank,
                "shape": tuple(output.shape),
                "max_diff": float((output.detach() - expected).abs().max()),
                "x_grad": x.grad is not None,
                "weight_grad": margin.weight.grad is not None,
                "local_out_features": margin.local_out_features,
            }
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_distributed_partialfc_matches_full_classifier_for_two_ranks(tmp_path):
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    world_size = 2
    init_file = tmp_path / "partialfc_init"
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=partialfc_worker,
            args=(rank, world_size, init_file, queue),
        )
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)

    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()
        assert process.exitcode == 0

    results = [queue.get(timeout=5) for _ in range(world_size)]
    assert sorted(result["rank"] for result in results) == [0, 1]
    for result in results:
        assert result["shape"] == (2, 6)
        assert result["max_diff"] < 1e-5
        assert result["x_grad"] is True
        assert result["weight_grad"] is True
        assert result["local_out_features"] == 3


def test_distributed_partialfc_lazily_uses_late_process_group_info(monkeypatch):
    margin = DistributedPartialFCArcMarginProduct(
        in_features=4,
        out_features=6,
        s=1.0,
        m=0.0,
        sample_rate=1.0,
        min_sample_classes=1,
        easy_margin=False,
        ls_eps=0.0,
    )
    monkeypatch.setattr(partialfc_module, "_distributed_info", lambda: (0, 2))

    margin._sync_distributed_info()
    sampled_global_classes = torch.tensor([0, 2])
    sampled_local_offsets = sampled_global_classes - margin.class_start
    sampled_weight = margin._select_sampled_weight(
        sampled_global_classes,
        sampled_local_offsets,
    )

    assert margin.world_size == 2
    assert margin.local_out_features == 3
    assert margin.weight.shape == (6, 4)
    assert torch.equal(
        sampled_weight, margin.weight.index_select(0, sampled_global_classes)
    )
