import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("albumentations")

from src.utils import load_ckp, save_ckp


def test_checkpoint_restores_optimizer_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(2, 2)
    loss = model(data).sum()
    loss.backward()
    optimizer.step()

    checkpoint_path = tmp_path / "model.pt"
    save_ckp(checkpoint_path, model=model, optimizer=optimizer, epoch=3)

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(
        restored_model.parameters(),
        lr=0.1,
        momentum=0.9,
    )

    _, restored_optimizer, epoch, _ = load_ckp(
        checkpoint_path,
        restored_model,
        optimizer=restored_optimizer,
        device="cpu",
    )

    assert epoch == 3
    assert restored_optimizer.state_dict()["state"]


def test_checkpoint_rejects_shape_mismatches_by_default(tmp_path):
    model = torch.nn.Linear(2, 1)
    checkpoint_path = tmp_path / "model.pt"
    save_ckp(
        checkpoint_path,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )

    incompatible_model = torch.nn.Linear(3, 1)

    with pytest.raises(RuntimeError, match="shape mismatches"):
        load_ckp(checkpoint_path, incompatible_model, device="cpu")


def test_checkpoint_rejects_module_class_mismatches_by_default(tmp_path):
    class SourceModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))

    class TargetModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))

    model = SourceModule()
    checkpoint_path = tmp_path / "model.pt"
    save_ckp(
        checkpoint_path,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )

    with pytest.raises(RuntimeError, match="module class mismatches"):
        load_ckp(checkpoint_path, TargetModule(), device="cpu")
