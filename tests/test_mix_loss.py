import importlib.util
from pathlib import Path


def load_mix_criterion():
    module_path = Path(__file__).resolve().parents[1] / "src" / "loss" / "mix.py"
    spec = importlib.util.spec_from_file_location("mix_loss_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MixCriterion


def test_mix_criterion_combines_base_loss_values():
    MixCriterion = load_mix_criterion()

    def base_loss(preds, targets):
        return targets

    criterion = MixCriterion(base_loss)

    assert criterion(None, (10.0, 30.0, 0.25)) == 25.0
