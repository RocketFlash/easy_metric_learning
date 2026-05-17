from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from src.evaluator.ddp import DDPEvaluator


class ObjectGatherAccelerator:
    device = "cpu"
    is_local_main_process = True
    num_processes = 2

    def prepare(self, dataloader):
        return dataloader

    def unwrap_model(self, model):
        return model

    def gather_for_metrics(self, value, use_gather_object=False):
        if use_gather_object:
            raise TypeError("object gather unavailable")
        return value


def test_ddp_file_name_gather_fails_loudly_on_short_result(tmp_path):
    evaluator = DDPEvaluator(
        config=SimpleNamespace(debug=False),
        model=object(),
        save_dir=tmp_path,
        accelerator=ObjectGatherAccelerator(),
        is_eval=False,
    )

    with pytest.raises(RuntimeError, match="Gathered 1 file names"):
        evaluator._gather_file_names(["a.jpg"], batch_size=2)
