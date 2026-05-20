from types import SimpleNamespace

from src.experiment_tracker.aim import AimTracker
from src.experiment_tracker.neptune import NeptuneTracker
from src.experiment_tracker.tensorboard import TensorBoardTracker


class DummyWriter:
    def __init__(self):
        self.scalars = []
        self.texts = []
        self.flushed = False
        self.closed = False

    def add_text(self, key, value, step):
        self.texts.append((key, value, step))

    def add_scalar(self, key, value, step):
        self.scalars.append((key, value, step))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


class DummyAimRun:
    def __init__(self):
        self.name = None
        self.items = {}
        self.tracked = []
        self.closed = False

    def __setitem__(self, key, value):
        self.items[key] = value

    def track(self, value, name, step):
        self.tracked.append((name, value, step))

    def close(self):
        self.closed = True


class DummyNeptuneChannel:
    def __init__(self):
        self.values = []

    def append(self, value, step):
        self.values.append((value, step))


class DummyNeptuneRun:
    def __init__(self):
        self.items = {}
        self.channels = {}
        self.stopped = False

    def __setitem__(self, key, value):
        self.items[key] = value

    def __getitem__(self, key):
        self.channels.setdefault(key, DummyNeptuneChannel())
        return self.channels[key]

    def stop(self):
        self.stopped = True


def make_config(tmp_path):
    return SimpleNamespace(
        visualize_batch=False,
        work_dirs=str(tmp_path),
        run_name="run",
        project_name="project",
        tensorboard_log_dir=None,
    )


def make_stats():
    return {
        "epoch": 3,
        "learning_rate": 0.1,
        "train": {"losses": {"total_loss": 1.5}},
        "valid": {"losses": {"total_loss": 1.0}},
        "eval": {"tiny": {"R@1": 0.75}},
    }


def test_tensorboard_tracker_logs_numeric_stats(tmp_path):
    writer = DummyWriter()
    tracker = TensorBoardTracker(
        make_config(tmp_path),
        {"run_name": "run"},
        writer=writer,
    )

    tracker.send_stats(make_stats())
    tracker.finish_run()

    assert ("learning_rate", 0.1, 3) in writer.scalars
    assert ("train/total_loss", 1.5, 3) in writer.scalars
    assert ("tiny/R@1", 0.75, 3) in writer.scalars
    assert writer.flushed is True
    assert writer.closed is True


def test_aim_tracker_logs_numeric_stats(tmp_path):
    run = DummyAimRun()
    tracker = AimTracker(make_config(tmp_path), {"run_name": "run"}, run=run)

    tracker.send_stats(make_stats())
    tracker.finish_run()

    assert run.name == "run"
    assert run.items["config"] == {"run_name": "run"}
    assert ("train/total_loss", 1.5, 3) in run.tracked
    assert run.closed is True


def test_neptune_tracker_logs_numeric_stats(tmp_path):
    run = DummyNeptuneRun()
    config = make_config(tmp_path)
    config.neptune_project = "workspace/project"
    tracker = NeptuneTracker(config, {"run_name": "run"}, run=run)

    tracker.send_stats(make_stats())
    tracker.finish_run()

    assert run.items["config"] == {"run_name": "run"}
    assert run.channels["tiny/R@1"].values == [(0.75, 3)]
    assert run.stopped is True
