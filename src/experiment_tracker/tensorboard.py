from pathlib import Path

from .base import ExperimentTracker


class TensorBoardTracker(ExperimentTracker):
    def __init__(self, config, config_dict, writer=None):
        self.visualize_batch = config.visualize_batch
        self.work_dir = Path(config.work_dirs) / config.run_name
        if writer is None:
            from torch.utils.tensorboard import SummaryWriter

            log_root = getattr(config, "tensorboard_log_dir", None)
            if log_root:
                log_dir = Path(log_root) / config.run_name
            else:
                log_dir = self.work_dir / "tensorboard"
            writer = SummaryWriter(log_dir=str(log_dir))
        self.writer = writer
        self.writer.add_text("config", "```yaml\n" + str(config_dict) + "\n```", 0)

    def send_stats(self, stats):
        epoch = stats["epoch"]
        tensorboard_stats = self.parse_stats(stats)
        for key, value in tensorboard_stats.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

        if self.visualize_batch and epoch == 1:
            for split in ["train", "valid"]:
                batch_path = self.work_dir / f"{split}_batch.png"
                if batch_path.is_file():
                    self.writer.add_text(
                        f"{split}_batch_path",
                        str(batch_path),
                        epoch,
                    )
        self.writer.flush()

    def finish_run(self):
        self.writer.close()
