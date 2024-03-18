import wandb
from pathlib import Path
from .base import ExperimentTracker


class WandbTracker(ExperimentTracker):
    def __init__(self, config, config_dict):
        self.visualize_batch = config.visualize_batch
        self.work_dir = Path(config.work_dirs) / config.run_name

        self.wandb_run = wandb.init(
            project=config.project_name,
            name=config.run_name,
            reinit=True
        )
        
        self.wandb_run.config.update(config_dict)


    def send_stats(self, stats):
        epoch = stats['epoch']
        wandb_stats = self.parse_stats(stats)
        
        if self.visualize_batch and epoch==1:
            batch_images_train_path = self.work_dir / 'train_batch.png'
            batch_images_valid_path = self.work_dir / 'valid_batch.png'

            if batch_images_train_path.is_file():
                batch_images_train = wandb.Image(
                    str(batch_images_train_path), 
                    caption=f'train_batch'
                )
                wandb_stats["train_batch"] = batch_images_train

            if batch_images_valid_path.is_file():
                batch_images_valid = wandb.Image(
                    str(batch_images_valid_path), 
                    caption=f'valid_batch'
                )
                wandb_stats["valid_batch"] = batch_images_valid

        self.wandb_run.log(wandb_stats, step=epoch)

    
    def finish_run(self):
        self.wandb_run.finish()