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
        wandb_stats = {}
        epoch = stats['epoch']
        stats_train = stats['train']
        stats_valid = stats['valid'] if 'valid' in stats else None

        wandb_stats['learning_rate'] = stats['lr']

        for k_loss, v_loss in stats_train['losses'].items():
            wandb_stats[f'train/{k_loss}'] = v_loss

        if 'metrics' in stats_train:
            for k_metric, v_metric in stats_train['metrics'].items():
                wandb_stats[f'train/{k_metric}'] = v_metric

        if stats_valid is not None:
            for k_loss, v_loss in stats_valid['losses'].items():
                wandb_stats[f'valid/{k_loss}'] = v_loss

            if 'metrics' in stats_valid:
                for k_metric, v_metric in stats_valid['metrics'].items():
                    wandb_stats[f'valid/{k_metric}'] = v_metric

        if self.visualize_batch and epoch==1:
            batch_images_train_path = self.work_dir / 'train_batch.png'
            if batch_images_train_path.is_file():
                batch_images_train = wandb.Image(
                    str(batch_images_train_path), 
                    caption=f'train_batch'
                )
                wandb_stats["train_batch"] = batch_images_train

            if stats_valid is not None:
                batch_images_valid_path = self.work_dir / 'valid_batch.png'
                if batch_images_valid_path.is_file():
                    batch_images_valid = wandb.Image(
                        str(batch_images_valid_path), 
                        caption=f'valid_batch'
                    )
                    wandb_stats["valid_batch"] = batch_images_valid

        self.wandb_run.log(wandb_stats, step=epoch)

    
    def finish_run(self):
        self.wandb_run.finish()