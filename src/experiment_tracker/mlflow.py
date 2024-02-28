import mlflow
from pathlib import Path
from .base import ExperimentTracker


class MLFlowTracker(ExperimentTracker):
    def __init__(self, config, config_dict):
        self.visualize_batch = config.visualize_batch
        self.work_dir = Path(config.work_dirs) / config.run_name
        mlflow.set_tracking_uri(config.mlflow_server_uri)
        mlflow.set_experiment(experiment_name=config.project_name)
        mlflow.start_run(run_name=config.run_name)
        mlflow.log_dict(config_dict, "config.yaml")


    def send_stats(self, stats):
        mlflow_stats = {}
        epoch = stats['epoch']
        stats_train = stats['train']
        stats_valid = stats['valid'] if 'valid' in stats else None
        mlflow_stats['learning_rate'] = stats['lr']

        for k_loss, v_loss in stats_train['losses'].items():
            mlflow_stats[f'train/{k_loss}'] = v_loss

        if 'metrics' in stats_train:
            for k_metric, v_metric in stats_train['metrics'].items():
                mlflow_stats[f'train/{k_metric}'] = v_metric

        if stats_valid is not None:
            for k_loss, v_loss in stats_valid['losses'].items():
                mlflow_stats[f'valid/{k_loss}'] = v_loss

            if 'metrics' in stats_valid:
                for k_metric, v_metric in stats_valid['metrics'].items():
                    mlflow_stats[f'valid/{k_metric}'] = v_metric

        if self.visualize_batch and epoch==1:
            batch_images_train_path = self.work_dir / 'train_batch.png'
            if batch_images_train_path.is_file():
                mlflow.log_artifact(batch_images_train_path)

            if stats_valid is not None:
                batch_images_valid_path = self.work_dir / 'valid_batch.png'
                if batch_images_valid_path.is_file():
                    mlflow.log_artifact(batch_images_valid_path)

        # last_model = stats['last_epoch_model']
        # mlflow.pytorch.log_model(last_model, 'last_checkpoint')

        # if 'best_epoch_model' in stats:
        #     best_model = stats['best_epoch_model']
        #     mlflow.pytorch.log_model(best_model, 'best_checkpoint')
        
        for k, v in mlflow_stats.items():
            k = k.replace('@', '/')
            k = k.replace(':', '-')
            mlflow.log_metric(k, v, step=epoch)


    def finish_run(self):
        mlflow.end_run()