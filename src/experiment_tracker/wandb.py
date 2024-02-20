import wandb


class WandbTracker():
    def __init__(self, config, config_dict):
        project_name = config.project_name
        run_name = config.run_name

        self.wandb_run = wandb.init(project=project_name,
                               name=run_name,
                               reinit=True)
        
        self.wandb_run.config.update(config_dict)


    def send_stats(self, stats):
        wandb_stats = {}
        stats_train = stats['train']
        stats_valid = stats['valid'] if 'valid' in stats else None

        wandb_stats['learning_rate'] = stats['lr']

        if stats_train['images_wdb']:
            wandb_stats["training batch"] = stats_train['images_wdb']

        for k, v in stats_train['losses'].items():
            wandb_stats[f'train/{k}'] = v
        
        if 'metrics' in stats_train:
            for mode_i, stats_valid_i in stats_train['metrics'].items():
                for k_metric, v_metric in stats_valid_i.items():
                    for k, v in v_metric.items():
                        if k == 'per_class':
                            pass
                        else:
                            wandb_stats[f'train/{mode_i}_{k_metric}_{k}'] = v
            
        if stats_valid is not None:
            if stats_valid['images_wdb']:
                wandb_stats["validation batch"] = stats_valid['images_wdb']

            for k, v in stats_valid['losses'].items():
                wandb_stats[f'validation/{k}'] = v

            if 'metrics' in stats_valid:
                for mode_i, stats_valid_i in stats_valid['metrics'].items():
                    for k_metric, v_metric in stats_valid_i.items():
                        for k, v in v_metric.items():
                            if k == 'per_class':
                                pass
                            else:
                                wandb_stats[f'validation/{mode_i}_{k_metric}_{k}'] = v

        self.wandb_run.log(wandb_stats, step=stats['epoch'])

    
    def finish_run(self):
        self.wandb_run.finish()