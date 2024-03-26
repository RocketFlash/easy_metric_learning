from omegaconf import OmegaConf


def get_experiment_trackers(config):
    exp_trackers = {}

    config_dict = OmegaConf.to_container(
        config, 
        resolve=True
    )

    if config.use_wandb:
        try:
            from .wandb import WandbTracker
            use_wandb = True
        except:
            use_wandb = False
            print('wandb is not installed')

        if use_wandb:
            exp_trackers['wandb'] = WandbTracker(
                config,
                config_dict
            )
    
    if config.use_mlflow:
        try:
            from .mlflow import MLFlowTracker
            use_mlflow = True
        except:
            use_mlflow = False
            print('mlflow is not installed')

        if use_mlflow:
            exp_trackers['mlflow'] = MLFlowTracker(
                config,
                config_dict
            )
        
    return exp_trackers