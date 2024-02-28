from omegaconf import OmegaConf


def get_experiment_trackers(config):
    exp_trackers = {}

    config_dict = OmegaConf.to_container(
        config, 
        resolve=True
    )

    if not config.debug:
        if config.use_wandb:
            try:
                from .wandb import WandbTracker
                exp_trackers['wandb']  = WandbTracker(
                    config,
                    config_dict
                )
            except:
                print('wandb is not installed')
        
        if config.use_mlflow:
            try:
                from .mlflow import MLFlowTracker
                exp_trackers['mlflow'] = MLFlowTracker(
                    config,
                    config_dict
                )
            except:
                print('mlflow is not installed')
            
    return exp_trackers