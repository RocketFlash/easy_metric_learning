from omegaconf import OmegaConf


def get_experiment_trackers(config):
    exp_trackers = {}

    config_dict = OmegaConf.to_container(config, resolve=True)

    if config.use_wandb:
        try:
            from .wandb import WandbTracker

            use_wandb = True
        except ModuleNotFoundError as exc:
            if exc.name != "wandb":
                raise
            use_wandb = False
            print("wandb is not installed")

        if use_wandb:
            exp_trackers["wandb"] = WandbTracker(config, config_dict)

    if config.use_mlflow:
        if not getattr(config, "mlflow_server_uri", ""):
            print("mlflow tracking uri is not configured; skipping mlflow")
        else:
            try:
                from .mlflow import MLFlowTracker

                use_mlflow = True
            except ModuleNotFoundError as exc:
                if exc.name != "mlflow":
                    raise
                use_mlflow = False
                print("mlflow is not installed")

            if use_mlflow:
                exp_trackers["mlflow"] = MLFlowTracker(config, config_dict)

    if getattr(config, "use_tensorboard", False):
        try:
            from .tensorboard import TensorBoardTracker

            exp_trackers["tensorboard"] = TensorBoardTracker(config, config_dict)
        except ModuleNotFoundError as exc:
            if exc.name != "tensorboard":
                raise
            print("tensorboard is not installed")

    if getattr(config, "use_aim", False):
        try:
            from .aim import AimTracker

            exp_trackers["aim"] = AimTracker(config, config_dict)
        except ModuleNotFoundError as exc:
            if exc.name != "aim":
                raise
            print("aim is not installed")

    if getattr(config, "use_neptune", False):
        if not getattr(config, "neptune_project", ""):
            print("neptune project is not configured; skipping neptune")
        else:
            try:
                from .neptune import NeptuneTracker

                exp_trackers["neptune"] = NeptuneTracker(config, config_dict)
            except ModuleNotFoundError as exc:
                if exc.name != "neptune":
                    raise
                print("neptune is not installed")

    return exp_trackers
