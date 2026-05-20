from .base import ExperimentTracker


class NeptuneTracker(ExperimentTracker):
    def __init__(self, config, config_dict, run=None):
        if run is None:
            import neptune

            run = neptune.init_run(
                project=config.neptune_project,
                api_token=getattr(config, "neptune_api_token", None) or None,
                name=config.run_name,
                mode=getattr(config, "neptune_mode", "async"),
            )
        self.run = run
        self.run["config"] = config_dict

    def send_stats(self, stats):
        epoch = stats["epoch"]
        neptune_stats = self.parse_stats(stats)
        for key, value in neptune_stats.items():
            if isinstance(value, (int, float)):
                self.run[key].append(value, step=epoch)

    def finish_run(self):
        self.run.stop()
