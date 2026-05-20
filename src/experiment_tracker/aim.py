from .base import ExperimentTracker


class AimTracker(ExperimentTracker):
    def __init__(self, config, config_dict, run=None):
        if run is None:
            from aim import Run

            aim_repo = getattr(config, "aim_repo", None)
            kwargs = {"experiment": config.project_name}
            if aim_repo:
                kwargs["repo"] = aim_repo
            run = Run(**kwargs)
        self.run = run
        self.run.name = config.run_name
        self.run["config"] = config_dict

    def send_stats(self, stats):
        epoch = stats["epoch"]
        aim_stats = self.parse_stats(stats)
        for key, value in aim_stats.items():
            if isinstance(value, (int, float)):
                self.run.track(value, name=key, step=epoch)

    def finish_run(self):
        self.run.close()
