
class ExperimentTracker():
    def __init__(self, config, config_dict=None):
        self.config = config
        self.config_dict = config_dict


    def parse_stats(self, stats):
        stats_dict = {}
        stats_train = stats['train']
        stats_valid = stats['valid'] if 'valid' in stats else None
        stats_eval  = stats['eval']

        stats_dict['learning_rate'] = stats_train['learning_rate']
        if 'm' in stats_train:
            stats_dict['train/m'] = stats_train['m']

        for k_loss, v_loss in stats_train['losses'].items():
            stats_dict[f'train/{k_loss}'] = v_loss

        if 'metrics' in stats_train:
            for k_metric, v_metric in stats_train['metrics'].items():
                stats_dict[f'train/{k_metric}'] = v_metric

        if stats_valid is not None:
            for k_loss, v_loss in stats_valid['losses'].items():
                stats_dict[f'valid/{k_loss}'] = v_loss

            if 'metrics' in stats_valid:
                for k_metric, v_metric in stats_valid['metrics'].items():
                    stats_dict[f'valid/{k_metric}'] = v_metric

        if stats_eval:
            for dataset_name, dataset_metrics in stats_eval.items():
                for k_metric, v_metric in dataset_metrics.items():
                    stats_dict[f'{dataset_name}/{k_metric}'] = v_metric

        return stats_dict

        

    def send_stats(self, stats):
        pass

    
    def finish_run(self):
        pass