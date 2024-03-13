import logging
import time
import json
from omegaconf import OmegaConf


class DayHourMinute(object):
    def __init__(self, seconds):
        self.days = int(seconds // 86400)
        self.hours = int((seconds - (self.days * 86400)) // 3600)
        self.minutes = int((seconds - self.days * 86400 - self.hours * 3600) // 60)


def calculate_time(start_time, start_epoch, epoch, epochs):
    t = time.time() - start_time
    elapsed = DayHourMinute(t)
    t /= (epoch + 1) - start_epoch  # seconds per epoch
    t = (epochs - epoch - 1) * t
    remaining = DayHourMinute(t)
    return elapsed, remaining


class Logger():
    def __init__(self, path=None):
        self.logger = logging.getLogger("Logger")

        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.stdout_handler)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

        if path is not None:
            self.file_handler = logging.FileHandler(path, "w")
            self.logger.addHandler(self.file_handler)
            self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        
        self.logger.setLevel(logging.INFO)
        self.tab_string = '    '


    def info(self, txt):
        self.logger.info(txt)


    def info_config(
            self,
            config
        ):

        config_dict = OmegaConf.to_container(
            config, 
            resolve=True
        )
        
        self.info(
            json.dumps(
                config_dict, 
                sort_keys=False, 
                indent=4
            )
        )


    def info_data(
            self, 
            dataset_stats
        ):
        if dataset_stats is not None:
            header_str = f"{dataset_stats.split} dataset info"
            dataset_info_header = f'\n{self.tab_string*2}=============== {header_str:<15} ===============\n'
            dataset_info_str = dataset_info_header + dataset_stats.__repr__()
            self.info(dataset_info_str)

    
    def info_model(
            self, 
            config, 
        ):

        model_info_header = f'\n{self.tab_string*2}=============== {"Model params":<15} ===============\n'
        model_info_str = model_info_header
        model_info_str += f'{self.tab_string*2}{"Backbone":<15}: {config.backbone.type}\n'
        model_info_str += f'{self.tab_string*2}{"Head":<15}: {config.head.type}\n'
        model_info_str += f'{self.tab_string*2}{"Margin":<15}: {config.margin.type}\n'
        model_info_str += f'{self.tab_string*2}{"Embeddings size":<15}: {config.embeddings_size}\n'
        self.info(model_info_str)


    def info_epoch_train(self, epoch, stats_train, stats_valid):
        epoch_info_str = f'Epoch {epoch} stats:\n'
        epoch_info_str += f'{self.tab_string}Train Losses:\n'
        for k_loss, v_loss in stats_train['losses'].items():
            epoch_info_str += f'{self.tab_string*2}{k_loss:<15} : {v_loss:.5f}\n'

        if stats_valid is not None:
            epoch_info_str += f'{self.tab_string}Valid Losses:\n'
            for k_loss, v_loss in stats_valid['losses'].items():
                epoch_info_str += f'{self.tab_string*2}{k_loss:<15} : {v_loss:.5f}\n'
            if 'metrics' in stats_valid:
                epoch_info_str += f'{self.tab_string}Valid metrics:\n'
                for k_metric, v_metric in stats_valid['metrics'].items():
                    epoch_info_str += f'{self.tab_string*2}{k_metric:<15} : {v_metric:.5f}\n'
        self.info(epoch_info_str)
    

    def info_epoch_time(self, start_time, start_epoch, epoch, num_epochs, workdir_path):
        elapsed, remaining = calculate_time(
            start_time=start_time, 
            start_epoch=start_epoch, 
            epoch=epoch, 
            epochs=num_epochs
        )
        time_info_str = f"Epoch {epoch}/{num_epochs} finishied\n"
        time_info_str += f"Checkpoint was saved in {workdir_path}\n"
        time_info_str += f"{self.tab_string*2}{'Elapsed':<15} {elapsed.days:d} days {elapsed.hours:d} hours {elapsed.minutes:d} minutes\n"
        time_info_str += f"{self.tab_string*2}{'Remaining':<15} {remaining.days:d} days {remaining.hours:d} hours {remaining.minutes:d} minutes\n"
        self.info(time_info_str)


    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()
