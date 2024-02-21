import logging
import time
import json


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
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger("Logger")
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.setLevel(logging.INFO)


    def info(self, txt):
        self.logger.info(txt)


    def info_config(
            self,
            config_dict
        ):
        self.logger.info(
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

        self.logger.info(f'''
        ============   Dataset info          ============
        N classes total    : {dataset_stats.n_classes_total}
        N classes train    : {dataset_stats.n_classes_train}
        N classes valid    : {dataset_stats.n_classes_valid}

        N samples total    : {dataset_stats.n_samples_total}
        N samples train    : {dataset_stats.n_samples_train}
        N samples valid    : {dataset_stats.n_samples_valid}
        =================================================''')

    
    def info_model(
            self, 
            config, 
        ):

        self.logger.info(f'''
        ============   Model parameters   ===============
        Backbone           : {config.backbone.type}
        Head               : {config.head.type}
        Margin             : {config.margin.type}
        Embeddings size    : {config.embeddings_size}
        =================================================''')


    def info_epoch_train(self, epoch, stats_train, stats_valid):
        epoch_info_str = f'Epoch: {epoch} Train Loss: {stats_train.loss:.5f} Train Acc: {stats_train.acc:.5f}\n'
        if stats_valid is not None:
            epoch_info_str += f'{" "*37} Valid Loss: {stats_valid.loss:.5f} Valid Acc: {stats_valid.acc:.5f}'
            if stats_valid.gap is not None:
                epoch_info_str += f'\n{" "*37} GAP value : {stats_valid.gap:.5f}'
        self.logger.info(epoch_info_str)
    

    def info_epoch_time(self, start_time, start_epoch, epoch, num_epochs, workdir_path):
        elapsed, remaining = calculate_time(start_time=start_time, 
                                            start_epoch=start_epoch, 
                                            epoch=epoch, 
                                            epochs=num_epochs)

        self.logger.info(f"Epoch {epoch}/{num_epochs} finishied, saved to {workdir_path} ." + \
                         f"\n{' '*37} Elapsed {elapsed.days:d} days {elapsed.hours:d} hours {elapsed.minutes:d} minutes." + \
                         f"\n{' '*37} Remaining {remaining.days:d} days {remaining.hours:d} hours {remaining.minutes:d} minutes.")

    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()
