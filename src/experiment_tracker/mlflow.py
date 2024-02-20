import mlflow
import cv2
from pathlib import Path
import yaml


class MLFlowTracker():
    def __init__(self, config, config_dict):
        self.work_dir = Path(config.MISC.WORK_DIR)
        mlflow.set_tracking_uri(config.GENERAL.MLFLOW_SERVER_URI)
        mlflow.set_experiment(experiment_name=configs.MISC.PROJECT_NAME)
        
        mlflow.start_run(run_name=configs.MISC.RUN_NAME)
        params = {
            'img_h' : configs.DATA.IMG_SIZE_H,
            'img_w' : configs.DATA.IMG_SIZE_W,
            'batch_size' : configs.DATA.BATCH_SIZE,
            'decoder' : configs.MODEL.DECODER,
            'decoder_ch' : configs.MODEL.DECODER_CH,
            'decoder_n_blocks' : configs.MODEL.DECODER_N_BLOCKS,
            'decoder_merge_policy' : configs.MODEL.DECODER_MERGE_POLICY,
            'encoder' : configs.MODEL.ENCODER,
            'head' : configs.MODEL.HEAD,
            'head_ch' : configs.MODEL.HEAD_CH,
            'head_n_blocks' : configs.MODEL.HEAD_N_BLOCKS,
            'head_use_separable_conv' : configs.MODEL.HEAD_SEP_CONV,
            'head_activation' : configs.MODEL.HEAD_ACTIVATION,
            'scale_R' : configs.MODEL.SCALE_R,
            'single_head' : configs.MODEL.SINGLE_HEAD,
            'loss/regr_loss_type_offset' : configs.TRAIN.LOSS.REGR_LOSS_TYPE_OFFSET,
            'loss/regr_loss_type_params' : configs.TRAIN.LOSS.REGR_LOSS_TYPE_PARAMS,
            'loss/w_clsf' : configs.TRAIN.LOSS.W_CLSF,
            'loss/w_offset' : configs.TRAIN.LOSS.W_OFFSET,
            'loss/w_params' : configs.TRAIN.LOSS.W_PARAMS,
            'loss/alpha' : configs.TRAIN.LOSS.ALPHA,
            'loss/beta' : configs.TRAIN.LOSS.BETA,
            'epochs' : configs.TRAIN.EPOCHS,
            'optimizer' : configs.TRAIN.OPTIMIZER.OPTIMIZER_TYPE,
            'scheduler' : configs.TRAIN.SCHEDULER.SCHEDULER_TYPE,
            'amp' : configs.TRAIN.AMP,
            'grad_clip' : configs.TRAIN.GRAD_CLIP,
            'dataset_info' : configs.MISC.DATASET_INFO,
            'mode' : configs.GENERAL.MODE
        }
        config_path = self.work_dir / 'config.yaml'
        mlflow.log_params(params)
        mlflow.log_dict(yaml.safe_load(open(config_path)), "config.yaml")


    def send_stats(self, stats):
        mlflow_stats = {}
        stats_train = stats['train']
        stats_valid = stats['valid'] if 'valid' in stats else None

        last_model = stats['last_epoch_model']
        mlflow.pytorch.log_model(last_model, 'last_checkpoint')

        if 'best_epoch_model' in stats:
            best_model = stats['best_epoch_model']
            mlflow.pytorch.log_model(best_model, 'best_checkpoint')

        mlflow_stats['learning_rate'] = stats['lr']

        if stats_train['images_wdb']:
            train_batch = cv2.imread(str(self.work_dir / 'train_batch.png'))
            train_batch = cv2.cvtColor(train_batch, cv2.COLOR_BGR2RGB)
            mlflow.log_image(train_batch, 'train_batch.png')

        for k, v in stats_train['losses'].items():
            mlflow_stats[f'train/{k}'] = v
        
        if 'metrics' in stats_train:
            for mode_i, stats_valid_i in stats_train['metrics'].items():
                for k_metric, v_metric in stats_valid_i.items():
                    for k, v in v_metric.items():
                        if k == 'per_class':
                            pass
                        else:
                            mlflow_stats[f'train/{mode_i}_{k_metric}_{k}'] = v
            
        if stats_valid is not None:
            if stats_valid['images_wdb']:
                valid_batch = cv2.imread(str(self.work_dir / 'valid_batch.png'))
                valid_batch = cv2.cvtColor(valid_batch, cv2.COLOR_BGR2RGB)
                mlflow.log_image(valid_batch, 'valid_batch.png')

            for k, v in stats_valid['losses'].items():
                mlflow_stats[f'validation/{k}'] = v

            if 'metrics' in stats_valid:
                for mode_i, stats_valid_i in stats_valid['metrics'].items():
                    for k_metric, v_metric in stats_valid_i.items():
                        for k, v in v_metric.items():
                            if k == 'per_class':
                                pass
                            else:
                                mlflow_stats[f'validation/{mode_i}_{k_metric}_{k}'] = v

        for k, v in mlflow_stats.items():
            k = k.replace('@', '/')
            k = k.replace(':', '-')
            mlflow.log_metric(k, v, step=stats['epoch'])


    def finish_run(self):
        mlflow.end_run()