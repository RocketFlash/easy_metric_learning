import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Model conversion for mobile devices')
    parser.add_argument('--work_dirs', type=str, default='work_dirs/', help='path to trained model working directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    work_dirs = Path(args.work_dirs)
    runs_paths = [f for f in work_dirs.iterdir() if f.is_dir()]

    for run_path in runs_paths:
        (run_path / 'best.pt').unlink(missing_ok=True)
        (run_path / 'last.pt').unlink(missing_ok=True)
        (run_path / 'debug_last.pt').unlink(missing_ok=True)
        (run_path / 'debug_best.pt').unlink(missing_ok=True)
        (run_path / 'debug_best_emb.pt').unlink(missing_ok=True)
        (run_path / 'debug_last_emb.pt').unlink(missing_ok=True)
        (run_path / 'train_batch_0.png').unlink(missing_ok=True)
        (run_path / 'valid_batch_0.png').unlink(missing_ok=True)