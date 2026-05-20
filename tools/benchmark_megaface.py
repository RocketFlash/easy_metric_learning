import argparse
import json
import sys

sys.path.append("./")

from src.evaluator.megaface import evaluate_megaface_npz


def _parse_float_list(value):
    return tuple(float(item) for item in str(value).split(",") if item)


def _parse_int_list(value):
    return tuple(int(item) for item in str(value).split(",") if item)


def _write_metrics(metrics, output_path=None):
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(payload)
            f.write("\n")
    print(payload)


def run_megaface_benchmark(
    probe_path,
    gallery_path,
    distractor_path=None,
    ranks=(1, 10),
    fpirs=(1e-3, 1e-2),
):
    return evaluate_megaface_npz(
        probe_path,
        gallery_path,
        distractor_path=distractor_path,
        ranks=ranks,
        fpirs=fpirs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run MegaFace-style identification from embedding npz files"
    )
    parser.add_argument("--probe", required=True, help="probe embeddings npz")
    parser.add_argument("--gallery", required=True, help="gallery embeddings npz")
    parser.add_argument("--distractor", help="optional distractor embeddings npz")
    parser.add_argument("--ranks", default="1,10")
    parser.add_argument("--fpirs", default="1e-3,1e-2")
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()

    metrics = run_megaface_benchmark(
        args.probe,
        args.gallery,
        distractor_path=args.distractor,
        ranks=_parse_int_list(args.ranks),
        fpirs=_parse_float_list(args.fpirs),
    )
    _write_metrics(metrics, output_path=args.output)


if __name__ == "__main__":
    main()
