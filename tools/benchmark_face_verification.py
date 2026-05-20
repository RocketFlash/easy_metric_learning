import argparse
import json
import sys

sys.path.append("./")

from src.evaluator.megaface import load_embedding_npz
from src.evaluator.verification import evaluate_verification_pairs, load_pairs_csv


def _parse_float_list(value):
    return tuple(float(item) for item in str(value).split(",") if item)


def _parse_json_mapping(value):
    if value is None:
        return {}
    return json.loads(value)


def _write_metrics(metrics, output_path=None):
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(payload)
            f.write("\n")
    print(payload)


def run_face_verification_benchmark(
    embeddings_path,
    pairs_path,
    protocol="csv",
    fars=(1e-4, 1e-3, 1e-2),
    normalize=True,
    protocol_kwargs=None,
    file1_col="file1",
    file2_col="file2",
    label_col="is_same",
):
    embeddings, _, file_names = load_embedding_npz(embeddings_path)
    if file_names is None:
        raise ValueError("Face verification benchmark requires file_names in npz")
    pairs, is_same = load_pairs_csv(
        pairs_path,
        file_names=file_names,
        file1_col=file1_col,
        file2_col=file2_col,
        label_col=label_col,
        protocol=protocol,
        protocol_kwargs=protocol_kwargs,
    )
    return evaluate_verification_pairs(
        embeddings,
        pairs,
        is_same,
        fars=fars,
        normalize=normalize,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run CSV/LFW/AgeDB/CFP face verification from embeddings.npz"
    )
    parser.add_argument("--embeddings", required=True, help="npz with embeddings")
    parser.add_argument("--pairs", required=True, help="pair protocol file")
    parser.add_argument("--protocol", default="csv")
    parser.add_argument("--fars", default="1e-4,1e-3,1e-2")
    parser.add_argument("--protocol-kwargs-json")
    parser.add_argument("--file1-col", default="file1")
    parser.add_argument("--file2-col", default="file2")
    parser.add_argument("--label-col", default="is_same")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()

    metrics = run_face_verification_benchmark(
        args.embeddings,
        args.pairs,
        protocol=args.protocol,
        fars=_parse_float_list(args.fars),
        normalize=not args.no_normalize,
        protocol_kwargs=_parse_json_mapping(args.protocol_kwargs_json),
        file1_col=args.file1_col,
        file2_col=args.file2_col,
        label_col=args.label_col,
    )
    _write_metrics(metrics, output_path=args.output)


if __name__ == "__main__":
    main()
