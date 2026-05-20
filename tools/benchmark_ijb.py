import argparse
import json
import sys

sys.path.append("./")

from src.evaluator.ijb import (
    aggregate_templates,
    load_template_metadata,
    load_template_pairs,
)
from src.evaluator.megaface import load_embedding_npz
from src.evaluator.verification import evaluate_verification_pairs


def _parse_float_list(value):
    return tuple(float(item) for item in str(value).split(",") if item)


def _write_metrics(metrics, output_path=None):
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(payload)
            f.write("\n")
    print(payload)


def run_ijb_template_benchmark(
    embeddings_path,
    metadata_path,
    pairs_path,
    fars=(1e-4, 1e-3, 1e-2),
    file_col="file_name",
    template_col="template_id",
    media_col="media_id",
    left_col="template1",
    right_col="template2",
    label_col="is_same",
):
    embeddings, _, file_names = load_embedding_npz(embeddings_path)
    if file_names is None:
        raise ValueError("IJB benchmark requires file_names in npz")
    template_ids, media_ids = load_template_metadata(
        metadata_path,
        file_names=file_names,
        file_col=file_col,
        template_col=template_col,
        media_col=media_col,
    )
    output_template_ids, template_embeddings = aggregate_templates(
        embeddings,
        template_ids,
        media_ids=media_ids,
        normalize=True,
    )
    template_id_to_index = {
        template_id: index for index, template_id in enumerate(output_template_ids)
    }
    pairs, is_same = load_template_pairs(
        pairs_path,
        template_id_to_index,
        left_col,
        right_col,
        label_col,
    )
    metrics = evaluate_verification_pairs(
        template_embeddings,
        pairs,
        is_same,
        fars=fars,
        normalize=False,
    )
    metrics["n_templates"] = len(output_template_ids)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run IJB-style template verification from embeddings.npz"
    )
    parser.add_argument("--embeddings", required=True, help="npz with embeddings")
    parser.add_argument(
        "--metadata", required=True, help="IJB face/template/media file"
    )
    parser.add_argument("--pairs", required=True, help="IJB template pair-label file")
    parser.add_argument("--fars", default="1e-4,1e-3,1e-2")
    parser.add_argument("--file-col", default="file_name")
    parser.add_argument("--template-col", default="template_id")
    parser.add_argument("--media-col", default="media_id")
    parser.add_argument("--left-col", default="template1")
    parser.add_argument("--right-col", default="template2")
    parser.add_argument("--label-col", default="is_same")
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()

    metrics = run_ijb_template_benchmark(
        args.embeddings,
        args.metadata,
        args.pairs,
        fars=_parse_float_list(args.fars),
        file_col=args.file_col,
        template_col=args.template_col,
        media_col=args.media_col,
        left_col=args.left_col,
        right_col=args.right_col,
        label_col=args.label_col,
    )
    _write_metrics(metrics, output_path=args.output)


if __name__ == "__main__":
    main()
