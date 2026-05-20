import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluator.protocols import read_lfw_style_pairs, read_whitespace_pairs

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_table(path, names):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


def _maybe_add_sizes(df, root_dir, include_sizes):
    if not include_sizes:
        return df

    import imagesize

    widths = []
    heights = []
    for file_name in df["file_name"].tolist():
        width, height = imagesize.get(Path(root_dir) / file_name)
        widths.append(width)
        heights.append(height)
    df["width"] = widths
    df["height"] = heights
    return df


def _label_from_path(file_name):
    parent = Path(str(file_name)).parent
    if str(parent) in {"", "."}:
        return Path(str(file_name)).stem
    return parent.name


def _dataset_info_from_files(
    root_dir,
    file_names,
    labels=None,
    include_sizes=False,
    strict=False,
):
    root_dir = Path(root_dir)
    labels = {} if labels is None else labels
    seen = set()
    rows = []
    missing = []
    for file_name in file_names:
        file_name = str(file_name)
        if file_name in seen:
            continue
        seen.add(file_name)
        if strict and not (root_dir / file_name).is_file():
            missing.append(file_name)
            continue
        rows.append(
            {
                "file_name": file_name,
                "label": str(labels.get(file_name, _label_from_path(file_name))),
            }
        )

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} referenced images are missing: {preview}"
        )

    df = pd.DataFrame(rows, columns=["file_name", "label"])
    return _maybe_add_sizes(df, root_dir, include_sizes=include_sizes)


def scan_image_dataset(root_dir, include_sizes=False):
    root_dir = Path(root_dir)
    file_names = []
    for image_path in sorted(root_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            file_names.append(image_path.relative_to(root_dir).as_posix())
    return _dataset_info_from_files(
        root_dir,
        file_names,
        include_sizes=include_sizes,
        strict=False,
    )


def prepare_lfw_like(
    root_dir,
    pairs_path,
    save_dir,
    image_ext=".jpg",
    name_template=None,
    include_sizes=False,
    strict=False,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    pairs = read_lfw_style_pairs(
        pairs_path,
        image_ext=image_ext,
        name_template=name_template,
    )
    file_names = pd.unique(pairs[["file1", "file2"]].values.ravel("K"))
    dataset_info = _dataset_info_from_files(
        root_dir,
        file_names,
        include_sizes=include_sizes,
        strict=strict,
    )
    dataset_info_path = save_dir / "dataset_info.csv"
    pairs_path_out = save_dir / "pairs.csv"
    dataset_info.to_csv(dataset_info_path, index=False)
    pairs.to_csv(pairs_path_out, index=False)
    return {
        "dataset_info": dataset_info_path,
        "pairs": pairs_path_out,
        "n_images": len(dataset_info),
        "n_pairs": len(pairs),
    }


def prepare_whitespace_verification(
    root_dir,
    pairs_path,
    save_dir,
    include_sizes=False,
    strict=False,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    pairs = read_whitespace_pairs(pairs_path)
    file_names = pd.unique(pairs[["file1", "file2"]].values.ravel("K"))
    dataset_info = _dataset_info_from_files(
        root_dir,
        file_names,
        include_sizes=include_sizes,
        strict=strict,
    )
    dataset_info_path = save_dir / "dataset_info.csv"
    pairs_path_out = save_dir / "pairs.csv"
    dataset_info.to_csv(dataset_info_path, index=False)
    pairs.to_csv(pairs_path_out, index=False)
    return {
        "dataset_info": dataset_info_path,
        "pairs": pairs_path_out,
        "n_images": len(dataset_info),
        "n_pairs": len(pairs),
    }


def prepare_ijb(
    root_dir,
    metadata_path,
    pairs_path,
    save_dir,
    file_col="file_name",
    template_col="template_id",
    media_col="media_id",
    left_col="template1",
    right_col="template2",
    label_col="is_same",
    include_sizes=False,
    strict=False,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    metadata = _read_table(metadata_path, [file_col, template_col, media_col])
    pairs = _read_table(pairs_path, [left_col, right_col, label_col])
    labels = {
        str(row[file_col]): str(row[template_col]) for _, row in metadata.iterrows()
    }
    dataset_info = _dataset_info_from_files(
        root_dir,
        metadata[file_col].astype(str).tolist(),
        labels=labels,
        include_sizes=include_sizes,
        strict=strict,
    )

    dataset_info_path = save_dir / "dataset_info.csv"
    metadata_path_out = save_dir / "ijb_metadata.csv"
    pairs_path_out = save_dir / "ijb_pairs.csv"
    dataset_info.to_csv(dataset_info_path, index=False)
    metadata.to_csv(metadata_path_out, index=False)
    pairs.to_csv(pairs_path_out, index=False)
    return {
        "dataset_info": dataset_info_path,
        "metadata": metadata_path_out,
        "pairs": pairs_path_out,
        "n_images": len(dataset_info),
        "n_pairs": len(pairs),
        "n_templates": int(metadata[template_col].nunique()),
    }


def prepare_megaface(
    save_dir,
    probe_root=None,
    gallery_root=None,
    distractor_root=None,
    include_sizes=False,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    roots = {
        "probe": probe_root,
        "gallery": gallery_root,
        "distractor": distractor_root,
    }
    manifest = {}
    for split, root_dir in roots.items():
        if root_dir is None:
            continue
        dataset_info = scan_image_dataset(root_dir, include_sizes=include_sizes)
        output_path = save_dir / f"{split}_dataset_info.csv"
        dataset_info.to_csv(output_path, index=False)
        manifest[split] = {
            "root_dir": str(root_dir),
            "dataset_info": str(output_path),
            "n_images": len(dataset_info),
            "n_labels": int(dataset_info["label"].nunique()),
        }

    manifest_path = save_dir / "megaface_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    manifest["manifest"] = str(manifest_path)
    return manifest


def _print_result(result):
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
    }
    print(json.dumps(serializable, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare normalized face benchmark protocol files"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    lfw = subparsers.add_parser("lfw", help="Prepare LFW/AgeDB-style pairs.txt")
    lfw.add_argument("--root-dir", required=True)
    lfw.add_argument("--pairs", required=True)
    lfw.add_argument("--save-dir", required=True)
    lfw.add_argument("--image-ext", default=".jpg")
    lfw.add_argument("--name-template")
    lfw.add_argument("--include-sizes", action="store_true")
    lfw.add_argument("--strict", action="store_true")

    whitespace = subparsers.add_parser(
        "whitespace", help="Prepare CFP-style whitespace pair files"
    )
    whitespace.add_argument("--root-dir", required=True)
    whitespace.add_argument("--pairs", required=True)
    whitespace.add_argument("--save-dir", required=True)
    whitespace.add_argument("--include-sizes", action="store_true")
    whitespace.add_argument("--strict", action="store_true")

    ijb = subparsers.add_parser("ijb", help="Prepare IJB template protocol files")
    ijb.add_argument("--root-dir", required=True)
    ijb.add_argument("--metadata", required=True)
    ijb.add_argument("--pairs", required=True)
    ijb.add_argument("--save-dir", required=True)
    ijb.add_argument("--include-sizes", action="store_true")
    ijb.add_argument("--strict", action="store_true")

    megaface = subparsers.add_parser("megaface", help="Prepare MegaFace split CSVs")
    megaface.add_argument("--save-dir", required=True)
    megaface.add_argument("--probe-root")
    megaface.add_argument("--gallery-root")
    megaface.add_argument("--distractor-root")
    megaface.add_argument("--include-sizes", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "lfw":
        result = prepare_lfw_like(
            args.root_dir,
            args.pairs,
            args.save_dir,
            image_ext=args.image_ext,
            name_template=args.name_template,
            include_sizes=args.include_sizes,
            strict=args.strict,
        )
    elif args.command == "whitespace":
        result = prepare_whitespace_verification(
            args.root_dir,
            args.pairs,
            args.save_dir,
            include_sizes=args.include_sizes,
            strict=args.strict,
        )
    elif args.command == "ijb":
        result = prepare_ijb(
            args.root_dir,
            args.metadata,
            args.pairs,
            args.save_dir,
            include_sizes=args.include_sizes,
            strict=args.strict,
        )
    else:
        result = prepare_megaface(
            args.save_dir,
            probe_root=args.probe_root,
            gallery_root=args.gallery_root,
            distractor_root=args.distractor_root,
            include_sizes=args.include_sizes,
        )
    _print_result(result)


if __name__ == "__main__":
    main()
