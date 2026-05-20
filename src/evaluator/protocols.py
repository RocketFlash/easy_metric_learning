from pathlib import Path

import pandas as pd


def _lfw_image_name(person, index, image_ext=".jpg", name_template=None):
    if name_template is None:
        name_template = "{person}/{person}_{index:04d}{ext}"
    return name_template.format(person=person, index=int(index), ext=image_ext)


def read_lfw_style_pairs(
    pairs_path,
    image_ext=".jpg",
    name_template=None,
    skip_header=True,
):
    pairs = []
    with Path(pairs_path).open() as file:
        lines = [line.strip() for line in file if line.strip()]

    if skip_header and lines:
        first_parts = lines[0].split()
        if len(first_parts) in {1, 2} and all(part.isdigit() for part in first_parts):
            lines = lines[1:]

    for line in lines:
        parts = line.split()
        if len(parts) == 3:
            person, index1, index2 = parts
            pairs.append(
                {
                    "file1": _lfw_image_name(
                        person, index1, image_ext=image_ext, name_template=name_template
                    ),
                    "file2": _lfw_image_name(
                        person, index2, image_ext=image_ext, name_template=name_template
                    ),
                    "is_same": True,
                }
            )
        elif len(parts) == 4:
            person1, index1, person2, index2 = parts
            pairs.append(
                {
                    "file1": _lfw_image_name(
                        person1,
                        index1,
                        image_ext=image_ext,
                        name_template=name_template,
                    ),
                    "file2": _lfw_image_name(
                        person2,
                        index2,
                        image_ext=image_ext,
                        name_template=name_template,
                    ),
                    "is_same": False,
                }
            )
        else:
            raise ValueError(f"Unsupported LFW-style pair line: {line}")

    return pd.DataFrame(pairs)


def read_whitespace_pairs(pairs_path, file1_col=0, file2_col=1, label_col=2):
    pairs = []
    with Path(pairs_path).open() as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            max_col = max(file1_col, file2_col, label_col)
            if len(parts) <= max_col:
                raise ValueError(f"Unsupported pair line: {line}")
            label = parts[label_col].lower()
            pairs.append(
                {
                    "file1": parts[file1_col],
                    "file2": parts[file2_col],
                    "is_same": label in {"1", "true", "same", "yes", "positive"},
                }
            )
    return pd.DataFrame(pairs)


def read_pair_protocol(pairs_path, protocol="csv", **kwargs):
    protocol = (protocol or "csv").lower()
    if protocol == "csv":
        return pd.read_csv(pairs_path)
    if protocol in {"lfw", "agedb"}:
        return read_lfw_style_pairs(pairs_path, **kwargs)
    if protocol in {"cfp", "whitespace"}:
        return read_whitespace_pairs(pairs_path)
    raise ValueError(f"Unknown pair protocol: {protocol}")
