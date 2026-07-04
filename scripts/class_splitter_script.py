from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "valid", "test")

def list_class_dirs(source_dir: Path) -> list[Path]:
    skip = set(SPLIT_NAMES) | {"__pycache__"}
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_dir() and path.name.lower() not in skip and not path.name.startswith(".")
    )

def list_files(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

def split_files(
    files: list[Path],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[list[Path], list[Path], list[Path]]:
    if not files:
        return [], [], []

    shuffled = files.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    valid_count = int(total * valid_ratio)

    train_files = shuffled[:train_count]
    valid_files = shuffled[train_count : train_count + valid_count]
    test_files = shuffled[train_count + valid_count :]

    return train_files, valid_files, test_files

def transfer_files(files: list[Path], destination_dir: Path, copy: bool) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    transferred = 0

    for file_path in files:
        target = destination_dir / file_path.name
        if target.exists():
            raise FileExistsError(f"Target already exists: {target}")

        if copy:
            shutil.copy2(file_path, target)
        else:
            shutil.move(str(file_path), str(target))
        transferred += 1

    return transferred


def balance_train_classes(train_dir: Path, seed: int = 42) -> None:
    if not train_dir.is_dir():
        return

    class_files: dict[str, list[Path]] = {}
    for class_dir in list_class_dirs(train_dir):
        files = list_files(class_dir)
        if files:
            class_files[class_dir.name] = files

    if len(class_files) < 2:
        return

    counts = {name: len(files) for name, files in class_files.items()}
    target_count = min(counts.values())

    if len(set(counts.values())) == 1:
        print(f"Train set already balanced ({target_count} images per class).")
        return

    print("-" * 60)
    print(f"Balancing train set to {target_count} images per class...")

    rng = random.Random(seed)
    for class_name, files in sorted(class_files.items()):
        if len(files) <= target_count:
            print(f"  {class_name}: {len(files)} (unchanged)")
            continue

        selected = files.copy()
        rng.shuffle(selected)
        keep = selected[:target_count]
        remove = selected[target_count:]
        keep_names = {path.name for path in keep}

        for file_path in files:
            if file_path.name not in keep_names:
                file_path.unlink()

        print(f"  {class_name}: {len(files)} -> {target_count} (removed {len(remove)})")

def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy: bool = False,
    balance_train: bool = True,
) -> None:
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum:.4f}")

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    class_dirs = list_class_dirs(source_dir)
    if not class_dirs:
        raise ValueError(f"No class folders found in: {source_dir}")

    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Split ratios -> train: {train_ratio}, valid: {valid_ratio}, test: {test_ratio}")
    print(f"Mode: {'copy' if copy else 'move'}")
    print("-" * 60)

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = list_files(class_dir)

        if not files:
            print(f"[skip] {class_name}: no image files found")
            continue

        train_files, valid_files, test_files = split_files(
            files, train_ratio, valid_ratio, test_ratio
        )

        counts = {}
        for split_name, files_for_split in zip(
            SPLIT_NAMES, (train_files, valid_files, test_files)
        ):
            dest = output_dir / split_name / class_name
            counts[split_name] = transfer_files(files_for_split, dest, copy=copy)

        shutil.rmtree(class_dir)
        print(
            f"{class_name}: {len(files)} total -> "
            f"train={counts['train']}, valid={counts['valid']}, test={counts['test']} "
            f"(removed from class_splitter)"
        )

    if balance_train:
        balance_train_classes(output_dir / "train", seed=seed)


def parse_args() -> argparse.Namespace:
    project_root = project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Split each class folder in class_splitter/ into train, valid, "
            "and test sets under datasets/detection/, then remove the class "
            "folder from class_splitter/."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=project_root / "class_splitter",
        help="Folder containing class subfolders (default: class_splitter/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=project_root / "datasets" / "detection",
        help="Output root; creates train/valid/test/<class>/ (default: datasets/detection/)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of files for training (default: 0.7)",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="Fraction of files for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of files for testing (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them (class folder is still removed after save)",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Skip balancing train classes to equal image counts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dataset(
        source_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy=args.copy,
        balance_train=not args.no_balance,
    )

if __name__ == "__main__":
    main()