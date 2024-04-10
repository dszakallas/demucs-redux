import argparse
from os import makedirs
from pathlib import Path

from ...pretrained import get_repo
import torch


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", nargs="*", help="Input model weights name/hash. If omitted exports all models."
    )
    parser.add_argument(
        "-o",
        "--out",
        default="exported",
        help="Output dir to save the exported model(s). Default: exported.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        help=f"Folder containing pre-trained models weights. If omitted remote repo is used.",
    )
    return parser


def main():
    # Example usage
    parser = get_parser()
    args = parser.parse_args()

    repo = None
    if args.repo:
        repo = Path(args.repo)

    repo = get_repo(repo)

    models = args.src

    if not models:
        models = repo.list_model()

    makedirs(args.out, exist_ok=True)

    for model in models:
        model = repo.get_model(model)
        torch.jit.save(torch.jit.script(model), args.out / f"{model}.pt")
