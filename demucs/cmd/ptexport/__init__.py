import argparse
from os import makedirs
from pathlib import Path

from ...states import load_model
import torch


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Input weights URL/path.")
    parser.add_argument("dst", help="Output path to write.")
    return parser


def main():
    # Example usage
    parser = get_parser()
    args = parser.parse_args()

    pkg = torch.hub.load_state_dict_from_url(
        args.src, map_location="cpu", check_hash=True
    )  # type: ignore

    m = load_model(pkg)

    args.dst = Path(args.dst)
    makedirs(args.dst.parent, exist_ok=True)
    torch.jit.save(torch.jit.script(m), args.dst)
