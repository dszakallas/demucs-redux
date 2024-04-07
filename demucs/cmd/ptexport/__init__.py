import argparse
from pathlib import Path
import inspect

import warnings

from ...htdemucs import HTDemucs
from ...states import set_state
import torch


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Input model file.")
    parser.add_argument("dst", help="Output file to save the exported TorchScript model.")
    parser.add_argument("--repo", type=str, help="Folder containing all pre-trained models.")
    return parser


def load_model(path_or_package, strict=False):
    """Load a model from the given serialized model, either given as a dict (already loaded)
    or a path to a file on disk."""
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            package = torch.load(path, "cpu")
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]
    set_state(model, state)
    return model


def main(opts=None):
    # Example usage
    parser = get_parser()
    args = parser.parse_args(opts)

    model = load_model(args.src)

    if isinstance(model, HTDemucs):
        model.segment = float(model.segment)

    torch.jit.save(torch.jit.script(model), args.dst)

    # model = get_model(args.name, args.repo)
    # htdemucs = []
    # if isinstance(model, BagOfModels):
    #     for model in model.models:
    #         if isinstance(model, HTDemucs):
    #             htdemucs.append(model)

    # if isinstance(model, HTDemucs):
    #     htdemucs.append(model)

    # for htd in htdemucs:
    #     htd.segment = float(htd.segment)
