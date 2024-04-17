#!/usr/bin/env python

import os
import sys
from typing import Iterator
from pathlib import Path

from urllib.parse import urlparse
from subprocess import run

OUT_DIR = os.getenv("OUT_DIR", "/tmp")


def model_urls(model_list: Path) -> Iterator[str]:
    for line in model_list.read_text().split("\n"):
        line = line.strip()
        if line.startswith("#"):
            continue
        elif len(line) == 0:
            continue
        yield line


if __name__ == "__main__":
    for i, url in enumerate(model_urls(Path("models.txt"))):
        path = urlparse(url).path
        model_id = path.split("/")[-1].split(".")[0]
        print(*["ptexport", url, f"{OUT_DIR}/{model_id}.pt"], file=sys.stderr)
        run(["ptexport", url, f"{OUT_DIR}/{model_id}.pt"], check=True)
