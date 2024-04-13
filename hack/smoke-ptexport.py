#!/usr/bin/env python

import sys
from typing import Iterator
from pathlib import Path

from subprocess import run


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
        print(*["ptexport", url, f"/tmp/{i}.pt"], file=sys.stderr)
        run(["ptexport", url, f"/tmp/{i}.pt"], check=True)
