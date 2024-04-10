# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Loading pretrained models.
"""

import logging
from pathlib import Path
from typing import Union, Dict

from ..demucs import Demucs
from ..hdemucs import HDemucs
from ..htdemucs import HTDemucs
from .repo import (
    RemoteRepo,
    LocalRepo,
    ModelOnlyRepo,
)  # noqa

logger = logging.getLogger(__name__)
ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/"
REMOTE_ROOT = Path(__file__).parent / "remote"

SOURCES = ["drums", "bass", "other", "vocals"]
DEFAULT_MODEL = "htdemucs"

Model = Union[Demucs, HDemucs, HTDemucs]


def _parse_remote_files(remote_file_list) -> Dict[str, str]:
    root: str = ""
    models: Dict[str, str] = {}
    for line in remote_file_list.read_text().split("\n"):
        line = line.strip()
        if line.startswith("#"):
            continue
        elif len(line) == 0:
            continue
        elif line.startswith("root:"):
            root = line.split(":", 1)[1].strip()
        else:
            sig = line.split("-", 1)[0]
            assert sig not in models
            models[sig] = ROOT_URL + root + line
    return models


def get_repo(repo: Union[Path, None] = None) -> ModelOnlyRepo:
    """`name` must be a bag of models name or a pretrained signature
    from the remote AWS model repo or the specified local repo if `repo` is not None.
    """
    if repo is None:
        models = _parse_remote_files(REMOTE_ROOT / "files.txt")
        return RemoteRepo(models)

    if not repo.is_dir():
        raise AssertionError(f"{repo} must exist and be a directory.")
    return LocalRepo(repo)
