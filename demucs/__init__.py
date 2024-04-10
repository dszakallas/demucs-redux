from typing import Union

from .demucs import Demucs
from .hdemucs import HDemucs
from .htdemucs import HTDemucs

Model = Union[Demucs, HDemucs, HTDemucs]
