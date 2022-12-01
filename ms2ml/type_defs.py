import os
import pathlib
from typing import Literal, Union

MassError = Literal["ppm", "da"]
ModModes = Literal["unimod", "delta_mass"]
PathLike = Union[str, pathlib.Path, os.PathLike]
