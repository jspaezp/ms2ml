import os
import pathlib
from typing import Literal, Union

MassError = Literal["ppm", "da"]
PathLike = Union[str, pathlib.Path, os.PathLike]
