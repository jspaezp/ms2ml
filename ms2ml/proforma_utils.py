from importlib import resources
from typing import Dict

from appdirs import AppDirs
from loguru import logger
from pyteomics import proforma
from pyteomics.mass import Unimod
from pyteomics.proforma import UnimodResolver

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version


__version__ = version("ms2ml")
my_appdirs = AppDirs(appname="ms2ml", version=__version__)

with resources.path("ms2ml.unimod", "unimod.xml") as f:
    LOCAL_UNIMOD_PATH = "file://" + str(f)


def set_local_unimod():
    proforma.set_unimod_path(LOCAL_UNIMOD_PATH)
    if proforma.obo_cache is not None:
        proforma.obo_cache.cache_path = my_appdirs.user_cache_dir
        proforma.obo_cache.enabled = True
    else:
        logger.warning(
            "Could not set the cache directory. "
            "This will impact the performance of "
            "parsing non-unimod modifications "
            "(try installing psimod if you are using python version 3.9 or higher)"
        )


set_local_unimod()


class LocalUnimodResolver(UnimodResolver):
    def load_database(self):
        unimod = Unimod(LOCAL_UNIMOD_PATH)
        return unimod


class MemoizedUnimodResolver:
    """Memoized Unimod resolver.

    Uses the UnimodResolver from pyteomics but stores the results in a dictionary
    internally, so when the same query is made, the result is MUCH FASTER.

    It is intended to be used as a singleton and be called directly from the
    class methods, instead of generating an instance of the object.

    Examples:
        >>> MemoizedUnimodResolver.resolve("Phospho")
        {'composition': Composition({'H': 1, 'O': 3, 'P': 1}),
        'name': 'Phospho', 'id': 21, 'mass': 79.966331, 'provider': 'unimod'}
        >>> MemoizedUnimodResolver.mod_id_mass(21)
        79.966331
    """

    _cache: Dict[str, Dict] = {
        "Carbamidomethyl": {
            "id": 4,
            "mass": 57.021464,
            "mono_mass": 57.021464,
            "provider": "unimod",
        },
        "Oxidation": {
            "id": 35,
            "mass": 15.994915,
            "mono_mass": 15.994915,
            "provider": "unimod",
        },
    }
    _id_cache: Dict[str, Dict] = {str(v["id"]): v for k, v in _cache.items()}

    _solver = None

    @classmethod
    def solver(cls):
        if cls._solver is None:
            logger.debug("Initializing UnimodResolver")
            cls._solver = LocalUnimodResolver()

        return cls._solver

    @classmethod
    def resolve(cls, mod_id):
        if isinstance(mod_id, str):
            mod_id_name = mod_id
        elif isinstance(mod_id, int):
            mod_id_name = str(mod_id)
        else:
            raise ValueError(f"Invalid mod_id: {mod_id}")

        if mod_id not in cls._cache:
            logger.debug(f"Resolving {mod_id}")
            try:
                cls._cache[mod_id_name] = cls.solver().resolve(mod_id, strict=False)
            except KeyError:
                logger.warning(
                    f"Could not resolve {mod_id_name} try"
                    " assigning an alias to it in the config"
                )
                raise
            logger.debug(f"Resolved to {cls._cache[mod_id_name]}")

        return cls._cache[mod_id_name]

    @classmethod
    def mod_id_mass(cls, mod_id: int = 21) -> float:
        if str(mod_id) in cls._id_cache:
            return cls._id_cache[str(mod_id)]["mono_mass"]

        logger.debug(f"Resolving mod_id: {mod_id}")
        mod = cls.solver().database[mod_id]
        logger.debug(f"Resolved to {mod}")
        return mod["mono_mass"]
