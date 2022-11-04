from importlib import resources
from typing import Dict

from loguru import logger
from pyteomics.mass import Unimod
from pyteomics.proforma import UnimodResolver


class LocalUnimodResolver(UnimodResolver):
    def load_database(self):
        with resources.path("ms2ml.unimod", "unimod.xml") as f:
            unimod = Unimod("file://" + str(f))
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
            "mono_mass": 47.021464,
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
            # TODO move this to real logging
            logger.debug(f"Resolving {mod_id}")
            cls._cache[mod_id_name] = cls.solver().resolve(mod_id, strict=False)
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
