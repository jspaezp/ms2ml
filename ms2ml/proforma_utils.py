from typing import Dict

from pyteomics.proforma import UnimodResolver


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
    """

    _cache: Dict[str, Dict] = {}
    _solver = None

    @classmethod
    def resolve(cls, mod_id):
        if cls._solver is None:
            cls._solver = UnimodResolver()

        if isinstance(mod_id, str):
            mod_id_name = mod_id
        elif isinstance(mod_id, int):
            mod_id_name = str(mod_id)
        else:
            raise ValueError(f"Invalid mod_id: {mod_id}")

        if mod_id not in cls._cache:
            cls._cache[mod_id_name] = cls._solver.resolve(mod_id, strict=False)

        return cls._cache[mod_id_name]
