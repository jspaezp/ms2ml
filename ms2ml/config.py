"""Defines the constants that are used in the rest of the project.

Such as the masses of aminoacids, supported modifications, length of the encodings,
maximum length supported, labels and order of the encoded ions ...

Greatly inspired/copied from:
https://github.com/kusterlab/prosit/blob/master/prosit/constants.py
"""


import string
import warnings
from dataclasses import dataclass

from .annotation_classes import AnnotatedIon
from .types import MassError

# TODO cosnsider wether we want a more strict enforcement
# of the requirement for configurations to be defined
# and passed


@dataclass
class Config:
    """General class to set and store the configuration of the project.

    Ideally every project will make one AND ONLY ONE of these.
    The provided defaults are meant to be reasonable for most projects
    but can be changed as needed.

    All annotation and endoding functionality should require one of this objects
    to work.

    Parameters:
        g_tolerances:
        g_tolerance_units:
        g_isotopes:
        peptide_max_length:
        precursor_charges:
        fragment_positions:
        ion_series:
        ion_charges:
        ion_neutral_losses:
        ion_encoding_nesting:
        ion_naming_convention:
        mod_ambiguity_threshold:
        mod_fixed_mods:

    Examples:
        >>> config = Config()
        >>> config
        Config(g_tolerances=(50, 50), ... mod_ambiguity_threshold=0.99,
        mod_fixed_mods=('[U:4]@C',))
        >>> config.fragment_labels
        ['y1^1', 'y1^2', ... 'b30^2']
    """

    # General Configs
    # = each tuple object is meant to define an MS level
    # = For instance, in tolerances, the first tolerance is for MS1
    # = and the second for MS2
    g_tolerances: tuple[float, ...] = (50, 50)
    g_tolerance_units: tuple[MassError, ...] = ("ppm", "ppm")

    # = Number of isotopes to check for each ion, 0 means that only
    # = the monoisotopic peak us used
    g_isotopes: tuple[int, ...] = (0, 0)

    # Peptide Configs
    peptide_max_length: int = 30

    # Precursor Configs
    precursor_charges: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

    # Fragment Configs
    # TODO consider wether to unify this the same way the general configs
    # = use positions within a tuple to denote MS levels
    fragment_positions: tuple[int, ...] = tuple(range(1, 31))

    # Ion config
    ion_series: str = "yb"
    ion_charges: tuple[int, ...] = (1, 2)
    # = Neutral Losss Configs
    ion_neutral_losses: tuple[str, ...] = ()

    ion_encoding_nesting: tuple[str, ...] = (
        "ion_charges",
        "fragment_positions",
        "ion_series",
    )
    ion_naming_convention: str = "{ion_series}{fragment_positions}^{ion_charges}"

    # Modifications
    mod_ambiguity_threshold: float = 0.99
    mod_fixed_mods: tuple[str] = ("[U:4]@C",)

    # Encoding Configs
    encoding_aa_order = (
        ["n_term"] + list(string.ascii_uppercase) + ["c_term", "__missing__"]
    )

    # TODO consider wether to remove the square brackets
    encoding_mod_order = [
        None,
        "[U:4]",
        "[U:21]",
        "[U:35]",
        "[U:737]",
        "[U:7]",
        "[U:1]",
        "__unknown1__",
        "__unknown2__",
        "__unknown3__",
    ]
    encoding_mod_alias = {
        "Phospho": "[U:21]",
        "TMT6plex": "[U:737]",
        "TMT10plex": "[U:737]",
        "Deamidation": "[U:7]",
        "Acetyl": "[U:1]",
        "Oxidation": "[U:35]",
        "Hydroxilation": "[U:35]",
        "Trimethyl": "[U:37]",
    }

    @property
    def fragment_labels(self) -> list[str]:
        """
        Returns a list of the labels that are used to encode the fragments.

        Examples:
            >>> config = Config()
            >>> config.fragment_labels
            ['y1^1', 'y1^2', ... 'b30^2']
        """
        if not hasattr(self, "_fragment_labels_cache"):
            self._fragment_labels_cache = self._framgnet_labels()

        return self._fragment_labels_cache

    @property
    def num_fragment_embeddings(self) -> int:
        return len(self.fragment_labels)

    def _framgnet_labels(self) -> list[str]:
        labels: list[str] = []
        for field in self.ion_encoding_nesting:
            _labels = labels
            labels = []
            for elem in getattr(self, field):
                mapper = _PermissiveMapper(**{field: str(elem)})
                if len(_labels) == 0:
                    x = self.ion_naming_convention.format_map(mapper)
                    labels.append(x)
                else:
                    for x in _labels:
                        x = x.format_map(mapper)
                        labels.append(x)
        return labels

    def ion_labeller(self, ion: AnnotatedIon) -> str:
        """Provided an ion, returns the label for that ion.

        Examples:
            >>> ion = AnnotatedIon(mass=123.2, charge=2, position=3, ion_series="z")
            >>> config = Config()
            >>> config.ion_labeller(ion)
            'z3^2'
        """
        return self.ion_naming_convention.format_map(ion.asdict())

    def validate(self):
        raise NotImplementedError


class _PermissiveMapper(dict):
    """This class is a helper to allow the bypass of missing keys in the generation of.

    the ion labels.
    """

    def __missing__(self, key):
        return "{" + key + "}"


def get_default_config() -> Config:
    warnings.warn("Using default config. Consider creating your own.", UserWarning)
    return Config()


class ConfigNotSetError(Exception):
    """Raised when a config is not set but is required."""


if __name__ == "__main__":
    foo = Config()
    print(foo)
    print(foo.fragment_labels)
