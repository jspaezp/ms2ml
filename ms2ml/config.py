"""Provides a place to define the project's configuration.

Defines the constants that are used in the rest of the project.

Such as the masses of aminoacids, supported modifications, length of the encodings,
maximum length supported, labels and order of the encoded ions ...
"""

from __future__ import annotations

import string
import sys
import warnings
from dataclasses import asdict, dataclass, field

if sys.version_info <= (3, 11):
    import tomli as tomllib
else:
    import tomllib

import numpy as np
import tomli_w

from .annotation_classes import AnnotatedIon
from .constants import C_TERMINUS, N_TERMINUS, STD_AA_MASS
from .proforma_utils import MemoizedUnimodResolver
from .type_defs import MassError, ModModes
from .utils.class_utils import lazy

# TODO cosnsider wether we want a more strict enforcement
# of the requirement for configurations to be defined
# and passed


def _default_mod_aliases():
    out = {
        "Phospho": "[UNIMOD:21]",
        "TMT6plex": "[UNIMOD:737]",
        "TMT10plex": "[UNIMOD:737]",
        "Deamidation": "[UNIMOD:7]",
        "Acetyl": "[UNIMOD:1]",
        "Oxidation": "[UNIMOD:35]",
        "Hydroxilation": "[UNIMOD:35]",
        "Trimethyl": "[UNIMOD:37]",
    }
    return out


def _default_mod_order():
    encoding_mod_order = tuple(
        [
            None,
            "[UNIMOD:4]",
            "[UNIMOD:21]",
            "[UNIMOD:35]",
            "[UNIMOD:7]",
            "[UNIMOD:1]",
            "__unknown1__",
        ]
    )
    return encoding_mod_order


def _default_var_mods():
    out = {
        "[UNIMOD:35]": [
            "M",
        ],
        "[UNIMOD:21]": [
            "S",
            "T",
            "Y",
        ],
    }
    return out


# TODO consider making this apydantic model


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
            A tuple of floats,
            where each float is the tolerance of that corresponding ms level.
            For instance (10, 20) means that the tolerance for ms1 is 10, 20 for ms2.
        g_tolerance_units:
            A tuple of strings, that denote what tolerance unit to use for each ms
            level. For instance ("ppm", "Da") means that the tolerance for ms1 is
            in ppm, and for ms2 in Da.
        g_isotopes:
        peptide_length_range:
            A tuple of ints, where the first int is the minimum length of peptides.
        precursor_charges:
            A tuple of ints, where each int is a possible precursor charge.
        fragment_positions:
            A tuple of ints, where each int is a possible fragment position.
        ion_series:
            A string of characters, where each character is a possible ion series.
            An example,a dn the default is ('by')
        ion_charges:
            A tuple of ints, where each int is a possible ion charge.
        ion_neutral_losses:
        ion_encoding_nesting:
        ion_naming_convention:
        mod_ambiguity_threshold:
        mod_fixed_mods:

    Examples:
        >>> config = Config()
        >>> config
        Config(g_tolerances=(50, 50), ...)
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
    peptide_length_range: tuple[int, int] = (5, 30)
    peptide_mz_range: tuple[float, float] = (200, 20000)

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
    mod_mode: ModModes = "unimod"
    mod_ambiguity_threshold: float = 0.99
    mod_fixed_mods: tuple[str] = ("[UNIMOD:4]@C",)
    mod_variable_mods: dict[str, tuple[str]] = field(default_factory=_default_var_mods)

    # Encoding Configs
    encoding_aa_order: tuple[str] = tuple(
        ["n_term", *list(string.ascii_uppercase), "c_term", "__missing__"]
    )

    encoding_mod_order: tuple[str | None, ...] = field(
        default_factory=_default_mod_order
    )
    encoding_mod_alias: dict[str, str] = field(default_factory=_default_mod_aliases)

    encoding_spec_bin_start: float = field(repr=False, default=0.0)
    encoding_spec_bin_end: float = field(repr=False, default=2000.0)
    encoding_spec_bin_binsize: float | None = field(repr=False, default=0.1)
    encoding_spec_bin_n_bins: int | None = field(repr=False, default=None)
    encoding_spec_bin_relative: bool = field(repr=False, default=False)
    encoding_spec_bin_offset: float = field(repr=False, default=0.0)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Validates the configuration."""
        self._validate_mods()
        self._validate_ions()

    def _validate_ions(self):
        if not isinstance(self.ion_series, str):
            raise ValueError(
                f"Ion series should be a string, got {self.ion_series!r}"
                f"\n For example 'yb'"
            )

        for ion in self.ion_series:
            if ion not in "abcxyz":
                raise ValueError(
                    f"Ion series should only contain a, b, c, x, y or z. Got {ion!r}"
                )

        if not isinstance(tuple(self.ion_charges), tuple):
            raise ValueError(
                f"Ion charges should be a tuple, got {self.ion_charges!r}"
                f"\n For example (1, 2)"
            )

        for charge in self.ion_charges:
            if not isinstance(charge, int):
                raise ValueError(
                    f"Ion charges should only contain integers. Got {charge!r}"
                )

    def _validate_mods(self):
        if not isinstance(self.mod_fixed_mods, tuple):
            raise ValueError(
                f"Fixed mods should be a tuple, got {self.mod_fixed_mods!r}"
                f"\n For example ('[UNIMOD:4]@C',)"
            )

        if not isinstance(self.mod_variable_mods, dict):
            raise ValueError(
                f"Variable mods should be a dict, got {self.mod_variable_mods!r}"
                f"\n For example {{'[UNIMOD:4]': ['C']}}"
            )

    @lazy
    def fragment_labels(self) -> list[str]:
        """
        Returns a list of the labels that are used to encode the fragments.

        Examples:
            >>> config = Config()
            >>> config.fragment_labels
            ['y1^1', 'y1^2', ... 'b30^2']
        """
        labels: list[str] = []
        for ion_field in self.ion_encoding_nesting:
            _labels = labels
            labels = []
            for elem in getattr(self, ion_field):
                mapper = _PermissiveMapper(**{ion_field: str(elem)})
                if len(_labels) == 0:
                    x = self.ion_naming_convention.format_map(mapper)
                    labels.append(x)
                else:
                    for x in _labels:
                        x = x.format_map(mapper)
                        labels.append(x)
        return labels

    @property
    def num_fragment_embeddings(self) -> int:
        return len(self.fragment_labels)

    def ion_labeller(self, ion: AnnotatedIon) -> str:
        """Labels an ion

        Provided an ion, returns the label for that ion.

        Examples:
            >>> ion = AnnotatedIon(mass=123.2, charge=2, position=3, ion_series="z")
            >>> config = Config()
            >>> config.ion_labeller(ion)
            'z3^2'
        """
        return self.ion_naming_convention.format_map(ion.asdict())

    @lazy
    def aa_masses(self):
        masses = STD_AA_MASS.copy()
        masses["n_term"] = N_TERMINUS
        masses["c_term"] = C_TERMINUS
        aa_mass = [masses.get(aa, 0) for aa in self.encoding_aa_order]
        return np.array(aa_mass)

    @lazy
    def mod_masses(self):
        mod_mass = []
        for mod in self.encoding_mod_order:
            if mod is None:
                mod_mass.append(0.0)

            elif "[UNIMOD:" in mod:
                mod_id = int(mod.split(":")[1].split("]")[0])
                mod_mass.append(MemoizedUnimodResolver.mod_id_mass(mod_id))

            elif "unknown" in mod:
                mod_mass.append(0.0)

            else:
                raise ValueError(f"Unknown mod {mod}")

        return np.array(mod_mass)

    @lazy
    def encoding_mod_order_mapping(self) -> dict[str | None, int]:
        return {mod: i for i, mod in enumerate(self.encoding_mod_order)}

    @lazy
    def encoding_aa_order_mapping(self) -> dict[str | None, int]:
        return {aa: i for i, aa in enumerate(self.encoding_aa_order)}

    @lazy
    def fixed_mods_dict(self) -> dict[str, set[str, float]]:
        """Returns a dictionary of fixed mods.

        The keys are the aminoacids and the values are the modifications.

        Examples:
            >>> config = Config()
            >>> config.fixed_mods_dict
            {'C': {'[UNIMOD:4]'}}
            >>> config = Config(mod_fixed_mods=("[+22.2222]@C",))
            >>> tmp = config.fixed_mods_dict

            # {'C': {'[+22.2222]', 22.2222}}

            >>> config = Config(mod_fixed_mods=("[+22.0]@C",))
            >>> tmp = config.fixed_mods_dict

            # {'C': {'[+22.0]', 22.0}}
            >>> {k: sorted([str(w) for w in v]) for k, v in tmp.items()}
            {'C': ['22.0', '[+22.0]']}
        """
        out = {}
        for mod in self.mod_fixed_mods:
            mod_key = mod[mod.find("@") + 1 :]
            mod_value = mod[: mod.find("@")]
            if len(mod_key) > 1:
                raise NotImplementedError(
                    f"Fixed mods with more than one aminoacid are not supported. Got {mod}"
                )
            if mod_key not in out:
                out[mod_key] = []

            out[mod_key].append(mod_value)
            try:
                mod_value = float(mod_value.replace("[", "").replace("]", ""))
            except ValueError:
                pass

            out[mod_key].append(mod_value)

        out = {k: set(v) for k, v in out.items()}
        return out

    def _resolve_mod_list(self, x):
        if isinstance(x, list):
            return [self._resolve_mod_list(y) for y in x]

        if x is None:
            return None

        if self.mod_mode == "unimod":
            return self._resolve_mod_list_unimod(x)
        elif self.mod_mode == "delta_mass":
            return self._resolve_mod_list_delta_mass(x)
        else:
            raise NotImplementedError(
                f"Mod mode {self.mod_mode} not implemented. Trying to resolve {x}"
            )

    def _resolve_mod_list_delta_mass(self, x):
        val = x.value
        if isinstance(val, float):
            return val

        if ":" in val and "Obs:" not in val:
            raise ValueError(f"Delta mass mods should not contain a colon. Got {val}")
        if "Obs:" in val:
            val = val.replace("Obs:", "")

        if not (val.startswith("+") or val.startswith("-")):
            val = "+" + val

        return val

    def _resolve_mod_list_unimod(self, x):
        """Resolves the names in a list of modifications to unimod Ids."""

        if not hasattr(self, "__mod_resol_cache"):
            setattr(self, "__mod_resol_cache", {None: None})

        cache = getattr(self, "__mod_resol_cache")

        if x.value in cache:
            return cache[x.value]

        # TODO consider moving this logic to the mod resolver class
        def _internal(y):
            # Mass modififications (only deinfed by mass, such as open mods)
            # or underfined aliases .... do not have a name

            # Write a better error message if the mod is not found
            if y.value in self.encoding_mod_alias:
                modname = y.value
            else:  # noqa
                if not hasattr(y, "name"):
                    modname = str(y)
                else:
                    modname = y.name

            if modname in self.encoding_mod_alias:
                solved_name = self.encoding_mod_alias[modname]
            else:
                solved_name = MemoizedUnimodResolver.resolve(modname)["id"]
                solved_name = f"[UNIMOD:{solved_name!s}]"

            return solved_name

        cache[x.value] = _internal(x)
        return cache[x.value]

    def asdict(self):
        """Returns a dictionary representation of the config."""
        return asdict(self)

    """
    # def validate(self):
    # test_all_modification_aliases_map():
    for v in constants.MOD_PEPTIDE_ALIASES.values():
        if v == "":
            continue
        constants.MODIFICATION[v]

    # Validate ion naming positions ...
    # Test all variable mods are in the mod order
    # Test all mod aliases in mod order
    # Test all aas in mod aliases in aa_order
    """
    #     raise NotImplementedError

    def to_toml(self, path: str):
        """Writes the config to a toml file."""

        out = {}
        for k, v in self.asdict().items():
            if isinstance(v, tuple):
                v = [x if x is not None else "__NONE__" for x in v]
            elif v is None:
                v = "__NONE__"
            out[k] = v
        with open(path, "wb") as f:
            tomli_w.dump(out, f)

    @staticmethod
    def from_toml(path: str):
        """Loads a config from a toml file."""
        with open(path, "rb") as f:
            config = tomllib.load(f)

        out_config = {}
        for k, v in config.items():
            if isinstance(v, list):
                v = tuple(x if x != "__NONE__" else None for x in v)
            elif v == "__NONE__":
                v = None
            out_config[k] = v

        return Config(**out_config)

    def from_comet(self, path: str, *args, **kwargs):
        """Loads a config from a comet params file."""
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
