from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from pyteomics.proforma import ProForma, parse, to_proforma

from .annotation_classes import AnnotatedIon
from .config import Config, get_default_config
from .constants import ION_OFFSET, OH, PROTON, WATER
from .proforma_utils import MemoizedUnimodResolver
from .utils import mz


class Peptide(ProForma):
    """Represents a peptide sequence with modifications.

    Examples:
        >>> p = Peptide.from_sequence("MYPEPTIDE")
        >>> p.mass
        1093.46377747225
        >>> p = Peptide.from_sequence("MYPEPTIDE/2")
        >>> p.charge
        2
        >>> p = Peptide.from_sequence("J")
        >>> p.mass
        131.09462866083
        >>> p = Peptide.from_sequence("X")
        >>> p.mass
        18.010564683699997
        >>> p = Peptide.from_sequence("Z")

        Note that it does not throw an error ... it should ...

        >>> p.mass
        18.010564683699997
    """

    def __init__(self, sequence, properties, config, extras) -> None:
        self._config = config
        self.extras = extras
        super().__init__(sequence, properties)

    @property
    def config(self):
        if self._config is None:
            raise

        return self._config

    @staticmethod
    def pre_parse_mods(seq, config) -> str:
        """Parse the modifications in the sequence."""
        if config.mod_fixed_mods:
            fixed_mods = config.mod_fixed_mods
        else:
            fixed_mods = []

        seq = (
            "".join([f"<{x}>" for x in fixed_mods if x[x.index("@") + 1] in seq]) + seq
        )
        return seq

    @classmethod
    def from_proforma_seq(
        cls, seq, config: Optional[Config] = None, extras=None
    ) -> "Peptide":
        """Generates a peptide from a proforma sequence.

        Examples:
            >>> p = Peptide.from_proforma_seq("PEPTIDE")
            >>> p.mass
            799.3599640267099
            >>> p = Peptide.from_proforma_seq("PEPTIDE", extras={"test": 1})
            >>> p.extras
            {'test': 1}
        """
        if config is None:
            config = get_default_config()

        sequence, properties = parse(cls.pre_parse_mods(seq, config))
        return Peptide(sequence, properties, config, extras)

    @classmethod
    def from_sequence(cls, *args, **kwargs):
        """Alias for from_proforma_seq."""
        return cls.from_proforma_seq(*args, **kwargs)

    @classmethod
    def from_ProForma(cls, proforma: ProForma, config, extras=None) -> "Peptide":
        """Creates a peptide from a pyteomics.proforma.ProForma object.

        Examples:
            >>> from pyteomics.proforma import ProForma, parse
            >>> config = Config()
            >>> seq, props = parse("PEPTIDE")
            >>> p = ProForma(seq, props)
            >>> p = Peptide.from_ProForma(p, config)
            >>> p.mass
            799.3599
        """

        return cls(proforma.sequence, proforma.properties, config, extras=extras)

    def to_proforma(self):
        """Converts the peptide to a string following the proforma specifications.

        Examples:
            >>> p = Peptide.from_sequence("AMC")
            >>> p.to_proforma()
            '<[UNIMOD:4]@C>AMC'
        """

        return to_proforma(self.sequence, **self.properties)

    def validate(self) -> bool:
        """Validates the built peptide.

        Not yet implemented."""
        raise NotImplementedError

    @property
    def ProForma(self):
        return ProForma(self.sequence, self.properties)

    @property
    def mass(self) -> float:
        """Returns the mass of the peptide."""
        # TODO see if this can be optimized
        mass = super().mass
        return mass

    @property
    def charge(self) -> int:
        return self.charge_state.charge

    @property
    def fragment_masses(self) -> list:
        if self._fragment_masses is None:
            raise NotImplementedError

    def __str__(self) -> str:
        return f"Peptide.from_sequence('{self.ProForma}')"

    def __getitem__(self, i):
        out = self.ProForma[i]
        return self.from_ProForma(out, config=self.config, extras=self.extras)

    def __len__(self) -> int:
        """Returns the length of the peptide sequence."""
        return len(self.sequence)

    @property
    def _position_masses(self) -> np.float32:
        """Calculates the masses of each termini and aminoacid.

        It is used as a basis to calculate the mass of ion series.
        """
        if not hasattr(self, "_position_masses_cache"):
            out = []
            for i in range(0, len(self) + 1):
                curr_mass = self[max(0, i - 1) : i].mass
                out.append(curr_mass)

            # A placeholder for the mass of the C-terminus
            curr_mass = self[0:0].mass
            out.append(curr_mass)

            out2 = []
            for i, curr_mass in enumerate(out):
                if i > len(self):
                    curr_mass -= PROTON
                elif i == 0:
                    curr_mass -= OH
                else:
                    curr_mass -= WATER

                out2.append(curr_mass)

            self._position_masses_cache = np.float32(out2)

        return self._position_masses_cache

    @property
    def _forward(self):
        """Calculates the masses of all fragments of the peptide in the forward.

        direction.

        For instance, for the peptide "AMC" the forward fragments are:
            [mass_of_n, mass_of_nA, mass_of_nAM, mass_of_nAMCc]
            Where n and c are the n- and c-terminal masses.

        The forward series is used as a basis for the a, b anc c ion series
        The backwards series is used as a basis for the x, y and z ion series
        """
        if not hasattr(self, "_forward_cache"):
            self._forward_cache = self._position_masses.cumsum()

        return self._forward_cache

    @property
    def _backward(self):
        """See the forward property."""
        if not hasattr(self, "_backward_cache"):
            self._backward_cache = self._position_masses[::-1].cumsum()

        return self._backward_cache

    def ion_series(self, ion_type: str, charge: int) -> np.float32:
        """Calculates all the masses of an ion type.

        Calculates the masses of all fragments of the peptide for a given ion type.
        and charge.

        Examples:
            >>> p = Peptide.from_sequence("AMC")
            >>> p.ion_series("a", 1)
            array([ 44.05003, 175.0905 ], dtype=float32)
        """
        if ion_type in ("a", "b", "c"):
            cumsum = self._forward
        elif ion_type in ("x", "y", "z"):
            cumsum = self._backward
        else:
            # TODO implement Impr ions
            # I(mmonium) and p(recursor) are easy to implement
            # M(middle/intermediate) and r(reporter) is a bit more complicated
            raise ValueError(f"Invalid ion type: {ion_type}")

        cumsum = cumsum[:-2]
        out = mz(cumsum + ION_OFFSET[ion_type], charge=charge)
        out = out[1:]
        return out

    def annotated_ion_series(self, ion_type: str, charge: int) -> List[AnnotatedIon]:
        """Returns a list of annotated ions.

        Examples:
            >>> p = Peptide.from_sequence("AMC")
            >>> p.annotated_ion_series("b", 1)
            [AnnotatedIon(mass=array(72.044945, dtype=float32), charge=1,
            position=1, ion_series='b', intensity=0, neutral_loss=None),
            AnnotatedIon(mass=array(203.08542, dtype=float32),
            charge=1, position=2, ion_series='b', intensity=0, neutral_loss=None)]
        """
        # TODO: Add neutral loss
        if hasattr(self, "ion_series_cache"):
            if ion_type in self.ion_series_cache:
                if charge in self.ion_series_cache[ion_type]:
                    return self.ion_series_cache[ion_type][charge]
        else:
            self.ion_series_cache = {}

        masses = self.ion_series(ion_type, charge)
        tmp = []
        for i, m in enumerate(np.nditer(masses)):
            elem = AnnotatedIon(
                mass=m, charge=charge, position=i + 1, ion_series=ion_type
            )
            tmp.append(elem)

        if ion_type not in self.ion_series_cache:
            self.ion_series_cache[ion_type] = {}

        self.ion_series_cache[ion_type][charge] = tmp
        return tmp

    @property
    def ion_series_dict(self) -> Dict[str, AnnotatedIon]:
        """Returns a dictionary of all ion series for the peptide.

        Raises:
            ValueError: If peptide does not have a charge state.

        Examples:
            >>> p = Peptide.from_sequence("PEPPINK/2")
            >>> p.ion_series_dict
            {'y1^1': AnnotatedIon(mass=array(147.11334, dtype=float32), ...
            charge=2, position=6, ion_series='b', intensity=0, neutral_loss=None)}
        """
        if not hasattr(self, "_ion_series_dict"):
            if self.charge is None:
                raise ValueError("Peptide charge is not set")

            possible_charges = [x for x in self.config.ion_charges if x <= self.charge]
            tmp = {}
            for ion_type in self.config.ion_series:
                for charge in possible_charges:
                    curr_series = self.annotated_ion_series(ion_type, charge)
                    for x in curr_series:
                        tmp[x.label(self.config.ion_naming_convention)] = x

            self._ion_series_dict = tmp

        return self._ion_series_dict

    # TODO implement a setter for the charge ...

    @property
    def theoretical_ion_labels(self) -> np.ndarray:
        if not hasattr(self, "_theoretical_ion_labels"):
            labels = list(self.ion_series_dict.keys())
            self._theoretical_ion_labels = np.array(labels)

        return self._theoretical_ion_labels

    @property
    def theoretical_ion_masses(self) -> np.ndarray:
        if not hasattr(self, "_theoretical_ion_masses"):
            ions = self.ion_series_dict.values()
            masses = [x.mass for x in ions]
            masses = np.array(masses)
            self._theoretical_ion_masses = masses

        return self._theoretical_ion_masses

    def aa_to_onehot(self):
        """Converts the peptide sequence to a one-hot encoding.

        Returns a binary array of shape:
            (nterm + peptide_length + cterm, len(self.config.encoding_aa_order))

        The positions along the second axis are the one-hot encoding of the
        aminoacid, matching the order of the encoding_aa_order argument in the config.

        For instance, if the peptide was "ABA" and the encoding_aa_order was
        ["n_term", "A", "B", "c_term"], the vector would be:

            [
                [1, 0, 0, 0 ,0],
                [0, 1, 0, 0 ,0],
                [0, 0, 1, 0 ,0],
                [0, 1, 0, 0 ,0],
                [0, 0, 0, 0 ,1]
            ]

        Examples:
            >>> foo = Peptide.from_sequence("AMC")
            >>> foo.aa_to_onehot()
            array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0]], dtype=int32)
        """
        vector = self.aa_to_vector()

        out = np.zeros(
            (len(vector), len(self.config.encoding_aa_order)), dtype=np.int32
        )
        for i, aa in enumerate(vector):
            out[i, aa] = 1

        return out

    def mod_to_onehot(self):
        """Converts the peptide sequence to a one-hot encoding.

        Returns a binary array of shape:
            (nterm + peptide_length + cterm, len(self.config.encoding_mod_order))

        The positions along the second axis are the one-hot encoding of the
        aminoacid, matching the order of the encoding_mod_order argument in the config.

        For instance, if the peptide was "AC" and the encoding_mod_order was
        [None, "[U:4]"], being [U:4] carbamidomethyl, the vector would be:

            [
                [1, 0],
                [1, 0],
                [0, 1],
                [1, 0],
            ]

        Note that the 3rd position shows up as modified due to the implicit
        carbamidomethylation of C.

        Examples:
            >>> foo = Peptide.from_sequence("AMC")
            >>> foo.mod_to_onehot()
            array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
        """
        vector = self.mod_to_vector()

        out = np.zeros(
            (len(vector), len(self.config.encoding_mod_order)), dtype=np.int32
        )
        for i, mod in enumerate(vector):
            out[i, mod] = 1

        return out

    def aa_to_count(self):
        """Converts the peptide sequence to a one-hot encoding.

        Returns a binary array of shape:
            (nterm + peptide_length + cterm, len(self.config.encoding_aa_order))

        The positions along the second axis are the one-hot encoding of the
        aminoacid, matching the order of the encoding_aa_order argument in the config.

        For instance, if the peptide was "ABA" and the encoding_aa_order was
        ["n_term", "A", "B", "C", "c_term"], the vector would be:

            [1, 2, 1, 0 ,1],

        Examples:
            >>> foo = Peptide.from_sequence("AAMC")
            >>> foo.aa_to_count()
            array([1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0])
        """
        return self.aa_to_onehot().sum(axis=0)

    def mod_to_count(self):
        return self.mod_to_onehot().sum(axis=0)

    def aa_to_vector(self):
        """Converts the peptide sequence to a vector encoding.

        Returns a binary array of length:
            (nterm + peptide_length + cterm)

        The number in every positions corresponds to the matching the order
        of the encoding_aa_order argument in the config.

        For instance, if the peptide was "ABA" and the encoding_aa_order was
        ["n_term", "A", "B", "c_term"], the vector would be:

            [0, 1, 2, 1, 3]

        Examples:
            >>> foo = Peptide.from_sequence("AMC")
            >>> foo.aa_to_vector()
            array([ 0,  1,  13,  3, 27])
        """
        aas = [x[0] for x in self]

        array_lst = []
        for x in aas:
            if x in self.config.encoding_aa_order:
                array_lst.append(self.config.encoding_aa_order.index(x))
            else:
                # TODO: make so it warns once when an aminoacid
                # is used that is not in the encoding_aa_order
                # set(aas).difference(self.config.encoding_aa_order)
                pass

        vector = np.array(array_lst)
        return vector

    def mod_to_vector(self):
        """Converts modifications to vectors

        Converts the modifications peptide sequence to a vector encoding.

        Examples:
            >>> foo = Peptide.from_sequence("AMC")  # Implicit Carbamido.
            >>> foo.mod_to_vector()
            array([0, 0, 0, 1, 0])
        """
        mods = [x[1] for x in self]
        vector = []

        for x in mods:
            if hasattr(x, "__iter__"):
                if len(x) > 1:
                    error_msg = (
                        "Multiple modifications on the same aminoacid are not supported"
                    )
                    error_msg = f"{error_msg} got:({x})"

                    # TODO consider is more informative messages are required
                    raise ValueError(error_msg)
                x = x[0]
            vector.append(self.config.encoding_mod_order.index(x))

        return np.array(vector)

    @classmethod
    def from_vector(cls, aa_vector: List[int], mod_vector, config: Config):
        """Converts vectors back to peptides.

        Examples:
            >>> foo = Peptide.from_vector([0, 1, 13, 3, 27], [0, 0, 0, 1, 0], Config())
            >>> foo.to_proforma()
            '<[UNIMOD:4]@C>AMC'
        """

        sequence = ""
        for aa, mod in zip(aa_vector, mod_vector):
            mod = config.encoding_mod_order[mod]
            aa = config.encoding_aa_order[aa]
            if aa in ("n_term", "c_term"):
                aa = ""

            # TODO check if this will handle static mods on termini correctly
            # Proforma does not seem to handle static mods on termini
            if mod is not None:
                tmp_mod = f"{mod}@{aa}"
                if tmp_mod not in config.mod_fixed_mods:
                    aa += f"{mod}"

            sequence += aa

        peptide = Peptide.from_proforma_seq(sequence, config)
        return peptide

    def __iter__(self) -> Iterator[Tuple[str, Optional[List[str]]]]:
        """Iterates over the peptide sequence.

        Yields:
            (aa, mod) tuples

        Examples:
            >>> foo = Peptide.from_sequence("AMC")
            >>> [x for x in foo]
            [('n_term', None), ('A', None), ('M', None),
            ('C', ['[U:4]']), ('c_term', None)]
            >>> foo = Peptide.from_sequence("AMS[Phospho]C")
            >>> [x for x in foo]
            [('n_term', None), ('A', None), ('M', None),
            ('S', ['[U:21]']), ('C', ['[U:4]']), ('c_term', None)]
        """

        def resolve_mod_list(x):
            """Resolves the names in a list of modifications to unimod Ids."""
            if x is None:
                return None
            else:
                out = []
                for y in x:
                    if y.name in self.config.encoding_mod_alias:
                        solved_name = self.config.encoding_mod_alias[y.name]
                    else:
                        solved_name = MemoizedUnimodResolver.resolve(y.name)["id"]
                        solved_name = f"[U:{str(solved_name)}]"

                    out.append(solved_name)

            return out

        fixed_mods = self.properties["fixed_modifications"]
        mod_rules = {}
        for mod in fixed_mods:
            for target in mod.targets:
                if target not in mod_rules:
                    mod_rules[target] = []
                mod_rules[target].append(mod.modification_tag)

        yield tuple(["n_term", resolve_mod_list(self.properties["n_term"])])

        for aa, mods in self.sequence:
            if aa in mod_rules:
                if mods is None:
                    mods = []
                mods.extend(mod_rules[aa])

            yield tuple([aa, resolve_mod_list(mods)])

        yield tuple(["c_term", resolve_mod_list(self.properties["c_term"])])
