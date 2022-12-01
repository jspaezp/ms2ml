from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from pyteomics.proforma import ProForma, ProFormaError, parse, to_proforma

from ms2ml.proforma_utils import set_local_unimod

from .annotation_classes import AnnotatedIon
from .config import Config, get_default_config
from .constants import ION_OFFSET
from .isoform_utils import get_mod_isoforms, get_mod_possible
from .utils.class_utils import lazy
from .utils.mz_utils import mz

set_local_unimod()


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
        self.config = config
        self.extras = extras
        super().__init__(sequence, properties)

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
        cls, seq, config: Config | None = None, extras=None
    ) -> Peptide:
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

        try:
            sequence, properties = parse(cls.pre_parse_mods(seq, config))
        except ProFormaError:
            raise ValueError(f"Could not parse sequence: {seq}")
        return Peptide(sequence, properties, config, extras)

    @classmethod
    def from_sequence(cls, *args, **kwargs):
        """Alias for from_proforma_seq."""
        return cls.from_proforma_seq(*args, **kwargs)

    @classmethod
    def from_ProForma(cls, proforma: ProForma, config, extras=None) -> Peptide:
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

    def to_proforma(self) -> str:
        """Converts the peptide to a string following the proforma specifications.

        Examples:
            >>> p = Peptide.from_sequence("AMC")
            >>> p.to_proforma()
            '<[UNIMOD:4]@C>AMC'
        """

        return to_proforma(self.sequence, **self.properties)

    def to_massdiff_seq(self) -> str:
        """Converts the peptide to a string following the massdiff specifications.

        Examples:
            >>> p = Peptide.from_sequence("AMC")
            >>> p.to_massdiff_seq()
            'AMC[+57.021464]'
            >>> p = Peptide.from_sequence("[U:1]-AMC")
            >>> p.to_massdiff_seq()
            'A[+42.010565]MC[+57.021464]'
        """
        aas = np.array(self.config.encoding_aa_order)

        aavector = aas[self.aa_to_vector()]
        nterm_mask = aavector == "n_term"
        cterm_mask = aavector == "c_term"

        if self.config.mod_mode == "delta_mass":
            diffs = self.mod_seq
            tmp = np.array([float(x) if x is not None else 0 for x in diffs])

        elif self.config.mod_mode == "unimod":
            diffs = self.config.mod_masses
            modvector = self.mod_to_vector()
            tmp = diffs[modvector]
        else:
            raise NotImplementedError

        nterm = tmp[nterm_mask]
        cterm = tmp[cterm_mask]
        massdiffs = tmp[(nterm_mask + cterm_mask).astype(bool) == False]  # noqa: E712
        aas = aavector[(nterm_mask + cterm_mask).astype(bool) == False]  # noqa: E712

        massdiffs[0] += nterm.sum()
        massdiffs[-1] += cterm.sum()
        massdiff_srt = [
            ("[" + ("+" if massdiff > 0 else "-") + f"{abs(massdiff):06f}" + "]")
            if massdiff
            else ""
            for massdiff in massdiffs
        ]

        out = "".join([f"{aa}{massdiff}" for aa, massdiff in zip(aas, massdiff_srt)])

        return out

    @property
    def stripped_sequence(self):
        """Returns the stripped sequence of the peptide.

        Examples:
            >>> p = Peptide.from_sequence("PEPTIDE")
            >>> p.stripped_sequence
            'PEPTIDE'
            >>> p = Peptide.from_sequence("PEPTIDE/2")
            >>> p.stripped_sequence
            'PEPTIDE'
            >>> p = Peptide.from_sequence("PEPM[Oxidation]ASDA")
            >>> p.stripped_sequence
            'PEPMASDA'
        """
        out = "".join(x[0] for x in self.sequence)
        return out

    def validate(self) -> bool:
        """Validates the built peptide.

        Not yet implemented."""
        raise NotImplementedError

    @property
    def ProForma(self):
        return ProForma(self.sequence, self.properties)

    @lazy
    def mass(self) -> float:
        """Calculates the mass of a peptide

        Examples:
            >>> p = Peptide.from_sequence("MYPEPTIDE")
            >>> p.mass
            1093.46377747225
        """
        curr_mass = 0.0

        # n and c temrini are accounted for ... assuming they are used
        curr_mass += np.einsum("ij,j->", self.aa_to_onehot(), self.config.aa_masses)
        curr_mass += np.einsum("ij,j->", self.mod_to_onehot(), self.config.mod_masses)

        # labile mods
        # unlocalized
        return curr_mass

    @property
    def mass_pyteomics(self) -> float:
        """Returns the mass of the peptide."""
        mass = super().mass
        return mass

    @property
    def charge(self) -> int:
        return self.charge_state.charge

    @property
    def fragment_masses(self) -> list:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"Peptide.from_sequence('{self.ProForma}')"

    def __getitem__(self, i):
        """Slices a peptide

        Examples:
            >>> pep = Peptide._sample()
            >>> pep.stripped_sequence
            'PEPTIDEPINK'
            >>> foo = pep[:2]
            >>> foo.stripped_sequence
            'PE'
        """
        out = self.ProForma[i]
        return self.from_ProForma(out, config=self.config, extras=self.extras)

    def __len__(self) -> int:
        """Returns the length of the peptide sequence."""
        return len(self.sequence)

    @staticmethod
    def _sample() -> Peptide:
        config = Config()
        return Peptide.from_sequence("[U:1]-PEPT[U:21]IDEPINK", config=config)

    @lazy
    def _position_masses(self) -> NDArray[np.float32]:
        """Calculates the masses of each termini and aminoacid.

        It is used as a basis to calculate the mass of ion series.

        Examples:
            >>> p = Peptide._sample()
            >>> len(p._position_masses)
            13
            >>> p._position_masses
            array([ 43.01839 ,  97.052765, 129.04259 ,  97.052765, 181.014   ,
            113.08406 , 115.02694 , 129.04259 ,  97.052765, 113.08406 ,
            114.04293 , 128.09496 ,  17.002739], dtype=float32)
        """

        curr_mass = np.einsum("ij,j->i", self.aa_to_onehot(), self.config.aa_masses)
        curr_mass += np.einsum("ij,j->i", self.mod_to_onehot(), self.config.mod_masses)

        return curr_mass.astype(np.float32)

    @lazy
    def _forward(self):
        """Calculates the masses of all fragments of the peptide in the forward.

        direction.

        For instance, for the peptide "AMC" the forward fragments are:
            [mass_of_n, mass_of_nA, mass_of_nAM, mass_of_nAMCc]
            Where n and c are the n- and c-terminal masses.

        The forward series is used as a basis for the a, b anc c ion series
        The backwards series is used as a basis for the x, y and z ion series
        """
        return self._position_masses.cumsum()

    @property
    def _backward(self):
        """See the forward property."""
        return self._position_masses[::-1].cumsum()

    def ion_series(self, ion_type: str, charge: int) -> NDArray[np.float32]:
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

    def annotated_ion_series(self, ion_type: str, charge: int) -> list[AnnotatedIon]:
        """Returns a list of annotated ions.

        Examples:
            >>> p = Peptide.from_sequence("AMC")
            >>> p.annotated_ion_series("b", 1)
            [AnnotatedIon(mass=72.044945, charge=1,
            position=1, ion_series='b', intensity=0, neutral_loss=None),
            AnnotatedIon(mass=203.08542,
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
        for i, m in enumerate(masses):
            elem = AnnotatedIon(
                mass=m, charge=charge, position=i + 1, ion_series=ion_type
            )
            tmp.append(elem)

        if ion_type not in self.ion_series_cache:
            self.ion_series_cache[ion_type] = {}

        self.ion_series_cache[ion_type][charge] = tmp
        return tmp

    @lazy
    def ion_dict(self) -> dict[str, AnnotatedIon]:
        """Returns a dictionary of all ion series for the peptide.

        Raises:
            ValueError: If peptide does not have a charge state.

        Examples:
            >>> p = Peptide.from_sequence("PEPPINK/2")
            >>> p.ion_dict
            {'y1^1': AnnotatedIon(mass=147.11334, ...
            charge=2, position=6, ion_series='b', intensity=0, neutral_loss=None)}
            >>> p.ion_dict["y5^1"].mass
            568.34537
        """
        if not hasattr(self, "charge") or self.charge is None:
            raise ValueError("Peptide charge is not set")

        possible_charges = [x for x in self.config.ion_charges if x <= self.charge]
        tmp = {}
        for ion_type in self.config.ion_series:
            for charge in possible_charges:
                curr_series = self.annotated_ion_series(ion_type, charge)
                for x in curr_series:
                    tmp[x.label(self.config.ion_naming_convention)] = x

        return tmp

    # TODO implement a setter for the charge ...

    @lazy
    def theoretical_ion_labels(self) -> np.ndarray:
        labels = list(self.ion_dict.keys())
        return np.array(labels)

    @lazy
    def theoretical_ion_masses(self) -> np.ndarray:
        ions = self.ion_dict.values()
        masses = [x.mass for x in ions]
        masses = np.array(masses)
        return masses

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
            array([[1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0]], dtype=int32)
        """
        vector = self.mod_to_vector()

        out = np.zeros(
            (len(vector), len(self.config.encoding_mod_order)), dtype=np.int32
        )
        for i, mod in enumerate(vector):
            out[i, mod] = 1

        return out

    @staticmethod
    def decode_onehot(
        config: Config, seq_onehot: np.ndarray, mod_onehot: np.ndarray | None = None
    ) -> Peptide:
        """Decodes a one-hot encoded vector into a peptide sequence.

        Examples:
            >>> config = Config()
            >>> foo = Peptide.from_sequence("AMC", config=config)
            >>> onehot = foo.aa_to_onehot()
            >>> mod_onehot = foo.mod_to_onehot()
            >>> Peptide.decode_onehot(config, onehot, mod_onehot)
            Peptide([('A', None), ('M', None),
             ('C', [UnimodModification('4', None, None)])],
             {'n_term': None, 'c_term': None, 'unlocalized_modifications': [],
              'labile_modifications': [],
              'fixed_modifications':
                  [ModificationRule(UnimodModification('4', None, None), ['C'])],
              'intervals': [], 'isotopes': [], 'group_ids': [], 'charge_state': None})

        """
        seq = np.argmax(seq_onehot, axis=1)
        mod = np.argmax(mod_onehot, axis=1) if mod_onehot is not None else None

        return Peptide.decode_vector(config, seq, mod)

    @staticmethod
    def decode_vector(
        config: Config,
        seq: np.ndarray,
        mod: np.ndarray | None,
        charge: int | None = None,
    ) -> Peptide:
        """Decodes a one-hot encoded vector into a peptide sequence.

        Examples:
            >>> config = Config()
            >>> foo = Peptide.from_sequence("AMC", config)
            >>> foo.aa_to_vector()
            array([ 0,  1, 13,  3, 27])
            >>> foo.mod_to_vector()  # Default config has carbamido
            array([0, 0, 0, 1, 0])
            >>> Peptide.decode_vector(
            ...     foo.config, foo.aa_to_vector(), foo.mod_to_vector()
            ... )
            Peptide([('A', None), ('M', None),
             ('C', [UnimodModification('4', None, None)])],
             {'n_term': None, 'c_term': None, 'unlocalized_modifications': [],
              'labile_modifications': [],
              'fixed_modifications':
                  [ModificationRule(UnimodModification('4', None, None), ['C'])],
              'intervals': [], 'isotopes': [], 'group_ids': [], 'charge_state': None})
        """

        def special_handling(seq, mod):
            if seq == "n_term":
                if mod:
                    return f"{mod}-"
                else:
                    return ""
            if seq == "c_term":
                if mod:
                    return f"-{mod}"
                else:
                    return ""
            return seq + mod

        if mod is None:
            mod_adds = [None] * len(seq)
        else:
            mod_adds = [config.encoding_mod_order[x] for x in mod]

        mod_adds = [x if x is not None else "" for x in mod_adds]
        seq_adds = [config.encoding_aa_order[x] for x in seq]

        seq_out = "".join([special_handling(x, y) for x, y in zip(seq_adds, mod_adds)])
        if charge:
            seq_out = f"{seq_out}/{charge}"
        return Peptide.from_proforma_seq(config=config, seq=seq_out)

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
        enc_order = self.config.encoding_aa_order_mapping

        # TODO: make so it warns once when an aminoacid
        # is used that is not in the encoding_aa_order
        # set(aas).difference(self.config.encoding_aa_order)

        vector = np.array([enc_order[x] for x in aas if x in enc_order], dtype=int)
        return vector

    @lazy
    def mod_seq(self):
        """Returns the sequence of modifications mathhing the aminoacid positions

        Examples:
            >>> foo = Peptide.from_sequence("AMC")
            >>> foo.mod_seq
            [None, None, None, '[U:4]', None]
        """
        mods = [x[1] for x in self]
        vector = []

        for x in mods:
            if hasattr(x, "__iter__"):
                x = list(set(x))
                if len(x) > 1:
                    error_msg = "Multiple modifications on the"
                    error_msg += " same aminoacid are not supported"
                    error_msg += f" got:({x})"

                    # TODO consider is more informative messages are required
                    raise ValueError(error_msg)
                x = x[0]

            vector.append(x)
        return vector

    def mod_to_vector(self):
        """Converts modifications to vectors

        Converts the modifications peptide sequence to a vector encoding.

        Examples:
            >>> foo = Peptide.from_sequence("AMC")  # Implicit Carbamido.
            >>> foo.mod_to_vector()
            array([0, 0, 0, 1, 0])
        """

        vector = self.mod_seq
        order_mapping = self.config.encoding_mod_order_mapping
        vector = [order_mapping[x] for x in vector]
        return np.array(vector)

    @classmethod
    def from_vector(cls, aa_vector: list[int], mod_vector, config: Config):
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

    def __iter__(self) -> Iterator[tuple[str, list[str] | None]]:
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
        yield from self.__iter_base

    @staticmethod
    def from_iter(it, config: Config):
        """Creates a peptide from an iterator of (aa, mod) tuples.

        Examples:
            >>> foo = Peptide.from_iter(
            ...     [
            ...         ("n_term", None),
            ...         ("A", None),
            ...         ("M", None),
            ...         ("C", ["[U:4]"]),
            ...         ("c_term", None),
            ...     ],
            ...     config=Config(),
            ... )
            >>> foo.to_proforma()
            '<[UNIMOD:4]@C>AMC[UNIMOD:4]'
            >>> foo = Peptide._sample()
            >>> foo.to_proforma()
            '[UNIMOD:1]-PEPT[UNIMOD:21]IDEPINK'
            >>> elems = [x for x in foo]
            >>> foo = Peptide.from_iter(elems, config=Config())
            >>> foo.to_proforma()
            '[UNIMOD:1]-PEPT[UNIMOD:21]IDEPINK'
        """
        # TODO fix this so it is not redundant generating fixed+variable mod
        # notation on carbamidomethyl

        seqs = []
        for aa, mod in it:
            if mod is not None:
                mod = "".join(mod)
                if aa == "n_term":
                    mod = mod + "-"
                if mod == "c_term":
                    mod = "-" + mod
            else:
                mod = ""
            if aa in ("n_term", "c_term"):
                aa = ""

            aa += mod
            seqs.append(aa)

        return Peptide.from_proforma_seq("".join(seqs), config=config)

    @lazy
    def __iter_base(self) -> list[tuple[str, list[str] | None]]:
        iter_out = []
        fixed_mods = self.properties["fixed_modifications"]
        resolve_mod_list = self.config._resolve_mod_list
        mod_rules = {}
        for mod in fixed_mods:
            for target in mod.targets:
                if target not in mod_rules:
                    mod_rules[target] = []
                mod_rules[target].append(mod.modification_tag)

        iter_out.append(tuple(["n_term", resolve_mod_list(self.properties["n_term"])]))

        for aa, mods in self.sequence:
            if aa in mod_rules:
                if mods is None:
                    mods = []
                mods.extend(mod_rules[aa])

            iter_out.append(tuple([aa, resolve_mod_list(mods)]))

        iter_out.append(tuple(["c_term", resolve_mod_list(self.properties["c_term"])]))
        return iter_out

    def get_mod_isoforms(self) -> list[Peptide]:
        """Returns a list of possible modifications isoforms of a peptide.

        Examples:
            >>> foo = Peptide.from_sequence("AM[UNIMOD:35]AMK")
            >>> out = foo.get_mod_isoforms()
            >>> sorted([x.to_proforma() for x in out])
            ['AMAM[UNIMOD:35]K', 'AM[UNIMOD:35]AMK']
        """
        iters = get_mod_isoforms(self.__iter_base, self.config.mod_variable_mods)
        out = [self.from_iter(x, self.config) for x in iters]
        return out

    def get_variable_possible_mods(self):
        """Returns a list of possible modifications for each aminoacid.

        Examples:
            >>> foo = Peptide.from_sequence("AMAMK")
            >>> out = foo.get_variable_possible_mods()
            >>> sorted([x.to_proforma() for x in out])
            ['AMAMK', 'AMAM[UNIMOD:35]K', 'AM[UNIMOD:35]AMK',
             'AM[UNIMOD:35]AM[UNIMOD:35]K']

        """
        iters = get_mod_possible(self.__iter_base, self.config.mod_variable_mods)
        out = [self.from_iter(x, self.config) for x in iters]
        return out
