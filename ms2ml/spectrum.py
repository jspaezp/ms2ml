"""Implements classes to represent spectra.

This module implements classes to represent spectra and their
annotations (when they have any).

There are broadly two types of spectra:
1. General Spectra
2. Annotated Spectra
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, overload

import numpy as np
from numpy.typing import NDArray

from .annotation_classes import AnnotatedIon, RetentionTime
from .config import Config, get_default_config
from .peptide import Peptide
from .utils.class_utils import clear_lazy_cache, lazy
from .utils.mz_utils import annotate_peaks, get_tolerance, mz, stack_mz_pairs

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from spectrum_utils.plot import spectrum as plotspec
    from spectrum_utils.spectrum import MsmsSpectrum as sus_MsmsSpectrum

try:
    from matplotlib import pyplot as plt
    from spectrum_utils.plot import spectrum as plotspec
    from spectrum_utils.spectrum import MsmsSpectrum as sus_MsmsSpectrum
except ImportError:
    sus_MsmsSpectrum = None
    plotspec = None
    plt = None


@dataclass
class Spectrum:
    """Class to store the spectrum information.

    Examples:
        >>> spectrum = Spectrum(
        ...     mz=np.array([1000.0, 1500.0, 2000.0]),
        ...     intensity=np.array([1.0, 2.0, 3.0]),
        ...     precursor_mz=1000.0,
        ...     precursor_charge=2,
        ...     ms_level=2,
        ...     extras={"EXTRAS": ["extra1", "extra2"]},
        ... )
        >>> spectrum
        Spectrum(mz=array([1000., 1500., 2000.]), ...)
    """

    mz: NDArray[np.float64]
    intensity: NDArray[np.float32]
    ms_level: int
    precursor_mz: float | None
    precursor_charge: int | None = None
    instrument: str | None = None
    analyzer: str | None = None
    collision_energy: float = field(default=float("nan"))
    activation: str | None = None
    extras: dict = field(default_factory=dict)
    retention_time: RetentionTime | float = field(
        default_factory=lambda: RetentionTime(rt=float("nan"), units="seconds")
    )
    config: Config = field(repr=False, default_factory=get_default_config)

    def __post_init__(self):
        if self.retention_time is None:
            self.retention_time = RetentionTime(rt=np.nan, units="minutes")

    def replace(self, *args: Any, **kwargs: Any) -> Spectrum:
        """Replaces the attributes of the spectrum with the given values.

        Arguments are passed to the class Spectrum constructor.

        Returns:
            Spectrum, A new Spectrum object with the replaced attributes.
        """
        return dataclasses.replace(self, *args, **kwargs)

    def filter_top(self, n: int) -> Spectrum:
        """Filters the spectrum to the top n peaks.


        Args:
            n: The number of peaks to keep.

        Returns:
            The spectrum with the filtered top n peaks.

        Examples:
            >>> spectrum = Spectrum._sample()
            >>> spectrum.intensity
            array([ 50., 200.,   1.,   2.,   3.])
            >>> spectrum.filter_top(2).intensity
            array([ 50., 200.])
            >>> Spectrum._sample().filter_top(3).intensity
            array([  3.,  50., 200.])
        """
        ind = np.argsort(self.intensity)
        ind = ind[-n:]
        mz, intensity = self.mz[ind], self.intensity[ind]
        out = self.replace(mz=mz, intensity=intensity)
        clear_lazy_cache(out)
        return out

    def filter_mz_range(self, min_mz: float, max_mz: float) -> Spectrum:
        """Filters the spectrum to a given m/z range.

        Returns:
            The spectrum with the filtered m/z range.

        Examples:
            >>> spectrum = Spectrum._sample()
            >>> spectrum.mz
            array([  50.     ,  147.11333, 1000.     , 1500.     , 2000.     ])
            >>> spectrum.tic
            256.0
            >>> spectrum = spectrum.filter_mz_range(124, 1600)
            >>> spectrum.mz
            array([ 147.11333, 1000.     , 1500.     ])
            >>> spectrum.filter_mz_range(124, 1600).tic
            203.0
        """
        mask = (self.mz >= min_mz) & (self.mz <= max_mz)
        mz, intensity = self.mz[mask], self.intensity[mask]
        out = self.replace(mz=mz, intensity=intensity)
        clear_lazy_cache(out)
        return out

    def remove_precursor(self) -> Spectrum:
        """Removes the precursor peak from the spectrum

        Returns:
            Spectrum, A new Spectrum object with the precursor peak removed.

        """
        if self.precursor_mz is None:
            warnings.warn("Precursor m/z not set. Cannot remove precursor.")
            return self

        tolerance = self.config.g_tolerances[self.ms_level - 1]
        tolerance_unit = self.config.g_tolerance_units[self.ms_level - 1]

        tol = get_tolerance(
            tolerance=tolerance, theoretical=self.precursor_mz, unit=tolerance_unit
        )
        mask = np.abs(self.mz - self.precursor_mz) > tol
        mz, intensity = self.mz[mask], self.intensity[mask]
        out = self.replace(mz=mz, intensity=intensity)
        clear_lazy_cache(out)

        return out

    def intensity_cutoff(self, cutoff: float) -> Spectrum:
        """Filters the spectrum to a given intensity cutoff.

        Args:
            cutoff: The intensity cutoff.
                All peaks with less than that intensity will be deleted.

        Returns:
            A new Spectrum object with the filtered intensity cutoff.

        Examples:
            >>> spectrum = Spectrum._sample()
            >>> spectrum.intensity
            array([ 50., 200.,   1.,   2.,   3.])
            >>> spectrum.intensity_cutoff(2.1).intensity
            array([ 50.0, 200.,  3.])
            >>> spectrum.intensity
            array([ 50., 200.,   1.,   2.,   3.])
        """
        mask = self.intensity >= cutoff
        mz, intensity = self.mz[mask], self.intensity[mask]
        out = self.replace(mz=mz, intensity=intensity)
        clear_lazy_cache(out)

        return out

    def normalize_intensity(self, method: str = "max") -> Spectrum:
        """Normalizes the spectrum intensities.

        Args:
            method: The method to use for normalization.
                Can be one of "max", "sum", "rank", "log".

        Returns:
            The normalized spectrum.

        Examples:
            >>> spectrum = Spectrum._sample()
            >>> spectrum.intensity
            array([ 50., 200.,   1.,   2.,   3.])
            >>> Spectrum._sample().normalize_intensity("max").intensity
            array([0.25 , 1.   , 0.005, 0.01 , 0.015])
            >>> Spectrum._sample().normalize_intensity("sum").intensity
            array([0.1953125 , 0.78125   , 0.00390625, 0.0078125 , 0.01171875])
            >>> Spectrum._sample().normalize_intensity("log").intensity
            array([3.91202301, 5.29831737, 0.        , 0.69314718, 1.09861229])
            >>> Spectrum._sample().normalize_intensity("sqrt").intensity
            array([ 7.07106781, 14.14213562,  1.        ,  1.41421356,  1.73205081])
        """
        if method == "max":
            intensity = self.intensity / np.max(self.intensity)
        elif method == "sum":
            intensity = self.intensity / np.sum(self.intensity)
        elif method == "rank":
            intensity = self.intensity / np.argsort(self.intensity)
        elif method == "log":
            intensity = np.log(self.intensity)
        elif method == "sqrt":
            intensity = np.sqrt(self.intensity)
        else:
            raise ValueError(f"Normalization method {method} not recognized.")

        out = self.replace(intensity=intensity)
        clear_lazy_cache(out)
        return out

    def encode_spec_bins(self) -> NDArray[np.float32]:
        """Encodes the spectrum into bins.

        For a version of this function that takes arguments indead of reading the
        options from the config, see `Spectrum.bin_spectrum`.

        Uses the following options from the config:
            Config.encoding_spec_bin_start,
            Config.encoding_spec_bin_end,
            Config.encoding_spec_bin_n_bins,
            Config.encoding_spec_bin_binsize,
            Config.encoding_spec_bin_relative,
            Config.encoding_spec_bin_offset,

        Returns:
            The encoded spectrum.

        Examples:
            >>> Spectrum._sample().encode_spec_bins().shape
            (19999,)
        """
        return self.bin_spectrum(
            start=self.config.encoding_spec_bin_start,
            end=self.config.encoding_spec_bin_end,
            n_bins=self.config.encoding_spec_bin_n_bins,
            binsize=self.config.encoding_spec_bin_binsize,
            relative=self.config.encoding_spec_bin_relative,
            offset=self.config.encoding_spec_bin_offset,
            get_breaks=False,
        )

    @overload
    def bin_spectrum(
        self,
        start: float,
        end: float,
        binsize: float | None = None,
        n_bins: int | None = None,
        relative: bool = False,
        offset: float = 0,
        get_breaks: Literal[True] = True,
    ) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
        ...

    @overload
    def bin_spectrum(
        self,
        start: float,
        end: float,
        binsize: float | None = None,
        n_bins: int | None = None,
        relative: bool = False,
        offset: float = 0,
        get_breaks: Literal[False] = False,
    ) -> NDArray[np.float32]:
        ...

    def bin_spectrum(
        self,
        start: float,
        end: float,
        binsize: float | None = None,
        n_bins: int | None = None,
        relative: bool = False,
        offset: float = 0,
        get_breaks: bool = False,
    ) -> NDArray[np.float32] | tuple[NDArray[np.float32], NDArray[np.float64]]:
        """Bins the spectrum.

        Args:
            start: The start of the binning range.
                If missing will use the lowest mz value.
            end: The end of the binning range.
                If missing will use the highest mz value.
            binsize: The size of the bins. Cannot be used in conjunction with n_bins.
            n_bins: The number of bins. Cannot be used in conjunction with binsize.
            relative: Whether to use binning relative to the precursor mass.
            offset: The offset to use for relative binning.

        Returns:
            An array of binned intensities.
        """
        mz_arr = self.mz
        assert start < end

        if relative:
            if isinstance(relative, float) or isinstance(relative, int):
                relative_value = float(relative)
            else:
                if self.precursor_mz:
                    relative_value = float(self.precursor_mz)
                else:
                    raise ValueError(
                        "Cannot use relative binning without precursor m/z"
                        " or a relative value. Pass either a value to 'relative'"
                        " or a precursor m/z to the spectrum."
                    )

            start = start + relative_value
            end = end + relative_value
            mz_arr = mz_arr - relative_value + offset
        elif offset:
            msg = "Cannot use offset without relative binning."
            msg += " (relative=True) in bin_spectrum"

            raise ValueError(msg)

        binned = _bin_spectrum(
            mz=mz_arr,
            start=start,
            end=end,
            binsize=binsize,
            n_bins=n_bins,
            weights=self.intensity,
            get_breaks=get_breaks,
        )

        return binned

    @lazy
    def base_peak(self) -> np.float32:
        """Returns the base peak intensity of the spectrum."""
        return np.max(self.intensity)

    @base_peak.setter
    def base_peak(self, value) -> None:
        self._base_peak = value

    @lazy
    def tic(self) -> np.float32:
        """Returns the total ion current of the spectrum."""
        return np.sum(self.intensity)

    @tic.setter
    def tic(self, value) -> None:
        self._lazy_tic = value

    def sic(
        self, mzs: NDArray[np.float32], resolution: Callable = sum
    ) -> NDArray[np.float32]:
        """Returns the selected ion current for a given set of m/z values.

        Args:
            mzs: The m/z values to calculate the SIC for.
            resolution: The function used to resolve ambiguities when multiple
                peaks match.  possible options are `sum` and `max`.

        Returns:
            An array of SIC values. This array will have the same length as the
            input mzs array.

        Examples:
            >>> spectrum = Spectrum._sample()
            >>> spectrum.mz
            array([  50.     ,  147.11333, 1000.     , 1500.     , 2000.     ])
            >>> spectrum.intensity
            array([ 50., 200.,   1.,   2.,   3.])
            >>> spectrum.sic(
            ...     np.array([1000.0, 1000.00001, 1500.0, 2000.0, 20_000.0]),
            ...     resolution=sum,
            ... )
            array([1., 1., 2., 3., 0.])
        """

        theo_mz_indices, obs_mz_indices = annotate_peaks(
            theo_mz=mzs,
            mz=self.mz,
            tolerance=self.config.g_tolerances[self.ms_level - 1],
            unit=self.config.g_tolerance_units[self.ms_level - 1],
        )

        outs = []
        for i, _ in enumerate(mzs):
            ints_subset = self.intensity[obs_mz_indices[theo_mz_indices == i]]
            if len(ints_subset) == 0:
                outs.append(0)
            else:
                outs.append(resolution(ints_subset))

        return np.array(outs)

    @staticmethod
    def _sample(random: bool = False) -> Spectrum:
        """Returns a sample Spectrum object."""
        config = Config()
        if random:
            num_peaks = np.random.randint(low=1, high=5)
            mz = np.random.uniform(low=0, high=1000, size=num_peaks)
            intensity = np.random.uniform(low=0, high=1000, size=num_peaks)
        else:
            mz = np.array([50.0, 147.11333, 1000.0, 1500.0, 2000.0])
            intensity = np.array([50.0, 200.0, 1.0, 2.0, 3.0])

        spectrum = Spectrum(
            mz=mz,
            intensity=intensity,
            ms_level=2,
            extras={"EXTRAS": ["extra1", "extra2"]},
            config=config,
            precursor_mz=147.11333,
        )
        return spectrum

    def annotate(self, peptide: str | Peptide) -> AnnotatedPeptideSpectrum:
        """Annotates the spectrum with the given peptide.

        Args:
            peptide: The peptide to annotate the spectrum with.

        Returns:
            An AnnotatedPeptideSpectrum object.

        Examples:
            >>> spectrum = Spectrum._sample()
            >>> peptide = Peptide.from_sequence("PEPPINK/2", spectrum.config)
            >>> annotated_spectrum = spectrum.annotate(peptide)
            >>> annotated_spectrum
            AnnotatedPeptideSpectrum(mz=array([  50. ... precursor_isotope=0)
        """
        if isinstance(peptide, str):
            peptide = Peptide.from_sequence(peptide, config=self.config)

        spec_dict = dataclasses.asdict(self)
        spec_dict["config"] = self.config
        spec_dict["retention_time"] = self.retention_time
        spec = AnnotatedPeptideSpectrum(precursor_peptide=peptide, **spec_dict)
        return spec

    def to_sus(self):
        if sus_MsmsSpectrum is None:
            raise ImportError(
                "The spectrum_utils library is not installed. "
                "Please install it with `pip install spectrum_utils`."
            )
        msmsspec = sus_MsmsSpectrum(
            identifier="RandomSpec",
            mz=self.mz,
            precursor_mz=self.precursor_mz or 0,
            intensity=self.intensity,
            precursor_charge=self.precursor_charge or 0,
        )
        return msmsspec

    def plot(self, ax=None, **kwargs) -> plt.Axes:
        if plotspec is None:
            raise ImportError(
                "Unable to find spectrum_utils, "
                "please install it to use the plotting functionality of ms2ml."
            )
        msmsspec = self.to_sus()
        return plotspec(msmsspec, ax=ax, **kwargs)

    @classmethod
    def stack(cls, spectra: Iterable[Spectrum]) -> Spectrum:
        """Stacks multiple spectra into a single spectrum.

        Args:
            spectra: An iterable of spectra to stack.

        Returns:
            A stacked spectrum.

        Examples:
            >>> np.random.seed(0)
            >>> spectrum1 = Spectrum._sample()
            >>> spectrum2 = Spectrum._sample(random=True)
            >>> spectrum3 = Spectrum._sample(random=True)
            >>> spectrum4 = Spectrum._sample(random=False)
            >>> spectrum4.mz = spectrum1.mz + 1e-4
            >>> mzs, ints = Spectrum.stack([spectrum1, spectrum2, spectrum3, spectrum4])
            >>> mzs
            array([  50.        ,  147.11333   ,  423.65479934,  437.58721126,
                    544.883183  ,  592.84461823,  645.89411307, 1000.        ,
                   1500.        , 2000.        ])
            >>> ints
            array([[ 50.        ,   0.        ,   0.        ,  50.        ],
                   [200.        ,   0.        ,   0.        , 200.        ],
                   [  0.        ,   0.        , 963.6627605 ,   0.        ],
                   [  0.        ,   0.        , 791.72503808,   0.        ],
                   [  0.        ,   0.        , 891.77300078,   0.        ],
                   [  0.        , 844.26574858,   0.        ,   0.        ],
                   [  0.        ,   0.        , 383.44151883,   0.        ],
                   [  1.        ,   0.        ,   0.        ,   1.        ],
                   [  2.        ,   0.        ,   0.        ,   2.        ],
                   [  3.        ,   0.        ,   0.        ,   3.        ]])
            >>> ints.shape
            (10, 4)

        """
        spectra = list(spectra)
        ref_spec = spectra[0]
        assert all([s.ms_level == spectra[0].ms_level for s in spectra])
        assert all([s.config == spectra[0].config for s in spectra])

        mz_pairs = [(s.mz, s.intensity) for s in spectra]
        mz, new_int = stack_mz_pairs(
            mz_pairs,
            tolerance=ref_spec.config.g_tolerances[ref_spec.ms_level - 1],
            units=ref_spec.config.g_tolerance_units[ref_spec.ms_level - 1],
        )
        order = np.argsort(mz)
        mz = mz[order]
        new_int = new_int[order]
        return mz, new_int


def _bin_spectrum(
    mz: NDArray[np.float64],
    weights: NDArray[np.float32],
    start: float,
    end: float,
    binsize: float | None = None,
    n_bins: int | None = None,
    get_breaks: bool = False,
) -> NDArray[np.float32] | tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Bins the spectrum.

    Args:
        mz: The m/z values of the spectrum.
        weights: The intensity values of the spectrum.
        start: The start of the binning range.
        end: The end of the binning range.
        binsize: The size of the bins.
        n_bins: The number of bins.

    Returns:
        An array of binned intensities.
    """

    if binsize is None and n_bins is not None:
        pass
    elif binsize is not None and n_bins is None:
        n_bins = math.ceil((end - start) / binsize)
        bins = np.linspace(start, end, num=n_bins)
    else:
        msg = "Either binsize or n_bins must"
        msg += " be provided to bin_spectrum. Not both."
        raise ValueError(msg)

    bins = np.linspace(start, end, num=n_bins)
    hist = np.histogram(
        mz,
        bins=bins,
        weights=weights,
    )

    if get_breaks:
        return hist

    return hist[0]


@dataclass
class AnnotatedPeptideSpectrum(Spectrum):
    """Class to store the spectrum information.

    In combination with the peptide information, it can be used to
    annotate the spectrum.

    Examples:
        >>> config = Config()
        >>> peptide = Peptide.from_sequence("PEPPINK/2", config)
        >>> spectrum = AnnotatedPeptideSpectrum(
        ...     mz=np.array([50.0, 147.11333, 1000.0, 1500.0, 2000.0]),
        ...     intensity=np.array([50.0, 200.0, 1.0, 2.0, 3.0]),
        ...     ms_level=2,
        ...     extras={"EXTRAS": ["extra1", "extra2"]},
        ...     precursor_peptide=peptide,
        ...     precursor_mz=147.11333,
        ... )
        >>> spectrum
        AnnotatedPeptideSpectrum(mz=array([  50. ... precursor_isotope=0)
        >>> spectrum.fragment_intensities
        {'y1^1': 200.0}
        >>> spectrum["y1^1"]
        200.0
        >>> spectrum.fragments
        {'y1^1': AnnotatedIon(mass=147.11334,
        charge=1, position=1, ion_series='y', intensity=200.0, neutral_loss=None)}
    """

    # TODO find a way to not make this optional ...
    # right now it has to be due to the fact that when it inherits from
    # Spectrum, it already has optional arguments, that cannot be followed
    # by positional arguments
    precursor_peptide: Peptide | None = None
    precursor_isotope: int | None = 0
    precursor_charge: int | None = None

    def __post_init__(self, *args, **kwargs):
        if self.config is None and self.precursor_peptide is not None:
            warnings.warn(
                "No config provided, falling back to the one in the peptide",
                UserWarning,
            )
            self.config = self.precursor_peptide.config
        elif self.config is None:
            raise ValueError("No config provided, please provide one or a peptide")

        if self.precursor_charge is None and self.precursor_peptide is not None:
            self.precursor_charge = self.precursor_peptide.charge
        elif self.precursor_charge is None:
            raise ValueError(
                "No precursor charge provided, please provide one or a peptide"
            )

        super().__post_init__(*args, **kwargs)

    @property
    def charge(self):
        return self.precursor_charge

    @property
    def mass_error(self):
        raise NotImplementedError

    def _annotate_peaks(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Annotates the peaks of the spectrum.

        Internal function that returns what indices in the observed mz array match the
        theoretical mz array.
        """
        if self.precursor_peptide is None:
            raise ValueError(
                "No precursor peptide provided. Which is required to annotate the peaks"
            )
        theo_mzs = self.precursor_peptide.theoretical_ion_masses

        if self.config is None:
            raise ValueError(
                "No config provided. Which is required to annotate the peaks"
            )
        tolerance = self.config.g_tolerances[self.ms_level - 1]
        tolerance_unit = self.config.g_tolerance_units[self.ms_level - 1]

        annot_indices, mz_indices = annotate_peaks(
            theo_mz=theo_mzs, mz=self.mz, tolerance=tolerance, unit=tolerance_unit
        )
        return annot_indices, mz_indices

    @property
    def _annot_indices(self) -> NDArray[np.int32]:
        return self._indices[0]

    @property
    def _mz_indices(self) -> NDArray[np.int32]:
        return self._indices[1]

    # TODO implement getting individual ion series, same API as peptide

    @lazy
    def fragment_intensities(self) -> dict[str, float]:
        """
        Returs a dictionary with the fragment ion names as keys and the
        corresponding intensities as values.

        Note:
            The current implementation only keeps the last peak that matches
            the theoretical mass. future implementations should either keep all peaks
            or only the highest peak, or add the peaks.

        Examples:
            >>> spec = AnnotatedPeptideSpectrum._sample()
            >>> spec.fragment_intensities
            {'y1^1': 200.0}
        """
        return {label: v.intensity for label, v in self.fragments.items()}

    @lazy
    def fragments(self) -> dict[str, AnnotatedIon]:
        """
        Returs a dictionary with the fragment ion names as keys and the
        corresponding AnnotatedIon objects as values.

        Examples:
            >>> spec = AnnotatedPeptideSpectrum._sample()
            >>> spec.fragments
            {'y1^1': AnnotatedIon(mass=147.11334,
            charge=1, position=1, ion_series='y', intensity=200.0,
            neutral_loss=None)}

        """
        if self.precursor_peptide is None:
            raise ValueError(
                "No precursor peptide provided. Which is required to annotate the"
                " fragments"
            )

        if len(self._annot_indices) == 0:
            return {}

        labels = self.precursor_peptide.theoretical_ion_labels[self._annot_indices]
        intensities = self.intensity[self._mz_indices]
        mzs = self.mz[self._mz_indices]

        frags: dict[str, AnnotatedIon] = {}

        for label, i, _ in zip(labels, intensities, mzs):
            # TODO implement ambiguity resoluitions
            frag = frags.get(label, None)
            if frag is None:
                frags[label] = self.precursor_peptide.ion_dict[label]
                frags[label].intensity = 0.0

            frags[label].intensity += i

        return frags

    def __getitem__(self, index) -> float:
        """
        Returns the intensity of the fragment ion with the given name.

        Examples:
            >>> spec = AnnotatedPeptideSpectrum._sample()
            >>> spec["y1^1"]
            200.0
        """
        return self.fragment_intensities.get(index, 0.0)

    @lazy
    def _indices(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        return self._annotate_peaks()

    def encode_fragments(self) -> NDArray[np.float32]:
        """Encodes the fragment ions as a numpy array

        The order of the ions will be defined in the config file.

        Examples:
            >>> spec = AnnotatedPeptideSpectrum._sample()
            >>> spec.encode_fragments()
            array([200.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            dtype=float32)
        """
        return np.array([self[k] for k in self.fragment_labels], dtype=np.float32)

    @staticmethod
    def decode_fragments(peptide: Peptide, fragment_vector: NDArray[np.float32]):
        """
        Examples:
            >>> spec = AnnotatedPeptideSpectrum._sample()
            >>> pep = spec.precursor_peptide
            >>> frags = spec.encode_fragments()
            >>> AnnotatedPeptideSpectrum.decode_fragments(pep, frags)
            AnnotatedPeptideSpectrum(mz=array([...]),
            intensity=array([...], dtype=float32),
            ms_level=2, precursor_mz=397.724526907315,
            precursor_charge=2,
            instrument=None,
            analyzer=None,
            collision_energy=nan,
            activation=None,
            extras={},
            retention_time=RetentionTime(rt=nan, units='seconds',
            run=None),
            precursor_peptide=Peptide([...], {...}), precursor_isotope=0)

        """
        tmp = [
            (peptide.ion_dict[label].mass, v)
            for label, v in zip(peptide.config.fragment_labels, fragment_vector)
            if label in peptide.ion_dict
        ]
        masses, intensities = zip(*tmp)
        masses, intensities = np.array(masses, dtype=np.float64), np.array(
            intensities, dtype=np.float32
        )
        spec = Spectrum(
            mz=masses,
            intensity=intensities,
            precursor_mz=mz(mass=peptide.mass, charge=peptide.charge),
            precursor_charge=peptide.charge,
            ms_level=2,
            config=peptide.config,
        )
        return spec.annotate(peptide=peptide)

    @property
    def fragment_labels(self) -> list[str]:
        """Encodes the fragment ions as a numpy array

        The order of the ions will be defined in the config file.

        Examples:
            >>> spec = AnnotatedPeptideSpectrum._sample()
            >>> spec.fragment_labels
            ['y1^1', ...]
        """
        return self.config.fragment_labels

    def to_sus(self):
        """Return a spectrum object as a spectrum_utils.spectrum.Spectrum object"""
        msmsspec = super().to_sus()
        msmsspec = msmsspec.annotate_proforma(
            proforma_str=self.precursor_peptide.to_proforma(),
            fragment_tol_mass=self.config.g_tolerances[self.ms_level - 1],
            fragment_tol_mode=self.config.g_tolerance_units[self.ms_level - 1],
            ion_types=self.config.ion_series,
            neutral_losses=True,
        )
        return msmsspec

    @staticmethod
    def _sample():
        """Returns a sample AnnotatedPeptideSpectrum object.

        Examples:
            >>> config = Config()
            >>> peptide = Peptide.from_sequence("PEPPINK/2", config)
            >>> spectrum = AnnotatedPeptideSpectrum(
            ...     mz=np.array([50.0, 147.11333, 1000.0, 1500.0, 2000.0]),
            ...     intensity=np.array([50.0, 200.0, 1.0, 2.0, 3.0]),
            ...     ms_level=2,
            ...     extras={"EXTRAS": ["extra1", "extra2"]},
            ...     precursor_mz=147.1130,
            ...     precursor_peptide=peptide,
            ... )
            >>> spectrum
            AnnotatedPeptideSpectrum(mz=array([  50. ... precursor_isotope=0)

            _sample would retun the same as the former code

            >>> spec = AnnotatedPeptideSpectrum._sample()
            >>> spec
            AnnotatedPeptideSpectrum(mz=array([  50. ... precursor_isotope=0)
        """
        config = Config()
        peptide = Peptide.from_sequence("PEPPINK/2", config)
        spectrum = AnnotatedPeptideSpectrum(
            mz=np.array([50.0, 147.11333, 1000.0, 1500.0, 2000.0]),
            intensity=np.array([50.0, 200.0, 1.0, 2.0, 3.0]),
            ms_level=2,
            extras={"EXTRAS": ["extra1", "extra2"]},
            precursor_peptide=peptide,
            precursor_mz=147.11333,
        )
        return spectrum
