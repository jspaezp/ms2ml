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
from typing import Any

import numpy as np

from .annotation_classes import AnnotatedIon, RetentionTime
from .config import Config, get_default_config
from .peptide import Peptide
from .utils import annotate_peaks, clear_lazy_cache, get_tolerance, lazy

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

    mz: np.ndarray
    intensity: np.ndarray
    ms_level: int
    precursor_mz: float | None
    precursor_charge: int | None = None
    instrument: str | None = None
    analyzer: str | None = None
    extras: dict | None = None
    retention_time: RetentionTime | float | None = None
    config: Config | None = field(repr=False, default=None)

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}

        if self.config is None:
            self.config = get_default_config()

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

    def encode_spec_bins(self) -> np.typing.NDArray[np.float]:
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
        )

    def bin_spectrum(
        self,
        start: float,
        end: float,
        binsize: float | None = None,
        n_bins: int | None = None,
        relative: bool = False,
        offset: float = 0,
        get_breaks: bool = False,
    ) -> np.ndarray:
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
                relative_value = relative
            else:
                relative_value = self.precursor_mz

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
    def base_peak(self) -> float:
        """Returns the base peak intensity of the spectrum."""
        return np.max(self.intensity)

    @base_peak.setter
    def base_peak(self, value) -> None:
        self._base_peak = value

    @lazy
    def tic(self) -> float:
        """Returns the total ion current of the spectrum."""
        return np.sum(self.intensity)

    @tic.setter
    def tic(self, value) -> None:
        self._lazy_tic = value

    def sic(self, mzs: np.array, resolution: sum) -> np.array:
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
    def _sample():
        """Returns a sample Spectrum object."""
        config = Config()
        spectrum = Spectrum(
            mz=np.array([50.0, 147.11333, 1000.0, 1500.0, 2000.0]),
            intensity=np.array([50.0, 200.0, 1.0, 2.0, 3.0]),
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
            precursor_mz=self.precursor_mz,
            intensity=self.intensity,
            precursor_charge=self.precursor_charge,
        )
        return msmsspec

    def plot(self, ax=None, **kwargs) -> plt.Axes:
        msmsspec = self.to_sus()
        return plotspec(msmsspec, ax=ax, **kwargs)


def _bin_spectrum(
    mz: np.ndarray,
    weights: np.ndarray,
    start: float,
    end: float,
    binsize=None,
    n_bins=None,
    get_breaks=False,
) -> np.ndarray:
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
        {'y1^1': AnnotatedIon(mass=array(147.11334, dtype=float32),
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
        if self.config is None:
            warnings.warn(
                "No config provided, falling back to the one in the peptide",
                UserWarning,
            )
            self.config = self.precursor_peptide.config

        if self.precursor_charge is None:
            self.precursor_charge = self.precursor_peptide.charge

        super().__post_init__(*args, **kwargs)

    @property
    def charge(self):
        return self.precursor_peptide.charge

    @property
    def mass_error(self):
        raise NotImplementedError

    def _annotate_peaks(self) -> tuple[np.ndarray, np.ndarray]:
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
    def _annot_indices(self) -> np.ndarray:
        return self._indices[0]

    @property
    def _mz_indices(self) -> np.ndarray:
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
            {'y1^1': AnnotatedIon(mass=array(147.11334, dtype=float32),
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
    def _indices(self) -> tuple[np.ndarray, np.ndarray]:
        return self._annotate_peaks()

    def encode_fragments(self) -> np.float32:
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
        msmsspec = super().to_sus()
        msmsspec = msmsspec.annotate_proforma(
            proforma_str=self.precursor_peptide.to_proforma(),
            fragment_tol_mass=self.config.g_tolerance_units[self.ms_level - 1],
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
