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

import numpy as np

from .annotation_classes import AnnotatedIon, RetentionTime
from .config import Config, get_default_config
from .peptide import Peptide
from .utils import annotate_peaks


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
    precursor_mz: float
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

    # TODO add this to config
    def bin_spectrum(
        self,
        start: float,
        end: float,
        binsize: float = None,
        n_bins: int = None,
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

    @property
    def base_peak(self) -> float:
        """Returns the base peak intensity of the spectrum."""
        if not hasattr(self, "_base_peak"):
            self._base_peak = np.max(self.intensity)
        return self._base_peak

    @base_peak.setter
    def base_peak(self, value) -> None:
        self._base_peak = value

    @property
    def tic(self) -> float:
        """Returns the total ion current of the spectrum."""
        if not hasattr(self, "_tic"):
            self._tic = np.sum(self.intensity)
        return self._tic

    @tic.setter
    def tic(self, value) -> None:
        self._tic = value

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
        spec = AnnotatedPeptideSpectrum(precursor_peptide=peptide, **spec_dict)
        return spec


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

        mz_indices, annot_indices = annotate_peaks(
            theo_mz=theo_mzs, mz=self.mz, tolerance=tolerance, unit=tolerance_unit
        )
        return annot_indices, mz_indices

    @property
    def _annot_indices(self) -> np.ndarray:
        return self._indices[0]

    @property
    def _mz_indices(self) -> np.ndarray:
        return self._indices[1]

    @property
    def fragment_intensities(self) -> dict[str, float]:
        """
        Returs a dictionary with the fragment ion names as keys and the
        corresponding intensities as values.

        Examples:
            >>> spec = AnnotatedPeptideSpectrum._sample()
            >>> spec.fragment_intensities
            {'y1^1': 200.0}
        """
        if not hasattr(self, "_fragment_intensities"):
            self._fragment_intensities = {
                label: v.intensity for label, v in self.fragments.items()
            }

        return self._fragment_intensities

    @property
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

        if not hasattr(self, "_fragments"):
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
                    frags[label] = self.precursor_peptide.ion_series_dict[label]
                    frags[label].intensity = 0.0

                frags[label].intensity += i

            self._fragments = frags

        return self._fragments

    def __getitem__(self, index) -> float:
        """
        Returns the intensity of the fragment ion with the given name.

        Examples:
            >>> spec = AnnotatedPeptideSpectrum._sample()
            >>> spec["y1^1"]
            200.0
        """
        return self.fragment_intensities.get(index, 0.0)

    @property
    def _indices(self) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "_indices_"):
            self._indices_ = self._annotate_peaks()

        return self._indices_

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
