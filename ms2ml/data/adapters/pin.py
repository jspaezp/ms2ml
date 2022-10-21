from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterator

from ms2ml.config import Config
from ms2ml.data.parsing.pin import PinParser
from ms2ml.peptide import Peptide
from ms2ml.spectrum import AnnotatedPeptideSpectrum

from .base import BaseAdapter
from .mzml import MZMLAdapter


class PinAdapter(PinParser, BaseAdapter):
    def __init__(
        self,
        file: PathLike[Any],
        config: Config,
        in_hook: Callable | None = None,
        out_hook: Callable | None = None,
        collate_fn: Callable[..., Any] | None = None,
        raw_file_locations: list[PathLike] | None = None,
    ):
        """Provides an adapter for pin files.

        Args:
            file (PathLike[Any]): Path to the file to be parsed.
            config (Config): The config object to use.
            in_hook (Callable, optional): A function to apply to the
                input of the adapter. Defaults to None.
            out_hook (Callable, optional): A function to apply to the
                output of the adapter. Defaults to None.
            collate_fn (Callable[..., Any], optional):
                A function to use to collate the output of the adapter.
                Defaults to pad_collate.
            raw_file_locations (list[PathLike], optional):
                A list of locations to search for raw files,
                if none is provided, attempts to find the files recursively
                in the current working directory. Defaults to None.

        """
        BaseAdapter.__init__(
            self,
            config,
            in_hook=in_hook,
            out_hook=out_hook,
            collate_fn=collate_fn,
        )
        PinParser.__init__(self, file)

        if self.pin_flavour == "sage":
            self._to_elem = self._to_elem_sage
        elif self.pin_flavour == "comet":
            self._to_elem = self._to_elem_comet

        self.mzml_adapters = {}

        if raw_file_locations is None:
            self.raw_file_locations = ["."]
        elif isinstance(raw_file_locations, (str, Path, PathLike)):
            self.raw_file_locations = [raw_file_locations]
        else:
            self.raw_file_locations = raw_file_locations

    def parse_file(self, file: PathLike[Any]) -> Iterator[AnnotatedPeptideSpectrum]:
        """Parses a pin file and yields one spectrum at a time.

        The spectra are yielded as AnnotatedPeptideSpectrum"""
        if file != self.file:
            adapter = PinAdapter(
                file, self.config, self.in_hook, self.out_hook, self.collate_fn
            )
            yield from adapter.parse_file(file)
        else:
            for spec in super().parse_file(file):
                yield self._process_elem(spec)

    def _to_elem(self, spec_dict) -> AnnotatedPeptideSpectrum:
        raise NotImplementedError

    def _to_elem_sage(self, spec_dict) -> AnnotatedPeptideSpectrum:
        # TODO improve this ahdnling of carbamidomethyl
        seq = spec_dict["peptide"].replace("(57.0215)", "")
        seq = seq.replace("(", "[+").replace(")", "]")
        seq = f"{seq}/{spec_dict['charge']}"
        pep = Peptide.from_proforma_seq(seq, config=self.config)

        if self.file not in self.mzml_adapters:
            # sage pin files have results for a single mzml file
            mzml_file = Path(self.file.replace(".sage", "")).with_suffix(".mzML")
            if mzml_file.exists():
                mzml_file = mzml_file.resolve()
                self.mzml_adapters[self.file] = MZMLAdapter(str(mzml_file), self.config)
            elif Path(self.raw_file_locations[0]).is_file():
                self.mzml_adapters[self.file] = MZMLAdapter(
                    self.raw_file_locations[0], self.config
                )
            else:
                self.mzml_adapters[self.file] = MZMLAdapter(
                    self._find_raw_file(mzml_file.name), self.config
                )

        spec = self.mzml_adapters[self.file][spec_dict["scannr"]]
        spec = spec.annotate(pep)
        spec.extras.update(spec_dict)

        return spec

    def _to_elem_comet(self, spec_dict) -> AnnotatedPeptideSpectrum:
        seq = f"{spec_dict['PeptideSequence']}/{spec_dict['PrecursorCharge']}"
        pep = Peptide.from_proforma_seq(seq, config=self.config)

        if spec_dict["RawFile"] not in self.mzml_adapters:
            self.mzml_adapters[spec_dict["RawFile"]] = MZMLAdapter(
                self._find_raw_file(spec_dict["RawFile"]), self.config
            )

        spec = self.mzml_adapters[spec_dict["RawFile"]][spec_dict["SpectrumIndex"]]
        spec = spec.annotate(pep)
        spec.extras.update(spec_dict)

        return spec

    def _process_elem(self, elem: AnnotatedPeptideSpectrum) -> AnnotatedPeptideSpectrum:
        elem = super()._process_elem(elem)
        return elem

    def _find_raw_file(self, raw_file: str) -> str:
        outs = []
        for loc in self.raw_file_locations:
            tmp = list(Path(loc).rglob(f"*{raw_file}*"))
            outs.extend(tmp)
            tmp = list(Path(loc).rglob(f"*{Path(raw_file).name}*"))
            outs.extend(tmp)

        outs = [x.resolve() for x in outs if "mzML" in str(x)]
        outs = list(set(outs))
        if len(outs) == 0:
            raise FileNotFoundError(f"Could not find {raw_file}.mzML")
        elif len(outs) > 1:
            raise FileNotFoundError(f"Found multiple files for {raw_file}.mzML, {outs}")

        return outs[0]
