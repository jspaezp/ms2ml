from __future__ import annotations

from typing import Callable, Iterator

from ms2ml.config import Config
from ms2ml.data.parsing.fasta import FastaDataset
from ms2ml.peptide import Peptide

from .base import BaseAdapter


class FastaAdapter(BaseAdapter, FastaDataset):
    """Implements an adapter that reads fasta files

    Args:
        file (str): Path to the fasta file
        config (Config): The config object
        only_unique (bool, optional):
            Whether to only keep unique peptides. Defaults to True.
        enzyme (str, optional): The enzyme to use. Defaults to "trypsin".
        missed_cleavages (int, optional):
            The number of missed cleavages. Defaults to 2.
        in_hook (Callable, optional):
            A function to apply to each element before processing. Defaults to None.
        out_hook (Callable, optional):
            A function to apply to each element after processing. Defaults to None.
        collate_fn (Callable, optional):
            A function to collate the elements. Defaults to list.

    """

    def __init__(
        self,
        file: str,
        config: Config,
        only_unique: bool = True,
        enzyme: str = "trypsin",
        missed_cleavages: int = 2,
        in_hook: Callable | None = None,
        out_hook: Callable | None = None,
        collate_fn: Callable = list,
    ):
        BaseAdapter.__init__(
            self,
            config=config,
            in_hook=in_hook,
            out_hook=out_hook,
            collate_fn=collate_fn,
        )
        FastaDataset.__init__(
            self,
            file=file,
            enzyme=enzyme,
            missed_cleavages=missed_cleavages,
            min_length=config.peptide_length_range[0],
            max_length=config.peptide_length_range[1],
            only_unique=only_unique,
        )

    def parse(self) -> Iterator[Peptide]:
        charges = tuple(self.config.precursor_charges)
        for spec in super().parse():
            for charge in charges:
                spec["charge"] = charge
                yield self._process_elem(spec)

    def _to_elem(self, elem: dict) -> Peptide:
        pep = Peptide.from_proforma_seq(
            f"{elem['sequence']}/{elem['charge']}",
            config=self.config,
            extras=elem["header"],
        )
        return pep
