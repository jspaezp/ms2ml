from __future__ import annotations

from typing import Callable, Iterator

from loguru import logger

from ms2ml.config import Config
from ms2ml.data.parsing.fasta import FastaDataset
from ms2ml.peptide import Peptide
from ms2ml.type_defs import PathLike

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
        file: PathLike,
        config: Config,
        only_unique: bool = True,
        enzyme: str = "trypsin",
        missed_cleavages: int = 2,
        allow_modifications: bool = False,
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
        self.allow_modifications = allow_modifications

    def parse(self) -> Iterator[Peptide]:
        charges = tuple(self.config.precursor_charges)
        num_outs = 0
        out_of_mz_range = 0
        skipped = 0
        seen = set()
        for pep_seq_dict in super().parse():
            for charge in charges:
                pep_seq_dict["charge"] = charge
                elem = (
                    pep_seq_dict if self.in_hook is None else self.in_hook(pep_seq_dict)
                )
                elem = self._to_elem(elem)

                # Filter based on mz
                low_mz = elem.mz < self.config.peptide_mz_range[0]
                high_mz = elem.mz > self.config.peptide_mz_range[1]
                if low_mz or high_mz:
                    out_of_mz_range += 1
                    continue

                if self.allow_modifications:
                    elem_lst = elem.get_variable_possible_mods()
                else:
                    elem_lst = [elem]

                if self.only_unique:
                    elem_keep = []
                    for elem in elem_lst:
                        if elem.to_proforma() not in seen:
                            seen.add(elem.to_proforma())
                            elem_keep.append(elem)

                    elem_lst = elem_keep

                for elem in elem_lst:
                    elem = elem if self.out_hook is None else self.out_hook(elem)
                    if elem is not None:
                        num_outs += 1
                        yield elem
                    else:
                        skipped += 1

        logger.info(
            f"Number of peptides: {num_outs}; "
            f"Number of peptides out of mz range: {out_of_mz_range},"
            f" Number of skipped peptides for other reasons: {skipped}"
        )

    def _to_elem(self, elem: dict) -> Peptide:
        pep = Peptide.from_proforma_seq(
            f"{elem['sequence']}/{elem['charge']}",
            config=self.config,
            extras=elem["header"],
        )
        return pep
