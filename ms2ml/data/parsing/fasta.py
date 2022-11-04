import logging
from os import PathLike
from pathlib import Path
from typing import Iterator

from pyteomics import fasta, parser

from ms2ml.data.parsing import BaseParser

logger = logging.getLogger(__name__)


class FastaDataset(BaseParser):
    def __init__(
        self,
        file: PathLike,
        enzyme: str = "trypsin",
        missed_cleavages: int = 2,
        min_length: int = 5,
        max_length: int = 30,
        only_unique: bool = False,
    ) -> None:
        """FastaDataset constructor.

        Parameters:
            file: Path to the FASTA file to parse.
            enzyme: Enzyme used to cleave the peptides.
            missed_cleavages: Number of missed cleavages allowed.
            min_length: Minimum length of the peptides.
            max_length: Maximum length of the peptides.
            only_unique: Whether to only keep unique peptides.

        """
        self.file = Path(file)
        self.enzyme = enzyme
        self.missed_cleavages = missed_cleavages
        self.min_length = min_length
        self.max_length = max_length
        self.only_unique = only_unique
        super().__init__()

    def parse_file(self, file: PathLike) -> Iterator[dict]:
        """Parses the FASTA file and returns a generator of peptides.

        Peptides are returned as a dictionary with the following keys:
        - sequence: the peptide sequence
        - header: the header of the FASTA entry

        The generator yields peptides one by one.
        Renders peptides in alphabetical order within each protein.
        And in order of occurence within the FASTA file.

        """
        logger.info(
            f"Processing file {file},"
            f" with enzyme={self.enzyme}, "
            f" missed_cleavages={self.missed_cleavages}"
            f" min_length={self.min_length}"
            f" max_length={self.max_length}"
        )

        yielded_peps = {}
        peptides_count = 0
        with open(file) as f:
            for description, sequence in fasta.FASTA(f):
                new_peptides = parser.cleave(
                    sequence,
                    rule=self.enzyme,
                    missed_cleavages=self.missed_cleavages,
                    min_length=self.min_length,
                    max_length=self.max_length,
                )
                for x in sorted(list(new_peptides), reverse=True):
                    if self.only_unique:
                        if x in yielded_peps:
                            continue
                        else:
                            yielded_peps[x] = True
                    peptides_count += 1
                    yield {"sequence": x, "header": description}

        logger.info("Done, %i sequences", peptides_count)

    def parse(self):
        yield from self.parse_file(self.file)
