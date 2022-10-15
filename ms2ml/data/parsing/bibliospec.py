import sqlite3
import struct
import zlib
from typing import Iterator, Tuple

import numpy as np

from .base import BaseParser


def _decompress_peaks(
    compressed_mzs: bytes, compressed_int: bytes, num_peaks: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompress the m/z and intensity arrays from the Bibliospec format.

    The implementation of the compression is based on the C++ code in the
    Bibliospec source code.

    The compression is using concatenation of the intensity or mz arrray
    from a struct and then using zlib to compress the resulting byte array.
    """
    # [0, 15, 32, 47] could work ....maybe ...
    if len(compressed_mzs) != (4 * num_peaks):
        compressed_mzs = zlib.decompress(compressed_mzs, 32)

    mzs = struct.unpack("d" * num_peaks, compressed_mzs)

    if len(compressed_int) != (4 * num_peaks):
        compressed_int = zlib.decompress(compressed_int, 32)

    intensities = struct.unpack("f" * num_peaks, compressed_int)
    assert len(intensities) == len(mzs)

    return np.array(mzs), np.array(intensities)


class BibliosPecParser(BaseParser):
    """Parser for BibliosPec files.

    The bibliospec .blib format has these fields and tables:

        sqlite> pragma table_info(Modifications);
        "id"
        "RefSpectraID"
        "position"
        "mass"

        sqlite> pragma table_info(RefSpectra);
        "id",
        "peptideSeq",
        "precursorMZ",
        "precursorCharge",
        "peptideModSeq",
        "copies",
        "numPeaks",
        "ionMobility",
        "collisionalCrossSectionSqA",
        "ionMobilityHighEnergyOffset",
        "ionMobilityType",
        "retentionTime",
        "startTime",
        "endTime",
        "totalIonCurrent",
        "moleculeName",
        "chemicalFormula",
        "precursorAdduct",
        "inchiKey",
        "otherKeys",
        "fileID",
        "SpecIDinFile",
        "score",
        "scoreType",

        sqlite> pragma table_info(RefSpectraPeaks);
        "RefSpectraID",
        "peakMZ",
        "peakIntensity",

        sqlite> pragma table_info(RefSpectraPeakAnnotations);
        "id",
        "RefSpectraID",
        "peakIndex",
        "name",
        "formula",
        "inchiKey",
        "otherKeys",
        "charge",
        "adduct",
        "comment",
        "mzTheoretical",
        "mzObserved",

    Parameters
    ----------
    db_path : str
        Path to the BibliosPec database
    """

    ALL_SPECTRA_CMD = ""
    TABLES = (
        "IonMobilityTypes",
        "LibInfo",
        "Modifications",
        "Proteins",
        "RefSpectra",
        "RefSpectraPeakAnnotations",
        "RefSpectraPeaks",
        "RefSpectraProteins",
        "ScoreTypes",
        "SpectrumSourceFiles",
    )

    EXTRACT_FIELDS = (
        "id",
        "peptideSeq",
        "precursorMZ",
        "precursorCharge",
        "peptideModSeq",
        "copies",
        "numPeaks",
        "ionMobility",
        "collisionalCrossSectionSqA",
        "ionMobilityHighEnergyOffset",
        "ionMobilityType",
        "retentionTime",
        "startTime",
        "endTime",
        "totalIonCurrent",
        "moleculeName",
        "score",
        "scoreType",
        "peakMZ",
        "peakIntensity",
    )

    def __init__(self, db_path):
        self.db_path = db_path

    def parse_file(self, file):
        """Parse a file.

        Parameters
        ----------
        file : PathLike
            Path to the file to parse

        Returns
        -------
        Iterator
            Iterator over the parsed spectra
        """
        raise NotImplementedError

    def parse_text(self, text):
        """Parse a chunk of text.

        Currently not implemented.

        Parameters
        ----------
        text : str
            Chunk of text to parse

        Returns
        -------
        Iterator
            Iterator over the parsed spectra
        """
        raise NotImplementedError

    def parse(self) -> Iterator[dict]:
        """Parse the database.

        Returns
        -------
        Iterator
            Iterator over the parsed spectra
        """

        query = f"SELECT {', '.join(self.EXTRACT_FIELDS)} FROM RefSpectra"
        query += " JOIN RefSpectraPeaks"
        query += " ON RefSpectra.id=RefSpectraPeaks.RefSpectraID"
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute(query):
                row = dict(zip(self.EXTRACT_FIELDS, row))

                num_peaks = row["numPeaks"]
                compressed_mzs = row["peakMZ"]
                compressed_int = row["peakIntensity"]
                outs = _decompress_peaks(compressed_mzs, compressed_int, num_peaks)
                row["peakMZ"], row["peakIntensity"] = outs
                yield row

    def __iter__(self):
        return self.parse()

    def __next__(self):
        return next(self.parse())
