import sqlite3
import struct
import zlib
from typing import Iterator, Tuple

import numpy as np

from .base import BaseParser


def _decompress_peaks(
    compressed_mzs: bytes, compressed_int: bytes, num_peaks: int
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Decompress the m/z and intensity arrays from the Bibliospec format.

    The implementation of the compression is based on the C++ code in the
    Bibliospec source code.

    The compression is using concatenation of the intensity or mz arrray
    from a struct and then using zlib to compress the resulting byte array.

    Examples:
        >>> mzs = np.array([1, 2, 3, 4, 5])
        >>> intensities = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> compressed_mzs, compressed_int = _compress_peaks(mzs, intensities)
        >>> (compressed_mzs, compressed_int)
        (b'x\xdac`\x00\x81\x0f\xf6\x0c\x10\xe0\x00\xa18\xa0\xb4\x00\x94\x16q\x00\x006\x7f\x02\\',
         b'x\xdac`h\xb0g``p\x00" n\x00\xe2\x05\x0e\x00\x1b\x03\x03 ')
        >>> _decompress_peaks(compressed_mzs, compressed_int, 5)
        (array([1., 2., 3., 4., 5.]), array([1., 2., 3., 4., 5.]))
    """
    # [0, 15, 32, 47] could work ....maybe ...
    if len(compressed_mzs) != (8 * num_peaks):
        compressed_mzs = zlib.decompress(compressed_mzs, 32)

    mzs = struct.unpack("d" * num_peaks, compressed_mzs)

    if len(compressed_int) != (4 * num_peaks):
        compressed_int = zlib.decompress(compressed_int, 32)

    intensities = struct.unpack("f" * num_peaks, compressed_int)
    assert len(intensities) == len(mzs)

    return np.array(mzs), np.array(intensities)


def _compress_peaks(mzs: np.ndarray, intensities: np.ndarray) -> Tuple[bytes, bytes]:
    """Compress the m/z and intensity arrays to the Bibliospec format.

    The implementation of the compression is based on the C++ code in the
    Bibliospec source code.

    The compression is using concatenation of the intensity or mz arrray
    from a struct and then using zlib to compress the resulting byte array.
    """
    packed_mzs: bytes = struct.pack("d" * len(mzs), *mzs)
    packed_intensities: bytes = struct.pack("f" * len(intensities), *intensities)

    compressed_mzs = zlib.compress(packed_mzs, 9)
    compressed_int = zlib.compress(packed_intensities, 9)

    return compressed_mzs, compressed_int


class BiblioSpecParser(BaseParser):
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
        self.file = db_path

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

    def parse(self):
        """Parse the database.

        Returns
        -------
        Iterator
            Iterator over the parsed spectra
        """
        yield from self.parse_file(self.db_path)

    @classmethod
    def parse_file(cls, file) -> Iterator[dict]:
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

        query = f"SELECT {', '.join(BiblioSpecParser.EXTRACT_FIELDS)} FROM RefSpectra"
        query += " JOIN RefSpectraPeaks"
        query += " ON RefSpectra.id=RefSpectraPeaks.RefSpectraID"
        with sqlite3.connect(file) as conn:
            for row in conn.execute(query):
                row = dict(zip(BiblioSpecParser.EXTRACT_FIELDS, row))

                num_peaks = row["numPeaks"]
                compressed_mzs = row["peakMZ"]
                compressed_int = row["peakIntensity"]
                outs = _decompress_peaks(compressed_mzs, compressed_int, num_peaks)
                row["peakMZ"], row["peakIntensity"] = outs
                yield row

    def __iter__(self):
        return self.parse()
