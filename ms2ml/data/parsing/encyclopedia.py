import sqlite3
import struct
import zlib
from typing import Iterator

import numpy as np
from loguru import logger

from ms2ml.spectrum import AnnotatedPeptideSpectrum

from .base import BaseParser

_ENTRIES_SCHEMA = [
    "CREATE TABLE entries ( PrecursorMz double not null, PrecursorCharge int not null, PeptideModSeq string not null, PeptideSeq string not null, Copies int not null, RTInSeconds double not null, Score double not null, MassEncodedLength int not null, MassArray blob not null, IntensityEncodedLength int not null, IntensityArray blob not null, CorrelationEncodedLength int, CorrelationArray blob, RTInSecondsStart double, RTInSecondsStop double, MedianChromatogramEncodedLength int, MedianChromatogramArray blob, SourceFile string not null );",  # noqa
    "CREATE INDEX 'PeptideModSeq_PrecursorCharge_SourceFile_Entries_index' on 'entries' ('PeptideModSeq' ASC, 'PrecursorCharge' ASC, 'SourceFile' ASC);",  # noqa
    "CREATE INDEX 'PeptideSeq_Entries_index' on 'entries' ('PeptideSeq' ASC);",
    "CREATE INDEX 'PrecursorMz_Entries_index' on 'entries' ('PrecursorMz' ASC);",
    "CREATE TABLE metadata ( Key string not null, Value string not null );",
    "CREATE INDEX 'Key_Metadata_index' on 'metadata' ('Key' ASC);",
    "INSERT INTO metadata ('Key', 'Value') VALUES ('EncyclopediaVersion', '1.12.34') ;",
    "INSERT INTO metadata ('Key', 'Value') VALUES ('version', '0.1.14') ;",
]


def _compress_array(array: np.ndarray, dtype: str) -> bytes:
    """Compress the array to the EncyclopeDIA format."""
    packed = struct.pack(">" + (dtype * len(array)), *array)
    compressed = zlib.compress(packed, 9)
    return compressed


def _extract_array(byte_array: bytes, type_str="d") -> np.ndarray:
    r"""Extract the array from the byte array.

    The type for masses is double and the rest if floats.
    Examples:
        >>> samp_mass = b"x\xda\xb3\xff\xc0\x00\x06\x0e\x0cP\x9a\x03J\x0b@i\x11\x08\r\x00D\xc4\x02\\"
        >>> _extract_array(samp_mass, "d")
        array([1., 2., 3., 4., 5.])
        >>> _extract_array(b"x\xda\xb3o``p`\x00b \xe1\x00b/``\x00\x00 \xa0\x03 ", "f")
        array([1., 2., 3., 4., 5.], dtype=float32)
    """  # noqa
    dtype = np.dtype(type_str)
    decompressed = zlib.decompress(byte_array, 32)
    decompressed_length = len(decompressed) // dtype.itemsize
    unpacked = struct.unpack(">" + (type_str * decompressed_length), decompressed)
    return np.array(unpacked, dtype=dtype)


class EncyclopeDIAParser(BaseParser):
    """Parser for EncyclopeDIA files.

    Parameters
    ----------
    db_path : str
        Path to the EncyclopeDIA .DLIB or .ELIB database
    """

    TABLES = (
        "entries",
        "peptidequenats",
        "peptidelocalizations",
        "peptidescores",
        "peptidetoprotein",
        "proteinscores",
        "retentiontimes",
        "metadata",
    )

    EXTRACT_FIELDS = (
        "PrecursorMz",
        "PrecursorCharge",
        "PeptideModSeq",
        "PeptideSeq",
        "Copies",
        "RTInSeconds",
        "Score",
        "MassEncodedLength",
        "MassArray",
        "IntensityEncodedLength",
        "IntensityArray",
        "CorrelationEncodedLength",
        "CorrelationArray",
        "RTInSecondsStart",
        "RTInSecondsStop",
        "MedianChromatogramEncodedLength",
        "MedianChromatogramArray",
        "SourceFile",
    )

    def __init__(self, db_path):
        self.db_path = db_path
        self.file = db_path

    def parse_text(self, text):
        """Raises an error"""
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

        query = f"SELECT {', '.join(EncyclopeDIAParser.EXTRACT_FIELDS)} FROM entries"
        with sqlite3.connect(file) as conn:
            for row in conn.execute(query):
                row = dict(zip(EncyclopeDIAParser.EXTRACT_FIELDS, row))

                row["MassArray"] = _extract_array(row.pop("MassArray"), "d")
                row["IntensityArray"] = _extract_array(row.pop("IntensityArray"), "f")

                assert np.all(row["MassArray"] > 0)
                assert len(row["MassArray"]) == len(row["IntensityArray"])

                if row["CorrelationArray"]:
                    row["CorrelationArray"] = _extract_array(
                        row.pop("CorrelationArray"), "f"
                    )
                if row["MedianChromatogramArray"]:
                    row["MedianChromatogramArray"] = _extract_array(
                        row.pop("MedianChromatogramArray"), "f"
                    )

                yield row

    def __iter__(self):
        return self.parse()


def write_encyclopedia(
    file,
    spectra: Iterator[AnnotatedPeptideSpectrum],
    source_file="ms2ml",
):
    """Write spectra to an EncyclopeDIA database.

    Parameters
    ----------
    file : PathLike
        Path
    spectra: Iterator[AnnotatedPeptideSpectrum]
        Iterator over the spectra to write
    source_file : str
        String depicting what to use as a source file in the .blib database
    """

    logger.debug("Writing EncyclopeDIA database to {}", file)
    con = sqlite3.connect(file)
    for x in _ENTRIES_SCHEMA:
        con.execute(x)
        con.commit()

    num_spectra = 0
    for spec in spectra:
        inp_dict = _spec_to_entry(spec, source_file=source_file)
        inp = tuple(inp_dict.values())
        con.execute(
            f"INSERT INTO entries ({', '.join(inp_dict.keys())})"
            f" VALUES({', '.join(['?' for x in inp_dict.keys()])})",
            inp,
        )
        con.commit()
        num_spectra += 1

    con.close()
    logger.info("Finished writing EncyclopeDIA database to {}", file)
    logger.info("Wrote {} spectra", num_spectra)


def _spec_to_entry(spec: AnnotatedPeptideSpectrum, source_file="ms2ml") -> dict:
    out = {
        "PrecursorMz": spec.precursor_mz,
        "PrecursorCharge": spec.precursor_charge,
        "PeptideModSeq": spec.precursor_peptide.to_massdiff_seq(),
        "PeptideSeq": spec.precursor_peptide.stripped_sequence,
        "Copies": 1,
        "RTInSeconds": spec.retention_time.seconds(),
        "Score": 1,
        "MassEncodedLength": len(spec.mz) * 8,
        "MassArray": _compress_array(spec.mz, "d"),
        "IntensityEncodedLength": len(spec.intensity) * 4,
        "IntensityArray": _compress_array(spec.intensity, "f"),
        "SourceFile": source_file,
    }

    if np.isnan(out["RTInSeconds"]):
        logger.debug("Spectrum {} has no retention time, using 0", spec)
        out["RTInSeconds"] = 0

    return out
