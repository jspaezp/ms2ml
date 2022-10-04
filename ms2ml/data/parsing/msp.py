from typing import Any, ClassVar, Dict, Iterator, List, Optional, TypedDict

import numpy as np
from lark import Lark, Transformer

from .base import BaseParser

_msp_grammar = r"""
    start: spectrum+

    spectrum: header peaks NEWLINE?

    header: pair+

    pair : name PAIR_SEPARATOR (content SPACE*)+ NEWLINE
    PAIR_SEPARATOR: ":" SPACE+
    content : string

    SIGNED_NUMBER: ["+" | "-"]? NUMBER

    name : /\w+([ ]\w+)*/
    string : /\S+/

    peaks: (peak NEWLINE)+

    peak_annotation : SPACE string
    peak: SIGNED_NUMBER SPACE SIGNED_NUMBER peak_annotation SPACE?

    NEWLINE: (CR? LF)

    %import common.NUMBER
    %import common.CR
    %import common.LF
    %import common.WS_INLINE -> SPACE

    COMMENT: /#.*\n/
    %ignore COMMENT
    """


def _is_number_convertible(x):
    x = x.replace("e+", "", 1)
    x = x.replace("e-", "", 1)
    x = x.replace(".", "", 1)

    return x.isdigit()


class PeakDict(TypedDict):
    mz: float
    intensity: float
    annotation: str


class _MSPTransformer(Transformer):
    def start(self, items):
        return list(items)

    def SIGNED_NUMBER(self, items):
        return float(items)

    def PAIR_SEPARATOR(self, items):
        pass

    def pair(self, items):
        items = [item for item in items if item is not None]
        content = items[1:]
        if len(content) == 1:
            content = content[0]
        return {items[0]: content}

    def string(self, items):
        items = [item for item in items if item is not None]
        return str(items[0])

    def name(self, items):
        return self.string(items)

    def SPACE(self, items):
        pass

    def NEWLINE(self, items):
        pass

    def spectrum(self, items):
        out = {"header": items[0], "peaks": items[1]}
        return out

    def header(self, items):
        out = {}
        for item in items:
            out.update(item)

        if "Comment" in out:
            main_comment = {}
            comment_extras = []

            for c in out["Comment"]:
                if "=" in c:
                    kv = c.split("=")
                    if len(kv) == 2:
                        k, v = kv
                        if _is_number_convertible(v):
                            v = float(v)

                        main_comment[k] = v
                    else:
                        comment_extras.append(c)
                else:
                    comment_extras.append(c)

            main_comment["EXTRAS"] = comment_extras
            out["Comment"] = main_comment
        return out

    def content(self, items):
        items = [item for item in items if item is not None]
        if len(items) == 1:
            out = str(items[0])
        else:
            raise ValueError()
        return out

    def peak_annotation(self, items):
        # First element is space
        return items[1]

    def peak(self, items):
        items = [item for item in items if item is not None]
        out = PeakDict(mz=items[0], intensity=items[1], annotation=items[2])
        return out

    def peaks(self, items: List[Optional[PeakDict]]) -> Dict[str, np.ndarray]:
        nnone_items: List[PeakDict] = [item for item in items if item is not None]
        mzs: list[float] = [item["mz"] for item in nnone_items]
        intensities: list[float] = [item["intensity"] for item in nnone_items]
        annotations: list[str] = [item["annotation"] for item in nnone_items]

        out = {
            "mz": np.array(mzs),
            "intensity": np.array(intensities),
            "annotation": np.array(annotations),
        }

        return out


class _MSPLark(Lark):
    def __init__(self):
        transformer = _MSPTransformer()
        super().__init__(
            _msp_grammar,
            start="start",
            parser="lalr",
            lexer="contextual",
            transformer=transformer,
            debug=True,
        )

    def parse2(self, text: str, start: str = None, *args, **kwargs) -> Iterator[Dict]:
        out: Iterator[Dict] = super().parse(
            text, start, *args, **kwargs
        )  # type: ignore[assignment]
        return out


def _chunk_msp(file):
    """Chunk an MSP file into spectra.

    It uses newlines as separators, so it is not very robust.
    """
    with open(file) as f:
        chunk = []
        for line in f:
            if line.startswith("Name:"):
                chunk = []
                chunk.append(line)
                continue

            if len(chunk) > 0:
                chunk.append(line)
                if line.strip() == "":
                    out = chunk
                    chunk = []
                    yield out

        if len(chunk) > 0:
            # This is just in case the file does not end with a newline
            yield chunk


class MSPParser(BaseParser):
    """Implements a reader and parser for MSP files.

    (and some other formats, such as .sptxt)
    """

    parser: ClassVar = _MSPLark()

    def __init__(self, file=None):
        self.file = file

    @staticmethod
    def parse_text(text: str) -> Iterator[Dict[str, Any]]:
        """Parse an MSP file from a text input (string)."""
        return MSPParser.parser.parse2(text)

    @staticmethod
    def parse_file(file) -> Iterator[Dict[str, Any]]:
        """Parse an MSP file from a text file.

        This option reads the file line by line and parses
        each spectrum individually, so it is more memory efficient
        than the greedy_parse_file method.
        """
        for chunk in _chunk_msp(file):
            tmp = MSPParser.parse_text("".join(chunk))
            yield from tmp

    def parse(self) -> Iterator[dict]:
        """Parse an MSP file."""
        if self.file is None:
            msg = "No file specified when initializing the parser."
            msg += f" {type(self)}"
            raise ValueError()
        return self.parse_file(self.file)
