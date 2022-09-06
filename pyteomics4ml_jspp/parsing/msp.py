from typing import Iterator

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
        out = {"mz": items[0], "intensity": items[1], "annotation": items[2]}
        return out

    def peaks(self, items):
        out = {"mz": [], "intensity": [], "annotation": []}
        items = [item for item in items if item is not None]

        for item in items:
            out["mz"].append(item["mz"])
            out["intensity"].append(item["intensity"])
            out["annotation"].append(item["annotation"])

        return {k: np.array(v) for k, v in out.items()}


def _chunk_msp(file):
    """
    Chunk an MSP file into spectra

    It uses newlines as separators, so it is not very robust.
    """
    with open(file, "r") as f:
        chunk = []
        for line in f:
            if line.startswith("Name:"):
                chunk = []
                chunk.append(line)
                continue

            if len(chunk) > 0:
                chunk.append(line)
                if line.strip() == "":
                    yield chunk

        if len(chunk) > 0:
            # This is just in case the file does not end with a newline
            yield chunk


class MSPParser(BaseParser):
    """
    Implements a reader and parser for MSP files
    (and some other formats, such as .sptxt)
    """

    def __init__(self):
        self.transformer = _MSPTransformer()
        self.parser = Lark(
            _msp_grammar,
            start="start",
            parser="lalr",
            lexer="contextual",
            transformer=self.transformer,
            debug=True,
        )

    def parse_text(self, text: str) -> Iterator[dict]:
        """
        Parse an MSP file from a text input (string)
        """
        return self.parser.parse(text)

    def greedy_parse_file(self, file) -> Iterator[dict]:
        """
        Parse an MSP file from a text file.

        This option reads the whole file and parses it in
        memory, so it is not recommended for large files.

        Check the parse_file method for a more memory efficient
        approach.
        """
        with open(file) as f:
            out = self.parse_text(f.read())

        return out

    def parse_file(self, file) -> Iterator[dict]:
        """
        Parse an MSP file from a text file.

        This option reads the file line by line and parses
        each spectrum individually, so it is more memory efficient
        than the greedy_parse_file method.
        """
        for chunk in _chunk_msp(file):
            tmp = self.parse_text("".join(chunk))
            for spec in tmp:
                yield spec
