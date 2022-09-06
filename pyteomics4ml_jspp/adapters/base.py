from abc import ABC, abstractmethod
from typing import Any, Callable

from pyteomics4ml_jspp.config import Config
from pyteomics4ml_jspp.spectrum import AnnotatedPeptideSpectrum


class BaseAdapter(ABC):
    def __init__(
        self, config: Config, in_hook: Callable = None, out_hook: Callable = None
    ):
        self.config = config
        self.in_hook = in_hook
        self.out_hook = out_hook
        super().__init__()

    def _process_spec(self, spec):
        spec = spec if self.in_hook is None else self.in_hook(spec)
        spec = self._to_spec(spec, config=self.config)
        spec = spec if self.out_hook is None else self.out_hook(spec)
        return spec

    @abstractmethod
    def _to_spec(spec: Any, config: Config) -> AnnotatedPeptideSpectrum:
        """Implements conversion to AnnotatedPeptideSpectrum"""
        pass
