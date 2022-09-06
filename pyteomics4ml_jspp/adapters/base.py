from abc import ABC, abstractmethod
from typing import Any, Callable

from pyteomics4ml_jspp.config import Config


class BaseAdapter(ABC):
    def __init__(
        self,
        config: Config,
        in_hook: Callable = None,
        out_hook: Callable = None,
        collate_fn: Callable = None,
    ):
        self.config = config
        self.in_hook = in_hook
        self.out_hook = out_hook
        self.collate_fn = collate_fn
        super().__init__()

    def _process_elem(self, elem):
        elem = elem if self.in_hook is None else self.in_hook(elem)
        elem = self._to_elem(elem)
        elem = elem if self.out_hook is None else self.out_hook(elem)
        return elem

    def bundle(self, elems):
        elems = elems if self.collate_fn is None else self.collate_fn(list(elems))
        return elems

    def batch(self, elems, batch_size):
        batch = []
        for i, e in enumerate(elems):
            batch.append(e)
            if (i + 1) % batch_size == 0:
                yield self.bundle(batch)
                batch = []

        if len(batch) > 0:
            yield self.bundle(batch)

    @abstractmethod
    def _to_elem(self, elem: Any) -> Any:
        """Implements conversion to the intermediate object representation"""
        pass
