from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from ms2ml.config import Config
from ms2ml.data.utils import pad_collate


class BaseAdapter(ABC):
    def __init__(
        self,
        config: Config,
        in_hook: Optional[Callable[..., Any]] = None,
        out_hook: Optional[Callable[..., Any]] = None,
        collate_fn: Optional[Callable[..., Any]] = pad_collate,
    ):
        """Provides a base class for adapters.

        Args:
            config (Config): Configuration object.
            in_hook (Callable): Function to be applied to each element before
                processing.
            out_hook (Callable): Function to be applied to each element after
                processing. This function can also be used as a filter, where
                you can make it return None if the element should be removed
                form the dataset.
            collate_fn (Callable): Function to be applied to a batch of
                elements before returning it. This function combines a list of
                (possibly nested) elements into a single element.
        """
        self.config = config
        self.in_hook = in_hook
        self.out_hook = out_hook
        self.collate_fn = collate_fn

    def _process_elem(self, elem):
        """Process an element.

        Combines the `in_hook` and `out_hook` functions.
        while using _to_elem in the middle to convert the element to a
        "package-internal" datatype.
        Not meant to be called directly.
        """
        elem = elem if self.in_hook is None else self.in_hook(elem)
        elem = self._to_elem(elem)

        # TODO implement here a way to filter out elements that are not
        #  compatible with the model
        elem = elem if self.out_hook is None else self.out_hook(elem)
        return elem

    def bundle(self, elems):
        """Bundle a list of elements into a single element.

        It removes all elements that are `None` from being combined
        """
        elems = [e for e in elems if e is not None]
        elems = elems if self.collate_fn is None else self.collate_fn(elems)
        return elems

    def batch(self, elems, batch_size):
        """Split a list of elements into batches of size `batch_size`.

        It internally removes elments that are `None` from being combined
        """
        batch = []
        for e in elems:
            if e is not None:
                batch.append(e)
            if len(batch) >= batch_size:
                yield self.bundle(batch)
                batch = []

        if len(batch) > 0:
            yield self.bundle(batch)

    @abstractmethod
    def _to_elem(self, elem: Any) -> Any:
        """Implements conversion to the intermediate object representation.

        This element will typically be an AnnotatedPeptideSpectrum object.
        But there is no real implementation reason by which it could not be
        something else.
        """
