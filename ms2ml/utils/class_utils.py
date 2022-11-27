from functools import wraps


def lazy(func):
    """Decorator that makes a property lazy-evaluated."""

    attr_name = f"_lazy_{func.__name__}"

    @property
    @wraps(func)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property


def clear_lazy_cache(self):
    """Clears the lazy cache of the object.

    This method is intended to be used when a modification is made
    in-place on classes that use the @lazy decorator for properties.
    """
    for attr in dir(self):
        if attr.startswith("_lazy_"):
            delattr(self, attr)
