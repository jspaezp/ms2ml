"""Simple classes to annotate data.

Implements data classes that provide a consistent interface for.
annotated data.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AnnotatedIon:
    mass: float
    charge: int
    position: int
    ion_series: str
    intensity: float = 0
    neutral_loss: Optional[str] = None

    @property
    def fragment_positions(self):
        """Alias so the name corresponds to the config."""
        return self.position

    def asdict(self):
        """Return a dictionary representation of the object."""
        out = self.__dict__
        out.update(
            {"fragment_positions": self.fragment_positions, "ion_charges": self.charge}
        )
        return out

    def label(self, convention: str):
        """Generates the label for the ion based on the convention being passed.

        Examples
        --------
        >>> ion = AnnotatedIon(mass=123.2, charge=2, position=3, ion_series="z")
        >>> convention = "{ion_series}{fragment_positions}^{ion_charges}"
        >>> ion.label(convention)
        'z3^2'
        """

        return convention.format_map(self.asdict())


@dataclass
class RetentionTime:
    """Simple dataclass to hold retention time information."""

    rt: float
    units: str
    run: Optional[str] = None
