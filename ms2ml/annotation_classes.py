"""Simple classes to annotate data.

Implements data classes that provide a consistent interface for.
annotated data.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class AnnotatedIon:
    mass: float
    charge: int
    position: int
    ion_series: str
    intensity: float = 0
    neutral_loss: Optional[str] = None
    mass_error: Optional[float] = field(default=None, repr=False)
    mass_error_units: Optional[Literal["da", "ppm"]] = field(default=None, repr=False)

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


TimeLiteral = Literal["s", "seconds", "min", "minutes", "h", "hours"]


@dataclass
class RetentionTime:
    """Simple dataclass to hold retention time information.

    Examples:
        >>> rt = RetentionTime(rt=1.2, units="min")
        >>> rt.seconds()
        72.0
        >>> rt.minutes()
        1.2
        >>> rt = RetentionTime(rt=700, units="s")
        >>> rt.seconds()
        700
        >>> rt.minutes()
        11.666666666666666
    """

    rt: float
    units: TimeLiteral = "s"
    run: Optional[str] = None

    # TODO implement iRT conversion

    def __post_init__(self):
        """Converts the retention time to seconds."""
        self.rtinseconds = self.seconds()

    def seconds(self):
        """Converts the retention time to seconds."""
        if self.units == "s" or self.units == "seconds":
            return self.rt
        elif self.units.startswith("min"):
            return self.rt * 60
        elif self.units == "h":
            return self.rt * 60 * 60
        else:
            raise ValueError(
                f"Unknown units {self.units},"
                " must be one of s, min, h "
                "(international system approved units)"
            )

    def minutes(self):
        """Converts the retention time to minutes."""
        return self.seconds() / 60
