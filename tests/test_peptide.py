import pytest

from ms2ml import Config, Peptide


def test_peptide_raises_error_when_no_charge():
    config = Config()
    pep = Peptide.from_sequence("MPEP", config=config)

    with pytest.raises(ValueError) as error:
        pep.ion_dict

    assert "Peptide charge is not set" in str(error.value)

    config = Config()
    pep = Peptide.from_sequence("MPEP/2", config=config)
    pep.ion_dict
