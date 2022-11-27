import numpy as np
import pytest

from ms2ml.spectrum import Peptide, Spectrum


def _sample_annotated_spectra():
    sample_spectra = [
        """143.0814	24869	"b2/3.94ppm"
        272.123	20420	"b3/3.84ppm"
        359.1557	15797	"b4/0.92ppm"
        472.2363	14599	"b5/7.9ppm"
        756.3931	10386	"b7/9.09ppm"
        956.486	35345	"b9/-6.3ppm"
        1027.5098	15860	"b10/7.04ppm"
        1242.6079	40381	"b12/-0.73ppm"
        1313.634	17590	"b13/7.59ppm"
        1442.684	58765	"b14/2.11ppm"
        1555.7889	12996	"b15/-11.53ppm"
        1684.7969	41838	"b16/10.16ppm" """,
        """ 143.0816	1533755	"b2/3.09ppm"
        272.1245	277764	"b3/-1.66ppm"
        359.156	123107	"b4/-0.02ppm"
        600.2936	67887	"b6/9.03ppm"
        756.4019	225630	"b7/-2.53ppm"
        827.4403	158163	"b8/-4ppm"
        956.4791	382662	"b9/0.98ppm"
        1027.5178	218633	"b10/-0.8ppm"
        1128.5613	68407	"b11/2.41ppm"
        1242.6085	115289	"b12/-1.22ppm" """,
        """ 956.4833	25009	"b9/-3.49ppm"
        1442.6864	17228	"b14/0.41ppm"
        1684.8269	22282	"b16/-7.66ppm" """,
    ]

    proforma = [
        "AAESLQRAEATNAELER/2",
        "AAESLQRAEATNAELER/3",
        "AAESLQRAEATNAELER/2",
    ]

    sample_spectra = [
        [y.strip().split("\t") for y in x.split("\n")] for x in sample_spectra
    ]
    mzs = [np.array([y[0] for y in x], dtype=float) for x in sample_spectra]
    ints = [np.array([y[1] for y in x], dtype=float) for x in sample_spectra]
    annots = [[y[2] for y in x] for x in sample_spectra]
    return {"mzs": mzs, "ints": ints, "annots": annots, "proforma": proforma}


@pytest.fixture
def sample_annotated_spectra():
    return _sample_annotated_spectra()


def test_spectrum(sample_annotated_spectra):
    spec = Spectrum(
        mz=sample_annotated_spectra["mzs"][0],
        intensity=sample_annotated_spectra["ints"][0],
        precursor_mz=1000,
        precursor_charge=None,
        ms_level=2,
    )
    pep = Peptide.from_sequence(sample_annotated_spectra["proforma"][0])
    spec = spec.annotate(pep)

    for anno in spec.to_sus().annotation[0]:
        assert str(anno)[0] in ("b", "y")

    spec.plot()
    return spec


if __name__ == "__main__":
    spec = test_spectrum(sample_annotated_spectra=_sample_annotated_spectra())
