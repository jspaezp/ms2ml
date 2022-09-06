import numpy as np

from pyteomics4ml_jspp.adapters.msp import MSPAdapter
from pyteomics4ml_jspp.config import Config
from pyteomics4ml_jspp.spectrum import AnnotatedPeptideSpectrum

from torch.utils.data import default_collate
from torch import Tensor

def test_parsing_mgf_with_comments():
    text = (
        "### interact-20161213_NGHF_DBJ_SA_Exp3A_HeLa_1ug_60min_15000_02.sptxt"
        "  (Text version of interact-20161213_NGHF_DBJ_SA_"
        "Exp3A_HeLa_1ug_60min_15000_02.splib)\n"
        "### SpectraST (version 5.0, TPP v5.2.0 Flammagenitus,"
        " Build 201904252338-7913 (Linux-x86_64))\n"
        "### \n"
        "### IMPORT FROM PepXML "
        '"/data/interact-20161213_NGHF_DBJ_SA_Exp3A_HeLa_1ug_60min_15000_02.pep.xml",'
        ' (Comet against "2021-01-23-decoys-reviewed-contam-UP000005640.fas" (AA); ) '
        "[P=0.9;q=9999;n=;g=FALSE;o=FALSE;I=HCD;_RNT=0;_RDR=100000;"
        "_DCN=0;_NAA=6;_NPK=10;_MDF=9999;_CEN=FALSE;"
        "_XAN=FALSE;_BRK=FALSE;_BRM=FALSE;_IRT=;_IRR=FALSE]\n"
        "### ===\n"
        "Name: ASTSDYQVISDR/2\n"
        "LibID: 0\n"
        "MW: 1342.6354\n"
        "PrecursorMZ: 671.3177\n"
        "Status: Normal\n"
        "FullName: K.ASTSDYQVISDR.Q/2 (HCD)\n"
        "Comment: AvePrecursorMz=671.7060 BinaryFileOffset=401 "
        "CollisionEnergy=28.0 "
        "FracUnassigned=0.67,3/5;0.43,8/20;0.47,220/349 "
        "MassDiff=0.0012 Mods=0 NAA=12 NMC=0 NTT=2 Nreps=1/1 "
        "OrigMaxIntensity=2.1e+06 Parent=671.318 Pep=Tryptic "
        "PrecursorIntensity=5.1e+07 Prob=1.0000 "
        "Protein=1/sp|Q8NFH5|NUP35_HUMAN "
        "RawSpectrum=20161213_NGHF_DBJ_SA_Exp3A_HeLa_1ug_60min_15000_02.11507.11507 "
        "RetentionTime=600.1,600.1,600.1 "
        "Sample=1/_data_interact-20161213_NGHF"
        "_DBJ_SA_Exp3A_HeLa_1ug_60min_15000_02,1,1 "
        "Se=1^C1:pb=1.0000/0,fv=4.8531/0 Spec=Raw TotalIonCurrent=2.7e+07\n"
        "NumPeaks: 349\n"
        "101.0715	343.6	IQA/0.001	\n"
        "102.0550	95.3	IQAi/-0.016	\n"
        "129.1025	1859.5	IRA/-0.011,y1-46/-0.011	\n"
        "130.0867	659.1	IRAi/-0.027	\n"
        "\n"
    )

    msp_adapter = MSPAdapter(config=Config())
    parsed = msp_adapter.parse_text(text)

    elem = next(parsed)
    assert isinstance(elem, AnnotatedPeptideSpectrum)
    assert isinstance(elem.fragment_intensities, dict)
    assert len(elem.fragment_intensities) == 0


def test_parsing_mgf_from_prosit():
    text = (
        "Name: AFLYEIIDIGK/2\n"
        "MW: 1280.7017 \n"
        "Comment: Parent=641.3581 Mods=0 Modstring=AFLYEIIDIGK///2 iRT=137.54\n"
        "Num peaks: 24\n"
        '129.1019	15873	"y1-H2O/4.16ppm"\n'
        '147.1128	31837	"y1/1.3ppm"\n'
        '186.1232	11395	"y2-H2O/1.51ppm"\n'
        '204.1338	272863	"y2/0.81ppm"\n'
        "\n"
    )

    msp_adapter = MSPAdapter(config=Config())
    parsed = msp_adapter.parse_text(text)

    elem = next(parsed)
    assert isinstance(elem, AnnotatedPeptideSpectrum)
    assert isinstance(elem.fragment_intensities, dict)
    assert len(elem.fragment_intensities) == 2
    assert round(elem["y1^1"]) == 31837


def test_parsing_mgf_from_file(shared_datadir):
    my_file = shared_datadir / "head_FTMS_HCD_20_annotated_2019-11-12_filtered.msp"

    msp_adapter = MSPAdapter(config=Config())

    parsed = msp_adapter.parse_file(my_file)
    elem = next(parsed)
    assert isinstance(elem, AnnotatedPeptideSpectrum)
    assert isinstance(elem.fragment_intensities, dict)
    assert len(elem.fragment_intensities) == 19
    assert round(elem["y5^1"]) == 321534

    def post_hook(spec):
        return {
            "aa": spec.precursor_peptide.aa_to_vector(),
            "mods": spec.precursor_peptide.mod_to_vector(),
        }

    msp_adapter = MSPAdapter(config=Config(), out_hook=post_hook)

    parsed = msp_adapter.parse_file(my_file)
    elem = next(parsed)
    assert isinstance(elem, dict)
    assert isinstance(elem["aa"], np.ndarray)
    assert len(elem["aa"]) == 13

    # assert len(elem['aa'][0]) == 13
    assert len(elem["mods"]) == 13

    def post_hook(spec):
        return {
            "aa": spec.precursor_peptide.aa_to_onehot(),
            "mods": spec.precursor_peptide.mod_to_vector(),
        }

    msp_adapter = MSPAdapter(config=Config(), out_hook=post_hook)

    parsed = msp_adapter.parse_file(my_file)
    elem = next(parsed)
    assert isinstance(elem, dict)
    assert isinstance(elem["aa"], np.ndarray)
    assert len(elem["aa"]) == 13
    assert len(elem["aa"][0]) == 29
    assert len(elem["mods"]) == 13
    # assert len(elem['mods'][0]) == 13

    

def test_collating_MSPAdapter(shared_datadir):
    my_file = shared_datadir / "head_FTMS_HCD_20_annotated_2019-11-12_filtered.msp"
    def post_hook(spec):
        return {
            "aa": spec.precursor_peptide.aa_to_onehot(),
            "mods": spec.precursor_peptide.mod_to_vector(),
        }
    msp_adapter = MSPAdapter(config=Config(), out_hook=post_hook, collate_fn=default_collate)
    parsed = msp_adapter.parse_file(my_file)
    bundled = msp_adapter.bundle(parsed)

    assert isinstance(bundled, dict)
    assert isinstance(bundled["aa"], Tensor)

    assert len(bundled["aa"][0]) == 13
    assert len(bundled["aa"][0][0]) == 29
    assert len(bundled["mods"][0]) == 13

    assert len(bundled["aa"]) == 6
    assert len(bundled["mods"]) == 6