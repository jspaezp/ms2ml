
# Encodings to tensors

When getting the data into your model, you usually will have to process
it to some form of structured numeric input.

There is a plethora fo ways to encode information. And I will show a
couple (and how to get them using ms2ml)

## Encoding Peptides

``` python
import numpy as np
from ms2ml.peptide import Peptide

p = Peptide.from_sequence("MYPEPTIDE")
print(p)
```

    Peptide.from_sequence(MYPEPTIDE)

### Counting

``` python
count = p.aa_to_count()
print(count)
```

    [1 0 0 0 1 2 0 0 0 1 0 0 0 1 0 0 2 0 0 0 1 0 0 0 0 1 0 1 0]

### One-hot encoding

``` python
oh = p.aa_to_onehot()
# Note that there is an equivalent method for the modifications
# p.mod_to_onehot()
print(oh)
```

    [[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]]

### Vectorizing

``` python
vec = p.aa_to_vector()
print(vec)
```

    [ 0 13 25 16  5 16 20  9  4  5 27]

## Encoding Spectra

``` python
from ms2ml.spectrum import Spectrum
from ms2ml.config import Config

spectrum = Spectrum(
    mz=np.array([1000.0, 1500.0, 2000.0]),
    intensity=np.array([1.0, 2.0, 3.0]),
    ms_level=2,
)
print(spectrum)
```

    Spectrum(mz=array([1000., 1500., 2000.]), intensity=array([1., 2., 3.]), ms_level=2, precursor_mz=0.0, precursor_charge=0, instrument=None, analyzer=None, extras={})

## Encoding annotated spectra

``` python
from ms2ml.spectrum import AnnotatedPeptideSpectrum
config = Config()
peptide = Peptide.from_sequence("PEPPINK/2", config)
spectrum = AnnotatedPeptideSpectrum(
    mz=np.array([50.0, 147.11333, 1000.0, 1500.0, 2000.0]),
    intensity=np.array([50.0, 200.0, 1.0, 2.0, 3.0]),
    ms_level=2,
    precursor_peptide=peptide,
)
spectrum
print(spectrum)
```

    AnnotatedPeptideSpectrum(mz=array([  50.     ,  147.11333, 1000.     , 1500.     , 2000.     ]), intensity=array([ 50., 200.,   1.,   2.,   3.]), ms_level=2, precursor_mz=0.0, precursor_charge=0, instrument=None, analyzer=None, extras={}, precursor_peptide=Peptide([('P', None), ('E', None), ('P', None), ('P', None), ('I', None), ('N', None), ('K', None)], {'n_term': None, 'c_term': None, 'unlocalized_modifications': [], 'labile_modifications': [], 'fixed_modifications': [], 'intervals': [], 'isotopes': [], 'group_ids': [], 'charge_state': ChargeState(2, [])}), precursor_isotope=0)
