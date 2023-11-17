---
title: Encodings to tensors
toc-title: Table of contents
---

When getting the data into your model, you usually will have to process
it to some form of structured numeric input.

There is a plethora fo ways to encode information. And I will show a
couple (and how to get them using ms2ml)

## Encoding Peptides

::: {.cell execution_count="1"}
``` {.python .cell-code}
import numpy as np
from ms2ml.peptide import Peptide

p = Peptide.from_sequence("MYPEPTIDE")
print(p)
```

::: {.cell-output .cell-output-stdout}
    Peptide.from_sequence('MYPEPTIDE')
:::

::: {.cell-output .cell-output-stderr}
    /Users/sebastianpaez/Library/Caches/pypoetry/virtualenvs/ms2ml-zJc7qiTs-py3.9/lib/python3.9/site-packages/ms2ml/config.py:392: UserWarning: Using default config. Consider creating your own.
      warnings.warn("Using default config. Consider creating your own.", UserWarning)
:::
:::

### Counting

::: {.cell execution_count="2"}
``` {.python .cell-code}
count = p.aa_to_count()
print(count)
```

::: {.cell-output .cell-output-stdout}
    [1 0 0 0 1 2 0 0 0 1 0 0 0 1 0 0 2 0 0 0 1 0 0 0 0 1 0 1 0]
:::
:::

### One-hot encoding

::: {.cell execution_count="3"}
``` {.python .cell-code}
oh = p.aa_to_onehot()
# Note that there is an equivalent method for the modifications
# p.mod_to_onehot()
print(oh)
```

::: {.cell-output .cell-output-stdout}
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
:::
:::

### Vectorizing

::: {.cell execution_count="4"}
``` {.python .cell-code}
vec = p.aa_to_vector()
print(vec)
```

::: {.cell-output .cell-output-stdout}
    [ 0 13 25 16  5 16 20  9  4  5 27]
:::
:::

## Encoding Spectra

::: {.cell execution_count="5"}
``` {.python .cell-code}
from ms2ml.spectrum import Spectrum
from ms2ml.config import Config

spectrum = Spectrum(
    mz=np.array([1000.0, 1500.0, 2000.0]),
    intensity=np.array([1.0, 2.0, 3.0]),
    ms_level=2,
    precursor_mz=1000.0,
)
print(spectrum)
```

::: {.cell-output .cell-output-stdout}
    Spectrum(mz=array([1000., 1500., 2000.]), intensity=array([1., 2., 3.]), ms_level=2, precursor_mz=1000.0, precursor_charge=None, instrument=None, analyzer=None, collision_energy=nan, activation=None, extras={}, retention_time=RetentionTime(rt=nan, units='seconds', run=None), precursor_ion_mobility=None)
:::
:::

### Binning

::: {.cell execution_count="6"}
``` {.python .cell-code}
binned = spectrum.bin_spectrum(
    start = 500.0,
    end = 1500.0,
    binsize = 100.0, # For the real world this would be 1 or 0.02 ...
)

# or ....

binned = spectrum.bin_spectrum(
    start = 500.0,
    end = 1500.0,
    n_bins = 10,
)
print(binned)
```

::: {.cell-output .cell-output-stdout}
    [0. 0. 0. 0. 1. 0. 0. 0. 2.]
:::
:::

::: {.cell execution_count="7"}
``` {.python .cell-code}
# or ....
binned = spectrum.bin_spectrum(
    start = -500.0,
    end = +500.0,
    binsize = 100.0,
    relative = True, # Will make all bins relative to the precursor mz
)

# or ....
binned = spectrum.bin_spectrum(
    start = -500.0,
    end = +500.0,
    binsize = 100.0,
    relative = 600, # Will make all bins relative to 600!
    get_breaks = True
)
binned_spectrum, bin_breaks = binned
print(binned_spectrum)
print(bin_breaks)
```

::: {.cell-output .cell-output-stdout}
    [0. 0. 1. 0. 0. 0. 0. 2. 0.]
    [ 100.          211.11111111  322.22222222  433.33333333  544.44444444
      655.55555556  766.66666667  877.77777778  988.88888889 1100.        ]
:::
:::

## Encoding annotated spectra

::: {.cell execution_count="8"}
``` {.python .cell-code}
from ms2ml.spectrum import AnnotatedPeptideSpectrum
config = Config()
peptide = Peptide.from_sequence("PEPPINK/2", config)
spectrum = AnnotatedPeptideSpectrum(
    mz=np.array([50.0, 147.11333, 1000.0, 1500.0, 2000.0]),
    intensity=np.array([50.0, 200.0, 1.0, 2.0, 3.0]),
    ms_level=2,
    precursor_peptide=peptide,
    precursor_mz=147.11333,
)
spectrum
print(spectrum)
```

::: {.cell-output .cell-output-stdout}
    AnnotatedPeptideSpectrum(mz=array([  50.     ,  147.11333, 1000.     , 1500.     , 2000.     ]), intensity=array([ 50., 200.,   1.,   2.,   3.]), ms_level=2, precursor_mz=147.11333, precursor_charge=2, instrument=None, analyzer=None, collision_energy=nan, activation=None, extras={}, retention_time=RetentionTime(rt=nan, units='seconds', run=None), precursor_ion_mobility=None, precursor_peptide=Peptide([('P', None), ('E', None), ('P', None), ('P', None), ('I', None), ('N', None), ('K', None)], {'n_term': None, 'c_term': None, 'unlocalized_modifications': [], 'labile_modifications': [], 'fixed_modifications': [], 'intervals': [], 'isotopes': [], 'group_ids': [], 'charge_state': ChargeState(2, [])}), precursor_isotope=0)
:::
:::

::: {.cell execution_count="9"}
``` {.python .cell-code}
spectrum.encode_fragments()
```

::: {.cell-output .cell-output-stderr}
    2023-10-27 15:44:47.839 | DEBUG    | ms2ml.proforma_utils:mod_id_mass:117 - Resolving mod_id: 21
    2023-10-27 15:44:47.839 | DEBUG    | ms2ml.proforma_utils:solver:84 - Initializing UnimodResolver
    2023-10-27 15:44:48.048 | DEBUG    | ms2ml.proforma_utils:mod_id_mass:119 - Resolved to {'date_time_modified': datetime.datetime(2018, 8, 13, 13, 42, 59), 'date_time_posted': datetime.datetime(2002, 8, 19, 19, 17, 11), 'avge_mass': 79.9799, 'mono_mass': 79.966331, 'composition': Composition({'H': 1, 'O': 3, 'P': 1}), 'record_id': 21, 'approved': True, 'title': 'Phospho', 'full_name': 'Phosphorylation', 'username_of_poster': 'unimod', 'group_of_poster': 'admin', 'specificity': [{'hidden': True, 'spec_group': 8, 'site': 'E', 'position': 'Anywhere', 'classification': 'Post-translational'}, {'hidden': True, 'spec_group': 6, 'site': 'R', 'position': 'Anywhere', 'classification': 'Post-translational'}, {'hidden': True, 'spec_group': 7, 'site': 'K', 'position': 'Anywhere', 'classification': 'Post-translational'}, {'hidden': True, 'spec_group': 4, 'site': 'H', 'position': 'Anywhere', 'classification': 'Post-translational', 'note': 'Rare'}, {'hidden': True, 'spec_group': 5, 'site': 'C', 'position': 'Anywhere', 'classification': 'Post-translational', 'note': 'Rare'}, {'hidden': True, 'spec_group': 3, 'site': 'D', 'position': 'Anywhere', 'classification': 'Post-translational', 'note': 'Rare'}, {'hidden': False, 'spec_group': 2, 'site': 'Y', 'position': 'Anywhere', 'classification': 'Post-translational'}, {'hidden': False, 'spec_group': 1, 'site': 'T', 'position': 'Anywhere', 'classification': 'Post-translational'}, {'hidden': False, 'spec_group': 1, 'site': 'S', 'position': 'Anywhere', 'classification': 'Post-translational'}], 'refs': [{'text': 'AA0036', 'source': 'RESID', 'url': None}, {'text': 'IonSource', 'source': 'Misc. URL', 'url': 'http://www.ionsource.com/Card/phos/phos.htm'}, {'text': 'AA0037', 'source': 'RESID', 'url': None}, {'text': 'AA0033', 'source': 'RESID', 'url': None}, {'text': 'AA0038', 'source': 'RESID', 'url': None}, {'text': 'AA0039', 'source': 'RESID', 'url': None}, {'text': 'AA0222', 'source': 'RESID', 'url': None}, {'text': 'PHOS', 'source': 'FindMod', 'url': None}, {'text': 'AA0034', 'source': 'RESID', 'url': None}, {'text': 'AA0035', 'source': 'RESID', 'url': None}]}
    2023-10-27 15:44:48.048 | DEBUG    | ms2ml.proforma_utils:mod_id_mass:117 - Resolving mod_id: 7
    2023-10-27 15:44:48.048 | DEBUG    | ms2ml.proforma_utils:mod_id_mass:119 - Resolved to {'date_time_modified': datetime.datetime(2018, 10, 25, 9, 32, 26), 'date_time_posted': datetime.datetime(2002, 8, 19, 19, 17, 11), 'avge_mass': 0.9848, 'mono_mass': 0.984016, 'composition': Composition({'H': -1, 'N': -1, 'O': 1}), 'record_id': 7, 'approved': False, 'title': 'Deamidated', 'full_name': 'Deamidation', 'username_of_poster': 'unimod', 'group_of_poster': 'admin', 'specificity': [{'hidden': False, 'spec_group': 1, 'site': 'Q', 'position': 'Anywhere', 'classification': 'Artefact'}, {'hidden': True, 'spec_group': 2, 'site': 'R', 'position': 'Anywhere', 'classification': 'Post-translational', 'note': 'Protein which is post-translationally modified by the de-imination of one or more arginine residues; Peptidylarginine deiminase (PAD) converts protein bound to citrulline'}, {'hidden': False, 'spec_group': 1, 'site': 'N', 'position': 'Anywhere', 'classification': 'Artefact', 'note': 'Convertion of glycosylated asparagine residues upon deglycosylation with PNGase F in H2O'}, {'hidden': True, 'spec_group': 3, 'site': 'F', 'position': 'Protein N-term', 'classification': 'Post-translational'}], 'alt_names': ['phenyllactyl from N-term Phe', 'Citrullination'], 'refs': [{'text': '6838602', 'source': 'PubMed PMID', 'url': None}, {'text': 'AA0214', 'source': 'RESID', 'url': None}, {'text': 'IonSource tutorial', 'source': 'Misc. URL', 'url': 'http://www.ionsource.com/Card/Deamidation/deamidation.htm'}, {'text': 'CITR', 'source': 'FindMod', 'url': None}, {'text': 'AA0128', 'source': 'RESID', 'url': None}, {'text': 'FLAC', 'source': 'FindMod', 'url': None}, {'text': '15700232', 'source': 'PubMed PMID', 'url': None}, {'text': 'DEAM', 'source': 'FindMod', 'url': None}]}
    2023-10-27 15:44:48.049 | DEBUG    | ms2ml.proforma_utils:mod_id_mass:117 - Resolving mod_id: 1
    2023-10-27 15:44:48.049 | DEBUG    | ms2ml.proforma_utils:mod_id_mass:119 - Resolved to {'date_time_modified': datetime.datetime(2017, 11, 8, 16, 8, 56), 'date_time_posted': datetime.datetime(2002, 8, 19, 19, 17, 11), 'avge_mass': 42.0367, 'mono_mass': 42.010565, 'composition': Composition({'H': 2, 'C': 2, 'O': 1}), 'record_id': 1, 'approved': True, 'title': 'Acetyl', 'full_name': 'Acetylation', 'username_of_poster': 'unimod', 'group_of_poster': 'admin', 'specificity': [{'hidden': True, 'spec_group': 6, 'site': 'T', 'position': 'Anywhere', 'classification': 'Post-translational'}, {'hidden': False, 'spec_group': 5, 'site': 'N-term', 'position': 'Protein N-term', 'classification': 'Post-translational'}, {'hidden': True, 'spec_group': 4, 'site': 'S', 'position': 'Anywhere', 'classification': 'Post-translational'}, {'hidden': True, 'spec_group': 3, 'site': 'C', 'position': 'Anywhere', 'classification': 'Post-translational'}, {'hidden': False, 'spec_group': 2, 'site': 'N-term', 'position': 'Any N-term', 'classification': 'Multiple', 'note': 'GIST acetyl light'}, {'hidden': False, 'spec_group': 1, 'site': 'K', 'position': 'Anywhere', 'classification': 'Multiple', 'note': 'PT and GIST acetyl light'}, {'hidden': True, 'spec_group': 7, 'site': 'Y', 'position': 'Anywhere', 'classification': 'Chemical derivative', 'note': 'O-acetyl'}, {'hidden': True, 'spec_group': 8, 'site': 'H', 'position': 'Anywhere', 'classification': 'Chemical derivative'}, {'hidden': True, 'spec_group': 9, 'site': 'R', 'position': 'Anywhere', 'classification': 'Artefact', 'note': 'glyoxal-derived hydroimidazolone'}], 'refs': [{'text': 'AA0048', 'source': 'RESID', 'url': None}, {'text': 'AA0049', 'source': 'RESID', 'url': None}, {'text': 'AA0041', 'source': 'RESID', 'url': None}, {'text': 'AA0052', 'source': 'RESID', 'url': None}, {'text': 'AA0364', 'source': 'RESID', 'url': None}, {'text': 'AA0056', 'source': 'RESID', 'url': None}, {'text': 'AA0046', 'source': 'RESID', 'url': None}, {'text': 'AA0051', 'source': 'RESID', 'url': None}, {'text': 'AA0045', 'source': 'RESID', 'url': None}, {'text': 'AA0354', 'source': 'RESID', 'url': None}, {'text': 'AA0044', 'source': 'RESID', 'url': None}, {'text': 'AA0043', 'source': 'RESID', 'url': None}, {'text': '11999733', 'source': 'PubMed PMID', 'url': None}, {'text': 'Chemical Reagents for Protein Modification 3rd edition, pp 215-221, Roger L. Lundblad, CRC Press, New York, N.Y., 2005', 'source': 'Book', 'url': None}, {'text': 'IonSource acetylation tutorial', 'source': 'Misc. URL', 'url': 'http://www.ionsource.com/Card/acetylation/acetylation.htm'}, {'text': 'AA0055', 'source': 'RESID', 'url': None}, {'text': '14730666', 'source': 'PubMed PMID', 'url': None}, {'text': '15350136', 'source': 'PubMed PMID', 'url': None}, {'text': 'AA0047', 'source': 'RESID', 'url': None}, {'text': '12175151', 'source': 'PubMed PMID', 'url': None}, {'text': '11857757', 'source': 'PubMed PMID', 'url': None}, {'text': 'AA0042', 'source': 'RESID', 'url': None}, {'text': 'AA0050', 'source': 'RESID', 'url': None}, {'text': 'AA0053', 'source': 'RESID', 'url': None}, {'text': 'AA0054', 'source': 'RESID', 'url': None}, {'text': 'ACET', 'source': 'FindMod', 'url': None}, {'text': 'PNAS 2006 103: 18574-18579', 'source': 'Journal', 'url': 'http://dx.doi.org/10.1073/pnas.0608995103'}]}
:::

::: {.cell-output .cell-output-display execution_count="9"}
    array([200.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
          dtype=float32)
:::
:::

::: {.cell execution_count="10"}
``` {.python .cell-code}
print("First 5: ", spectrum.fragment_labels[:5])
print("Last 5:", spectrum.fragment_labels[-5:])
print("Total Number:", len(spectrum.fragment_labels))
```

::: {.cell-output .cell-output-stdout}
    First 5:  ['y1^1', 'y1^2', 'y2^1', 'y2^2', 'y3^1']
    Last 5: ['b28^2', 'b29^1', 'b29^2', 'b30^1', 'b30^2']
    Total Number: 120
:::
:::

#### Modifying the encoding schema

::: {.cell execution_count="11"}
``` {.python .cell-code}
random_config = Config(ion_series = 'zc', ion_charges = (1,))
spectrum.config = random_config
print(spectrum.encode_fragments())
print("First 5: ", spectrum.fragment_labels[:5])
print("Last 5:", spectrum.fragment_labels[-5:])
print("Total Number:", len(spectrum.fragment_labels))
```

::: {.cell-output .cell-output-stdout}
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    First 5:  ['z1^1', 'z2^1', 'z3^1', 'z4^1', 'z5^1']
    Last 5: ['c26^1', 'c27^1', 'c28^1', 'c29^1', 'c30^1']
    Total Number: 60
:::
:::
