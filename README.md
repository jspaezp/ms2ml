
# Pyteomics4ml_jspp

The idea of this package is to have an intermeiate layer between the pyteomics package and ML applications.

Since ML applications do not take MS data as input directly, it is necessary to convert/encode it. This package is meant to handle that aspect.

This project is meant to be opinionated but not arbitrary. By that I mean that it should attempt to enforce the "better way" of doing things (not give flexibility to do everything every way) but all design decisions are open to discussion (ideally though github).

## Target audience

People who want to train ML models from peptide/proteomics data instead of figuring out ways to encode their tensors and write parsers.

## Controlled Vocabulary

- `n_term` = denotes the n-terminus of a peptide
- `c_term` = denotes the c-terminus of a peptide
- `__missing__` = denotes missing/empty elements in a tensor

## Why the _jspp

I have that suffix (my initials) to prevent taking over the pyteomics4ml name (for now).
Since I am not part of the consortium, nor related to the pyteomics team, I do not want to take over a name they might use.
If this package becomes used and stable I might take it over.

## Peptide sequence notation

When possible I will attempt to allow 'Proforma' based sequence annotations.

Check:
- https://pyteomics.readthedocs.io/en/latest/api/proforma.html
- http://psidev.info/sites/default/files/2020-12/ProForma_v2_draft12_0.pdf
- https://www.psidev.info/proforma

# TODO

[x] Config Object, Object that stores the configuration of the encodings
[ ] Spectrum converter (extended object that allow to go from spectrum to ms encodings)
  - [ ] Spectrum Tensor batch
  - [ ] Annotation option (adding a peptide object).
    - [ ] sum/max combination
    - [ ] decimal precision
[x] Peptide converter (extended object that allows going from peptide to encodings)
  - [x] One hot
  - [x] Numeric encoding
  - [ ] Peptide Tensor Batch
[ ] Readers from mass spec data
  - 
[ ] Dataset Objects (torch dataset objects)
  - [ ] In disk caching
  - [ ] In mem caching
  - [ ] Peptide Dataset
  - [ ] Spectrum Dataset
  - [ ] Annotated Spectrum Dataset
[ ] HDF5/sqlite caching
[ ] Documentation, Documentation, Documentation

# Contribution

Right not this is a proof of concept package, I would be happy to make it something more stable if there is interest.
Feel free to open an issue and we can discuss what you need out of it!! (and decide who can implement it)