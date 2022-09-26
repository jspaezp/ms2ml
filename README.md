# ms2ml

**This package is in early development, I am actively taking ideas and requests**

The idea of this package is to have an intermeiate layer between the pyteomics package and ML applications.

Since ML applications do not take MS data as input directly, it is necessary to convert/encode it. This package is meant to handle that aspect.

This project is meant to be opinionated but not arbitrary. By that I mean that it should attempt to enforce the "better way" of doing things (not give flexibility to do everything every way) but all design decisions are open to discussion (ideally though github).

## Core parts

(subject to change...)

1. Parsers for external data are in ./ms2ml/data/parsing
    1. Parsers should only be able to read data and return a base python representation, dict/list etc.
1. Adapters are located in ./ms2ml/adapters, they should build on parsers (usually) but yield ms2ml representation objects (Spectrum, Peptide, AnnotatedSpectrum, LCMSSpectrum).
    1. Behavior can be modified/extended using hooks.
1. ms2ml representation objects have methods that converts them into tensor representations (Peptide.aa_to_onehot for instance)
1. As much configuration as possible should be stored in the config.Config class.
    1. It should contain information on the project and the way everything is getting encoded, so hypothetically one could just pass a config to a different data source adapter and get compatible results.
    1. Is our retention time in seconds or minutes?
        1. look at the config
    1. What position of the onehot is alanine?
        1. look at the config
    1. WHat order are our ions encoded in?
        1. Look at the config.

## Core design

1. Unified configuration
    - All configuration should be explicit or immediately visible upon request
1. Consistent API
    - It should feel similar to process the data inernally regardless of the input.
1. Flexible output
    - Every research need is different, therefore requesting different data from the API should be straightforward.
1. Extensibility.
    - It should be easy to adapt workflows to new and unexpected input data types.
    - This is achieved with the addition of hooks that allow an additional slim layer of compatibility
1. Abstract the loops away
    - I do not like writting boilerplate code, neither should you. Ideally you will not need to write loops when using the user-facing API
    - Therefore I try my best to abstract all the `[f(spec) for spec in file]` within reason.
1. Fail loudly
    - It is already hard to debug ML models, lets make it easier by having **sensical** error messages and checks. They should also contain suggestions to fix the bug. Explicit is better than implicit. Errors are better than bugs.
1. Api documentation.
    - Documentation is critical, if it is not documented, It will be deleted (because nobody will use it ...)
    - Within reason, all external api should be documented, typed, in-code commented, have a docstring, check that it renders well using mkdocs and an example.
    - All classes should have a static `_sample` static method that gives a sample of that object, and its docstring shoudl include an example on how to generate it.

## Target audience

People who want to train ML models from peptide/proteomics data instead of figuring out ways to encode their tensors and write parsers.

## Controlled Vocabulary

- `n_term` = denotes the n-terminus of a peptide
- `c_term` = denotes the c-terminus of a peptide
- `__missing__` = denotes missing/empty elements in a tensor

## Peptide sequence notation

When possible I will attempt to allow 'Proforma' based sequence annotations.

Check:

- https://pyteomics.readthedocs.io/en/latest/api/proforma.html
- http://psidev.info/sites/default/files/2020-12/ProForma_v2_draft12_0.pdf
- https://www.psidev.info/proforma

# TODO

- [x] Spectrum converter (extended object that allow to go from spectrum to ms encodings)
    - [x] Spectrum Tensor batch
    - [x] Annotation option (adding a peptide object).
        - [ ] sum/max combination
        - [ ] decimal precision
- [x] Peptide converter
    - [x] One hot
    - [x] Numeric encoding
    - [ ] Peptide Tensor Batch
- [x] Readers from mass spec data
    - [ ] Decide which other to implement/have available
- [ ] Dataset Objects (torch dataset objects)
    - [ ] In disk caching
    - [ ] In mem caching
    - [ ] Peptide Dataset
    - [ ] Spectrum Dataset
    - [ ] Annotated Spectrum Dataset
        - [ ] HDF5/sqlite caching
- [ ] *Documentation, Documentation, Documentation*
    - [ ] Spectrum Class and subclasses
    - [ ] Peptide Class and subclasses
    - [ ] Helper Annotation classes


- [ ] Style
  - [ ] remove D100 from the exclusions in linting (missing docstring in module)
  - [ ] remove D104 (missing docstring in package)
  - [ ] Fix all flake8/pylint complains ...

# Contribution

Right not this is a proof of concept package, I would be happy to make it something more stable if there is interest.
Feel free to open an issue and we can discuss what you need out of it!! (and decide who can implement it)
