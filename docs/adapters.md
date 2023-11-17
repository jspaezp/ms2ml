---
title: Adapters
toc-title: Table of contents
---

Adapters are objects that implement interfaces between raw data, usally
stored externally, to the internal representation objects.

Just as a reminder, these objects are the `Peptide`, the `Spectrum` and
the `AnnotatedPeptideSpectrum`.

For details on the available adapters, take a look at the adapter
section inside the `ms2ml API` documents.

## Basic Usage

::: {.cell execution_count="1"}
``` {.python .cell-code}
from tempfile import NamedTemporaryFile
from ms2ml.data.adapters import MSPAdapter
from ms2ml.config import Config
from pprint import pprint

# A very boring msp with 2 spectra
text = (
    "Name: ASDASD/2\n"
    "MW: 1234\n"
    '123 123 "asd"\n'
    "\n"
    "Name: ASDASDAAS/1\n"
    "MW: 12343\n"
    '123 123 "as5d"\n'
    '123 123 "asd"\n'
    "\n"
)

f = NamedTemporaryFile(delete=False)
with open(f.name, "w") as f:
    f.write(text)

my_file = f.name

my_adapter = MSPAdapter(config=Config(), file = my_file)

pprint(list(my_adapter.parse()))
```

::: {.cell-output .cell-output-stdout}
    [AnnotatedPeptideSpectrum(mz=array([123.]), intensity=array([123.]), ms_level=2, precursor_mz=0, precursor_charge=2, instrument=None, analyzer=None, collision_energy=nan, activation=None, extras=None, retention_time=RetentionTime(rt=nan, units='seconds', run=None), precursor_ion_mobility=None, precursor_peptide=Peptide([('A', None), ('S', None), ('D', None), ('A', None), ('S', None), ('D', None)], {'n_term': None, 'c_term': None, 'unlocalized_modifications': [], 'labile_modifications': [], 'fixed_modifications': [], 'intervals': [], 'isotopes': [], 'group_ids': [], 'charge_state': ChargeState(2, [])}), precursor_isotope=0),
     AnnotatedPeptideSpectrum(mz=array([123., 123.]), intensity=array([123., 123.]), ms_level=2, precursor_mz=0, precursor_charge=1, instrument=None, analyzer=None, collision_energy=nan, activation=None, extras=None, retention_time=RetentionTime(rt=nan, units='seconds', run=None), precursor_ion_mobility=None, precursor_peptide=Peptide([('A', None), ('S', None), ('D', None), ('A', None), ('S', None), ('D', None), ('A', None), ('A', None), ('S', None)], {'n_term': None, 'c_term': None, 'unlocalized_modifications': [], 'labile_modifications': [], 'fixed_modifications': [], 'intervals': [], 'isotopes': [], 'group_ids': [], 'charge_state': ChargeState(1, [])}), precursor_isotope=0)]
:::
:::

## Hook usage to modify the output

We will keep using the same file as before, but we will add a hook to
modify the output.

Since every intermediate entry is an annotated Spectrum object, we can
get the corresponding peptide from `spec.precurse_peptide` and encode it
on the fly and return an dictionary of tensors!

We could in theory do this in a loop .... something like this ...

    my_adapter = MSPAdapter(config=Config(), file = my_file)
    out1 = []
    out2 = []
    for i, x in enumerate(my_adapter.parse()):
        out1.append(x.precurse_peptide.aa_to_count())
        out2.append(x.precurse_peptide.mod_to_vector())

    out1 = np.array(out1)
    out2 = np.array(out2)

    print(out1.shape)
    print(out2.shape)

Or we can use the hook system to do it for us!

::: {.cell execution_count="2"}
``` {.python .cell-code}
def post_hook(spec):
    return {
        "aa": spec.precursor_peptide.aa_to_count(),
        "mods": spec.precursor_peptide.mod_to_vector(),
    }

my_adapter.out_hook = post_hook

# We can also add the hook when deifing the adapter
# my_adapter = MSPAdapter(
#     file=my_file, config=Config(), out_hook=post_hook
# )

for i, x in enumerate(my_adapter.parse()):
    print(i)
    pprint(x)
```

::: {.cell-output .cell-output-stdout}
    0
    {'aa': array([1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
           0, 0, 0, 0, 0, 1, 0]),
     'mods': array([0, 0, 0, 0, 0, 0, 0, 0])}
    1
    {'aa': array([1, 4, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
           0, 0, 0, 0, 0, 1, 0]),
     'mods': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
:::
:::

We can even combine those outputs into batches using the `bundle`
method.

::: {.cell execution_count="3"}
``` {.python .cell-code}
bundled = my_adapter.bundle(my_adapter.parse())
pprint({k:v.shape for k,v in bundled.items()})
```

::: {.cell-output .cell-output-stdout}
    {'aa': (2, 29), 'mods': (2, 11)}
:::

::: {.cell-output .cell-output-stderr}
    /Users/sebastianpaez/Library/Caches/pypoetry/virtualenvs/ms2ml-zJc7qiTs-py3.9/lib/python3.9/site-packages/ms2ml/utils/tensor_utils.py:110: UserWarning: Padding to shape (11,) because the shapes are not the same
      warnings.warn(
:::
:::

::: {.cell execution_count="4"}
``` {.python .cell-code}
pprint(bundled)
```

::: {.cell-output .cell-output-stdout}
    {'aa': array([[1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
            0, 0, 0, 0, 0, 1, 0],
           [1, 4, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
            0, 0, 0, 0, 0, 1, 0]]),
     'mods': array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
:::
:::

## Changing the configuration

We can also change the configuration of the adapter to change the
output.

This makes sense if the way your model is trained requires a different
variation of the encoding.

::: {.cell execution_count="5"}
``` {.python .cell-code}
my_conf = Config(encoding_aa_order = ('A','S','D'))
my_adapter = MSPAdapter(
    file=my_file, config=Config(), out_hook=post_hook
)
bundled = my_adapter.bundle(my_adapter.parse())
print("Default Shapes:")
pprint({k:v.shape for k,v in bundled.items()})

my_adapter.config = my_conf
bundled = my_adapter.bundle(my_adapter.parse())
print("Configured Shapes:")
pprint({k:v.shape for k,v in bundled.items()})
```

::: {.cell-output .cell-output-stdout}
    Default Shapes:
    {'aa': (2, 29), 'mods': (2, 11)}
    Configured Shapes:
    {'aa': (2, 3), 'mods': (2, 11)}
:::
:::

Note that the 'aa' element only has 3 dimensions, since we only have 3
defined aminoacids in the configuration!!

::: {.cell execution_count="6"}
``` {.python .cell-code}
# We close the connection to the dummy file because we are good
# programmers
f.close()
```
:::

## Batching

Sometimes we dont want either 1 element at a time or all of them at a
time. For those cases, we batch.

In this case we can get form a file of 1000 spectra, 10 at a time.

::: {.cell execution_count="7"}
``` {.python .cell-code}
# Same setup as before but for a much larger file (1000 spectra)
f = NamedTemporaryFile(delete=False)
with open(f.name, "w") as f:
    # We make a file with 1000 spectra
    f.write(text*500)

my_file = f.name

def post_hook(spec):
    return {
        "aa": spec.precursor_peptide.aa_to_count(),
        "mods": spec.precursor_peptide.mod_to_vector(),
     }
```
:::

When defining the adapter, we can specify the collate_fn. This function
is called to combine all the elements in a batch. it can be something as
simple as as 'list' or something as complicated as you want to make it.
(np.stack, torch.stack, torch.cat, combinations of the former ... the
sky is the limit)

::: {.cell execution_count="8"}
``` {.python .cell-code}
my_adapter = MSPAdapter(
    config=Config(),
     file = my_file,
     out_hook=post_hook,
     collate_fn = lambda x: x)
```
:::

Here we define the `batch_size` to be 10.

::: {.cell execution_count="9"}
``` {.python .cell-code}
for i, x in enumerate(my_adapter.batch(batch_size = 10)):
    print(i)
    break

print(x[1])
print(len(x))
print(type(x))
# The shapes of every element in the batch
```

::: {.cell-output .cell-output-stdout}
    0
    {'aa': array([1, 4, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
           0, 0, 0, 0, 0, 1, 0]), 'mods': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
    10
    <class 'list'>
:::
:::

And every iteration, x is a new group of 10 peptides, encoded the way we
specified before.

::: {.cell execution_count="10"}
``` {.python .cell-code}
my_adapter = MSPAdapter(
    config=Config(),
     file = my_file,
     out_hook=post_hook)

for i, x in enumerate(my_adapter.batch(batch_size = 4)):
    print(i)
    break


pprint(x)
print(len(x))
print(type(x))
f.close()
```

::: {.cell-output .cell-output-stdout}
    0
    {'aa': array([[1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
            0, 0, 0, 0, 0, 1, 0],
           [1, 4, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
            0, 0, 0, 0, 0, 1, 0],
           [1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
            0, 0, 0, 0, 0, 1, 0],
           [1, 4, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
            0, 0, 0, 0, 0, 1, 0]]),
     'mods': array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
    2
    <class 'dict'>
:::
:::
