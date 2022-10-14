from pprint import pprint

from tqdm.auto import tqdm

from ms2ml.config import Config
from ms2ml.data.adapters import MSPAdapter

my_file = "notebooks/FTMS_HCD_20_annotated_2019-11-12.msp"


def post_hook(spec):
    return {
        "aa": spec.precursor_peptide.aa_to_onehot(),
        "mods": spec.precursor_peptide.mod_to_vector(),
    }


my_adapter = MSPAdapter(file=my_file, config=Config(), out_hook=post_hook)

# 242399it [02:56, 1696.62it/s]
out = []
for i, x in enumerate(tqdm(my_adapter.parse())):
    if i > 10_000:
        break
    out.append(x)

bundled = my_adapter.bundle(out)
pprint({k: v.shape for k, v in bundled.items()})
print({k: f"{type(v)}: of shape: {v.shape}" for k, v in bundled.items()})
