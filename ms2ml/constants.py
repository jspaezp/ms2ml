from pyteomics import mass

WATER = mass.calculate_mass(formula="H2O")
OH = mass.calculate_mass(formula="OH")
PROTON = mass.calculate_mass(formula="H")

N_TERMINUS = PROTON
C_TERMINUS = OH
CO = mass.calculate_mass(formula="CO")
CHO = CO + PROTON
NH2 = mass.calculate_mass(formula="NH2")
NH3 = NH2 + PROTON

ION_OFFSET = {
    "a": 0 - CHO,
    "b": 0 - PROTON,
    "c": 0 + NH2,
    "x": 0 + CO - PROTON,
    "y": 0 + PROTON,
    "z": 0 - NH2,
}
