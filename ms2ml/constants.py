"""
Basic constants for the masses of some common molecules.

Such as water, ammonia, hydrogen, oxygen, carbon, ion series offsets.
Originally calculated using pyteomics
"""

# from pyteomics import mass # type: ignore

# WATER = mass.calculate_mass(formula="H2O")
WATER = 18.0105646837
# OH = mass.calculate_mass(formula="OH")
OH = 17.002739651629998
# PROTON = mass.calculate_mass(formula="H")
PROTON = 1.00782503207

N_TERMINUS = PROTON
C_TERMINUS = OH
# CO = mass.calculate_mass(formula="CO")
CO = 27.99491461956
CHO = CO + PROTON
# NH2 = mass.calculate_mass(formula="NH2")
NH2 = 16.01872406894
NH3 = NH2 + PROTON

ION_OFFSET = {
    "a": 0 - CHO,
    "b": 0 - PROTON,
    "c": 0 + NH2,
    "x": 0 + CO - PROTON,
    "y": 0 + PROTON,
    "z": 0 - NH2,
}
