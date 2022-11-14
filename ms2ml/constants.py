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

STD_AA_MASS = {
    "G": 57.02146372057,
    "A": 71.03711378471,
    "S": 87.03202840427001,
    "P": 97.05276384885,
    "V": 99.06841391299,
    "T": 101.04767846841,
    "C": 103.00918478471,
    "L": 113.08406397713001,
    "I": 113.08406397713001,
    "J": 113.08406397713001,
    "N": 114.04292744114001,
    "D": 115.02694302383001,
    "Q": 128.05857750527997,
    "K": 128.09496301399997,
    "E": 129.04259308796998,
    "M": 131.04048491299,
    "H": 137.05891185845002,
    "F": 147.06841391298997,
    "U": 150.95363508471,
    "R": 156.10111102359997,
    "Y": 163.06332853254997,
    "W": 186.07931294985997,
    "O": 237.14772686284996,
}
"""A dictionary with monoisotopic masses of the twenty standard
amino acid residues, selenocysteine and pyrrolysine.
Acquired from pyteomics

#From https://proteomicsresource.washington.edu/protocols06/masses.php
"""
