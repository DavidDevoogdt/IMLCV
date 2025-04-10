#
# This file is adapted from MolMod.
#


# *** constants ***
boltzmann = 3.1668154051341965e-06
avogadro = 6.0221415e23
lightspeed = 137.03599975303575
planck = 6.2831853071795864769


# *** Generic ***
au = 1.0


# *** Charge ***

coulomb = 1.0 / 1.602176462e-19

# Mol

mol = avogadro

# *** Mass ***

kilogram = 1.0 / 9.10938188e-31

gram = 1.0e-3 * kilogram
miligram = 1.0e-6 * kilogram
unified = 1.0e-3 * kilogram / mol
amu = unified

# *** Length ***

meter = 1.0 / 0.5291772083e-10

decimeter = 1.0e-1 * meter
centimeter = 1.0e-2 * meter
milimeter = 1.0e-3 * meter
micrometer = 1.0e-6 * meter
nanometer = 1.0e-9 * meter
angstrom = 1.0e-10 * meter
picometer = 1.0e-12 * meter

# *** Volume ***

liter = decimeter**3

# *** Energy ***

joule = 1 / 4.35974381e-18

calorie = 4.184 * joule
kjmol = 1.0e3 * joule / mol
kcalmol = 1.0e3 * calorie / mol
electronvolt = (1.0 / coulomb) * joule
rydberg = 0.5

# *** Force ***

newton = joule / meter

# *** Angles ***

deg = 0.017453292519943295
rad = 1.0

# *** Time ***

second = 1 / 2.418884326500e-17

nanosecond = 1e-9 * second
femtosecond = 1e-15 * second
picosecond = 1e-12 * second

# *** Frequency ***

hertz = 1 / second

# *** Pressure ***

pascal = newton / meter**2
bar = 100000 * pascal
atm = 1.01325 * bar

# *** Temperature ***

kelvin = 1.0

# *** Dipole ***

debye = 0.39343031369146675  # = 1e-21*coulomb*meter**2/second/lightspeed

# *** Current ***

ampere = coulomb / second
