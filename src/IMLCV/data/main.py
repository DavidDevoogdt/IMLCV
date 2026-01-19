from openff.toolkit.topology import Molecule
from openmm import XmlSerializer
from openmm.app import ForceField, PDBFile
from openmmforcefields.generators import GAFFTemplateGenerator


def generate_cyclohexane_files():
    # Load your molecule (from SMILES or SDF, not PDB)
    mol = Molecule.from_smiles("C1CCCCC1")  # cyclohexane
    mol.generate_conformers(n_conformers=1)

    # Write to PDB if needed
    mol.to_file("cyclohexane.sdf", file_format="SDF")

    # Set up GAFF generator
    gaff = GAFFTemplateGenerator(molecules=mol)
    forcefield = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    forcefield.registerTemplateGenerator(gaff.generator)

    # Load topology from OpenFF molecule, not from PDB!
    top = mol.to_topology().to_openmm()
    system = forcefield.createSystem(top)

    # 5. Export the Serialized System XML
    # This file contains ALL physics and can be loaded in Python 3.12
    with open("cyclohexane_system.xml", "w") as f:
        f.write(XmlSerializer.serialize(system))

    # 6. Export a matching PDB file
    # The atom order here will match the System XML exactly
    mol.to_file("cyclohexane_geometry.pdb", file_format="PDB")

    print("-" * 30)
    print("Success! Generated two files:")
    print("1. cyclohexane_system.xml  <- The Physics (Load this in 3.12)")
    print("2. cyclohexane_geometry.pdb <- The Coordinates")
    print("-" * 30)


if __name__ == "__main__":
    generate_cyclohexane_files()
