from openmm import *
from openmm.app import PDBFile, ForceField, PME, Simulation
from openmm.unit import *


def villin():

    pdb = PDBFile('villin.pdb')
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=PME,
                                     nonbondedCutoff=1 * nanometer,
                                     constraints=HBonds)

    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    T = 300 * kelvin
    timestep = 0.004 * picoseconds


if __name__ == "__main__":
    villin()