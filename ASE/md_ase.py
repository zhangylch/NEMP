# -*- coding: utf-8 -*-
import ase.io.vasp
from ase import Atoms, units
import getneigh as getneigh
from ase.calculators.NEMP import NEMP
from ase.io import extxyz
from ase.io.trajectory import Trajectory
import time

from ase.optimize import BFGS,FIRE
from ase.constraints import ExpCellFilter
from ase.md.verlet import VelocityVerlet  as VV
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as MBD
import numpy as np

fileobj=open("h2o.extxyz",'r')
configuration = extxyz.read_extxyz(fileobj,index=slice(0,1))
atoms = next(configuration)
#atoms = atoms.repeat((2, 2, 2))
calc=NEMP(skin=1.0, skin_neigh=7000, pbc=[1, 1, 1], getneigh=getneigh, fconfig='full_config.json', initatoms=atoms, properties=['energy', 'forces'])
#warm up
calc.reset()
atoms.calc=calc
MBD(atoms,temperature_K=300)
dyn = VV(atoms, timestep=0.25 * units.fs)
dyn.run(steps=1)

start=time.time()
calc.reset()
atoms.calc=calc
MBD(atoms, temperature_K=300)
dyn = VV(atoms, timestep=0.25 * units.fs, logfile='md.log', loginterval=100)
dyn.run(steps=5000)
#traj.close()
print("hello3")

end=time.time()
print(start-end)
