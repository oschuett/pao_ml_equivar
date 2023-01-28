#!/usr/bin/python3

import ase
import ase.io
import numpy as np
import os
import random

def save(atoms, fn):
    d = os.path.dirname(fn)
    if not os.path.exists(d):
        os.makedirs(d)
    ase.io.write(fn, atoms)
    print(f"Wrote {fn}")

def main():
    atoms_frame0 = ase.io.read("../2H2O_MD/frame_0000/coords.xyz")

    # Create samples rotated in 10 degree increments.
    for i, phi in enumerate(np.linspace(0, 360.0, 36)):
        atoms = atoms_frame0.copy()
        atoms.euler_rotate(phi=phi)
        save(atoms, f"./phi_{i:02d}/coords.xyz")

    # Create samples rotated by random angles.
    random.seed(42)
    for i in range(20):
        phi, theta, psi  = [360 * random.random() for _ in range(3)]
        atoms = atoms_frame0.copy()
        atoms.euler_rotate(phi=phi, theta=theta, psi=psi)
        save(atoms, f"./rand_{i:02d}/coords.xyz")

main()

#EOF