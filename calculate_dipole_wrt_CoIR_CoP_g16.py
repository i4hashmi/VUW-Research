#!/usr/bin/env python3
"""
Script to calculate the center of mass (CoM), center of charge (CoP for Mulliken, ESP, and NBO charges) of a molecule from an XYZ/log file,
then calculate the dipole moment from different charges, and write a new XYZ file with the CoM added as a pseudo-atom 'X'.
Script by Dr. Muhammad Ali Hashmi (26/06/2025)
"""

import numpy as np
import os
from contextlib import redirect_stdout # Redirect print output to both console and csv file

# Atomic weights dictionary
atomic_masses = {
"H" : 1.00794, "He": 4.002602, "Li": 6.941, "Be": 9.012182, "B": 10.811,
"C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.9984032, "Ne": 20.1797,
"Na": 22.98976928, "Mg": 24.3050, "Al": 26.9815386, "Si": 28.0855,
"P": 30.973762, "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.0983,
"Ca": 40.078, "Sc": 44.955912, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961,
"Mn": 54.938045, "Fe": 55.845, "Co": 58.933195, "Ni": 58.6934, "Cu": 63.546,
"Zn": 65.38, "Ga": 69.723, "Ge": 72.64, "As": 74.92160, "Se": 78.96,
"Br": 79.904, "Kr": 83.798, "Rb": 85.4678, "Sr": 87.62, "Y": 88.90585,
"Zr": 91.224, "Nb": 92.90638, "Mo": 95.96, "Tc": 98.0, "Ru": 101.07,
"Rh": 102.90550, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.411, "In": 114.818,
"Sn": 118.710, "Sb": 121.760, "Te": 127.60, "I": 126.90447, "Xe": 131.293,
"Cs": 132.9054519, "Ba": 137.327, "La": 138.90547, "Ce": 140.116,
"Pr": 140.90765, "Nd": 144.242, "Pm": 145.0, "Sm": 150.36, "Eu": 151.964,
"Gd": 157.25, "Tb": 158.92535, "Dy": 162.500, "Ho": 164.93032, "Er": 167.259,
"Tm": 168.93421, "Yb": 173.054, "Lu": 174.9668, "Hf": 178.49, "Ta": 180.94788,
"W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
"Au": 196.966569, "Hg": 200.59, "Tl": 204.3833, "Pb": 207.2, "Bi": 208.98040,
"Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0,
"Th": 232.03806, "Pa": 231.03588, "U": 238.02891, "Np": 237.0, "Pu": 244.0,
"Am": 243.0, "Cm": 247.0, "Bk": 247.0, "Cf": 251.0, "Es": 252.0, "Fm": 257.0,
"Md": 258.0, "No": 259.0, "Lr": 262.0, "Rf": 267.0, "Db": 268.0, "Sg": 271.0,
"Bh": 272.0, "Hs": 270.0, "Mt": 276.0, "Ds": 281.0, "Rg": 280.0, "Cn": 285.0,
"Nh": 286.0, "Fl": 289.0, "Mc": 290.0, "Lv": 293.0, "Ts": 294.0, "Og": 294.0}

# Atomic number to symbol mapping
atomic_symbols = {i: symbol for i, symbol in enumerate(atomic_masses.keys(), start=1)}

def is_float(s):
    try:
        float(s)
        return True
    except:
        return False
  
### A function to extract data (geometry, ESP charges, NBO charges) from the gaussian log file
def extract_data_from_log(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    atoms = []
    esp_charges = []
    nbo_charges = []
    mulliken_charges = []
    gaussian_dipole = []
    mol_charge = None

    # Extract total charge
    for line in lines:
        if 'Charge =' in line and 'Multiplicity =' in line:
            mol_charge = int(float(line.split()[2]))

    # Extract last "Standard orientation" block
    std_blocks = []
    for i, line in enumerate(lines):
        if 'Standard orientation:' in line:
            block = []
            for j in range(i+5, len(lines)):
                if '-----' in lines[j]:
                    break
                block.append(lines[j].split())
            std_blocks.append(block)
    last_block = std_blocks[-1]
    atoms = [(atomic_symbols[int(atom[1])], float(atom[3]), float(atom[4]), float(atom[5])) for atom in last_block]

    # Extract ESP charges
    esp_section = False
    for line in lines:
        if 'ESP charges:' in line:
            esp_section = True
            continue
        if esp_section:
            if line.strip() == "" or 'Sum of ESP charges' in line:
                break
            parts = line.split()
            if len(parts) >= 3 and is_float(parts[2]):
                esp_charges.append(float(parts[2]))

    # Extract NBO charges
    nbo_section = False
    for line in lines:
        if 'Summary of Natural Population Analysis:' in line:
            nbo_section = True
        elif nbo_section and '----' in line:
            continue
        elif nbo_section:
            if '===' in line:
                break
            parts = line.split()
            if len(parts) >= 3 and is_float(parts[2]):
                nbo_charges.append(float(parts[2]))

    # Extract Gaussian dipole
    for i, line in enumerate(lines):
        if 'Dipole moment (field-independent basis, Debye):' in line:
            for j in range(i+1, i+5):
                if 'X=' in lines[j]:
                    parts = lines[j].split()
                    Dx = float(parts[1])
                    Dy = float(parts[3])
                    Dz = float(parts[5])
                    gaussian_dipole = [Dx, Dy, Dz]
                    break

    return atoms, mol_charge, gaussian_dipole, esp_charges, nbo_charges

### Function to calculate the Center of Mass
def compute_center_of_mass(atoms):
    total_mass = 0.0
    weighted_sum = np.zeros(3)
    for symbol, x, y, z in atoms:
        mass = atomic_masses.get(symbol, 0.0)
        coord = np.array([x, y, z])
        weighted_sum += mass * coord
        total_mass += mass
    return weighted_sum / total_mass

### Function to calculate the Center of Positive Charge (taking atoms and charges as input)
def compute_center_of_positive_charge(atoms, charges):
    weighted_sum = np.zeros(3)
    total_charge = 0.0
    for i, (symbol, x, y, z) in enumerate(atoms):
        q = charges[i]
        if q > 0:
            weighted_sum += q * np.array([x, y, z])
            total_charge += q
    if total_charge == 0:
        print("⚠️ Warning: No net positive charge found.")
        return np.array([np.nan, np.nan, np.nan])
    return weighted_sum / total_charge

### Function to calculate the Dipole of atoms with their charges (ESP, NBO etc)
def compute_dipole(atoms, charges, origin):
    if np.isnan(origin).any():
        return np.array([np.nan, np.nan, np.nan])
    dipole = np.zeros(3)
    for i, (symbol, x, y, z) in enumerate(atoms):
        r = np.array([x, y, z]) - origin
        dipole += charges[i] * r
    return dipole * 4.80320427  # e·Å to Debye, http://openmopac.net/manual/dipole_moment.html

### Function to write the Center of Mass with molecule as a dummy atom X as an xyz file
def write_xyz_with_com(atoms, com, outname='molecule_with_com.xyz'):
    with open(outname, 'w') as f:
        f.write(f"{len(atoms)+1}\nXYZ with CoM as X\n")
        for atom in atoms:
            f.write(f"{atom[0]} {atom[1]:.6f} {atom[2]:.6f} {atom[3]:.6f}\n")
        f.write(f"X {com[0]:.6f} {com[1]:.6f} {com[2]:.6f}\n")
    print(f"✅ XYZ file written: {outname}")

### Function to write the dipole vector as dummy atoms X and Y. X will be at origin and Y will point towards arrow head
def write_dipole_vector(origin, dipole_vector, scale=1.0, outname='dipole_vec.xyz'):
    """
    Write a two-atom XYZ file:
    - 'X' at origin
    - 'Y' at origin + scaled dipole_vector
    'scale' adjusts arrow length in Å for visualization.
    """
    tip = origin + dipole_vector * scale
    total_atoms = len(atoms) + 2
    with open(outname, 'w') as f:
        f.write(f"{total_atoms}\nDipole vector with full molecule\n")
        for atom in atoms:
            f.write(f"{atom[0]} {atom[1]:.6f} {atom[2]:.6f} {atom[3]:.6f}\n")
        f.write(f"X {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n")
        f.write(f"Y {tip[0]:.6f} {tip[1]:.6f} {tip[2]:.6f}\n")

    print(f"✅ Dipole vector + molecule XYZ written: {outname}")

### Function to write the Center of Imidazole Ring with molecule as a dummy atom X as an xyz file
def write_xyz_with_imid_center(atoms, imid_center, outname='molecule_with_imid_center.xyz'):
    with open(outname, 'w') as f:
        f.write(f"{len(atoms)+1}\nXYZ with CoIR as X\n")
        for atom in atoms:
            f.write(f"{atom[0]} {atom[1]:.6f} {atom[2]:.6f} {atom[3]:.6f}\n")
        f.write(f"X {imid_center[0]:.6f} {imid_center[1]:.6f} {imid_center[2]:.6f}\n")
    print(f"✅ XYZ file written: {outname}")

### This function is to determine the imidazole ring in the molecule and compute its center
from rdkit import Chem
from rdkit.Chem import AllChem
def compute_center_of_imidazole(atoms):
    """
    Given atom list [(symbol, x, y, z)], construct RDKit Mol, identify 5-membered ring with two Ns,
    and return its centroid.
    """
    mol = Chem.RWMol()
    atom_indices = []
    for sym, _, _, _ in atoms:
        a = Chem.Atom(sym)
        idx = mol.AddAtom(a)
        atom_indices.append(idx)

    coords = Chem.Conformer(len(atoms))
    for i, (_, x, y, z) in enumerate(atoms):
        coords.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
    mol.AddConformer(coords)

    # Add bonds heuristically (based on distance threshold ~1.6 Å)
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            dist = np.linalg.norm(np.array(atoms[i][1:]) - np.array(atoms[j][1:]))
            if 0.9 < dist < 1.7:
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
    except:
        return None

    rings = Chem.GetSymmSSSR(mol)
    for ring in rings:
        if len(ring) == 5:
            symbols = [atoms[i][0] for i in ring]
            if symbols.count("N") == 2:
                coords = np.array([atoms[i][1:] for i in ring])
                return np.mean(coords, axis=0)
    return None

###########################################
### Main Part of the script starts here ###
###########################################
# Definitions: com (center of mass), cop (center of charge), mu (μ)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 calculate_dipole_CoM_CoP_g16log.py file.log")
        sys.exit(1)

    filename = sys.argv[1]
    # Construct filenames
    base = os.path.splitext(filename)[0]
    xyz_outname = base + "_CoM.xyz"
    xyz2_outname = base + "_CoIR.xyz"
    txt_outname = base + ".txt"
    atoms, mol_charge, g_dipole, esp, nbo = extract_data_from_log(filename)

    com = compute_center_of_mass(atoms)
    imid_center = compute_center_of_imidazole(atoms)
    # Initialize all optional outputs
    cop_esp = cop_nbo = cop_mulliken = None
    mu_esp_com = mu_esp_cop = None
    mu_nbo_com = mu_nbo_cop = None
    mu_esp_ring = mu_nbo_ring = None

    if esp:
        cop_esp = compute_center_of_positive_charge(atoms, esp)
        mu_esp_com = compute_dipole(atoms, esp, com)  ## Compute dipole from center of mass
        mu_esp_cop = compute_dipole(atoms, esp, cop_esp)  ## Compute dipole from center of positive charge (ESP)
        if mu_esp_com is not None: ## Write dipole vector as X and Y dummy atoms
                write_dipole_vector(com, mu_esp_com,scale=1.5,outname=base + "_ESP_CoM_dipole_vec.xyz")

    if nbo:
        cop_nbo = compute_center_of_positive_charge(atoms, nbo)
        mu_nbo_com = compute_dipole(atoms, nbo, com)
        mu_nbo_cop = compute_dipole(atoms, nbo, cop_nbo) ## Compute dipole from center of positive charge (NBO)

    if imid_center is not None:
        if esp:
            mu_esp_ring = compute_dipole(atoms, esp, imid_center) ## Compute dipole from center of ring
            if mu_esp_ring is not None:
                write_dipole_vector(imid_center, mu_esp_ring,scale=1.5,outname=base + "_CoIR_esp_dipole_vec.xyz")
        if nbo:
            mu_nbo_ring = compute_dipole(atoms, nbo, imid_center)
            if mu_nbo_ring is not None: ## Write dipole vector as X and Y dummy atoms
                write_dipole_vector(imid_center, mu_nbo_ring,scale=1.5,outname=base + "_CoIR_nbo_dipole_vec.xyz")
            
    with open(txt_outname, 'w') as f:
        with redirect_stdout(f):
            print(f"\n📋 Summary for {filename}")
            print(f"Total Charge: {mol_charge}")
            print(f"Gaussian Dipole (Debye): {g_dipole}  |μ| = {np.linalg.norm(g_dipole):.4f}")
            print(f"Center of Mass: {com}")

            if esp:
                print(f"Center of Positive Charge (ESP): {cop_esp}")
                print(f"Dipole (ESP, wrt CoM): {mu_esp_com}  |μ| = {np.linalg.norm(mu_esp_com):.4f}")
                print(f"Dipole (ESP, wrt CoP): {mu_esp_cop}  |μ| = {np.linalg.norm(mu_esp_cop):.4f}")
            else:
                print("⚠️ ESP charges not found.")

            if nbo:
                print(f"Center of Positive Charge (NBO): {cop_nbo}")
                print(f"Dipole (NBO, wrt CoM): {mu_nbo_com}  |μ| = {np.linalg.norm(mu_nbo_com):.4f}")
                print(f"Dipole (NBO, wrt CoP): {mu_nbo_cop}  |μ| = {np.linalg.norm(mu_nbo_cop):.4f}")
            else:
                print("⚠️ NBO charges not found.")

            if imid_center is not None:
                print(f"Center of Imidazole Ring: {imid_center}")
                if mu_esp_ring is not None:
                    print(f"Dipole (ESP, wrt Imidazole): {mu_esp_ring}  |\u03bc| = {np.linalg.norm(mu_esp_ring):.4f}")
                if mu_nbo_ring is not None:
                    print(f"Dipole (NBO, wrt Imidazole): {mu_nbo_ring}  |\u03bc| = {np.linalg.norm(mu_nbo_ring):.4f}")
            else:
                print("⚠️ Imidazole ring not found or could not determine center.")

    # Also print to console again
    print(f"\n📋 Analysis Summary for {filename}")
    print(f"Total Charge: {mol_charge}")
    print(f"Gaussian Dipole (Debye): {g_dipole}  |μ| = {np.linalg.norm(g_dipole):.4f}")
    print(f"Center of Mass: {com}")

    if esp:
        print(f"Center of Positive Charge (ESP): {cop_esp}")
        print(f"Dipole (ESP, wrt CoM): {mu_esp_com}  |μ| = {np.linalg.norm(mu_esp_com):.4f}")
        print(f"Dipole (ESP, wrt CoP): {mu_esp_cop}  |μ| = {np.linalg.norm(mu_esp_cop):.4f}")
    else:
        print("⚠️ ESP charges not found.")

    if nbo:
        print(f"Center of Positive Charge (NBO): {cop_nbo}")
        print(f"Dipole (NBO, wrt CoM): {mu_nbo_com}  |μ| = {np.linalg.norm(mu_nbo_com):.4f}")
        print(f"Dipole (NBO, wrt CoP): {mu_nbo_cop}  |μ| = {np.linalg.norm(mu_nbo_cop):.4f}")
    else:
        print("⚠️ NBO charges not found.")

    if imid_center is not None:
        print(f"Center of Imidazole Ring: {imid_center}")
        if mu_esp_ring is not None:
            print(f"Dipole (ESP, wrt Imidazole): {mu_esp_ring}  |\u03bc| = {np.linalg.norm(mu_esp_ring):.4f}")
        if mu_nbo_ring is not None:
            print(f"Dipole (NBO, wrt Imidazole): {mu_nbo_ring}  |\u03bc| = {np.linalg.norm(mu_nbo_ring):.4f}")
    else:
        print("⚠️ Imidazole ring not found or could not determine center.")

    # Write XYZ file
    write_xyz_with_com(atoms, com, xyz_outname)
    if imid_center is not None:
        write_xyz_with_imid_center(atoms, imid_center, xyz2_outname)
    else:
        print("⚠️ Skipping XYZ with Imidazole center: No ring found.")
