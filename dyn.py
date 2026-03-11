import sys
sys.path.append('/home/jkha/01_PROJECTS/07_direct_dynamics/code/gemini')
import numpy as np
from pyscf import gto, lib, scf
from pre_bo import pre_BO
import scipy
import math, os

#os.environ["OMP_NUM_THREADS"] = "10"

# Simulation parameters
ne_cas = (1, 1)
no_cas = 2
spin = 0
dt = 1.0
nstep = 200

# CI
mol = gto.M(
    #atom="../../../01_casscf/01_ciopt/opt.xyz",
    atom="./opt.xyz",
    basis='6-31G**',
    unit='Angstrom',
    verbose=0
)
coords0 = mol.atom_coords(unit='Bohr')

prebo = pre_BO(mol, ne_cas, no_cas)

prebo.csf_coeff[0] = 1.0

prebo.get_int_ao()
prebo.get_grad_ao()
#prebo.get_grad_mo()
prebo.mf = scf.RHF(prebo.mol)
prebo.mf.init_guess = 'mo'
prebo.mf.mo_coeff = np.copy(prebo.mo_coeff_old)
prebo.mf.run(conv_tol=1e-8)
prebo.mo_coeff = np.copy(prebo.mf.mo_coeff)
prebo.mc.mo_coeff = np.copy(prebo.mo_coeff)
prebo.get_V_csf()
prebo.get_int_mo()
prebo.get_D_csf()
prebo.get_dV_csf()
#prebo.get_dV_csf_2()
f = open('movie.xyz', 'w')
f.write(f"{prebo.nat}\n\n")
for iat in range(prebo.nat):
    string = f"{prebo.mol.elements[iat]}"
    for isp in range(3):
        string += f" {prebo.pos[iat, isp]*0.529177}"
    string += "\n"
    f.write(string)

kinetic = 0.5 * np.einsum('A, Ac ->', prebo.mass, prebo.vel ** 2)
potential = np.einsum('I, IJ, J->', prebo.csf_coeff.conj(), prebo.V_csf, prebo.csf_coeff).real
potential += prebo.V_core

f = open('energy.dat', 'w')
f.write(f"{kinetic} {potential} {kinetic+potential}\n")

for istep in range(nstep):
    prebo.calculate_force()
    prebo.vel += 0.5 * dt * prebo.force / np.column_stack([prebo.mass] * 3)
    prebo.pos += dt * prebo.vel
    prebo.mol_old = prebo.mol.copy()
    prebo.mo_coeff_old = np.copy(prebo.mo_coeff)
    prebo.mol.set_geom_(prebo.pos, unit='Bohr')
        
    prebo.get_int_ao()
    # Update existing mf and mc instead of re-creating
    prebo.mf.mol = prebo.mol
    prebo.mf.init_guess = 'mo'
    prebo.mf.mo_coeff = np.copy(prebo.mo_coeff_old)
    prebo.mf.run(conv_tol=1e-8)
    prebo.mo_coeff = np.copy(prebo.mf.mo_coeff)
    
    # Obtain lowdin orbital
    #s_tmp = prebo.mol.intor('int1e_ovlp')
    #s_sqrt = scipy.linalg.sqrtm(s_tmp)
    #prebo.mo_coeff = scipy.linalg.inv(s_sqrt)

    #prebo.align_mo_coeff_lowdin(prebo.mol_old, prebo.mo_coeff_old)
    prebo.align_mo_coeff(prebo.mol_old, prebo.mo_coeff_old)
    #prebo.align_mo_coeff_simple(prebo.mol_old, prebo.mo_coeff_old)

    prebo.get_grad_ao()
    prebo.mc.mol = prebo.mol
    prebo.mc.mo_coeff = np.copy(prebo.mo_coeff)
    prebo.get_V_csf()
    prebo.get_int_mo()
    prebo.get_D_csf()
    prebo.get_dV_csf()
    #prebo.get_dV_csf_2()
    prebo.calculate_force()
    prebo.vel += 0.5 * dt * prebo.force / np.column_stack([prebo.mass] * 3)
    kinetic = 0.5 * np.einsum('A, Ac ->', prebo.mass, prebo.vel ** 2)
    potential = np.einsum('I, IJ, J ->', prebo.csf_coeff.conj(), prebo.V_csf, prebo.csf_coeff).real
    potential += prebo.V_core
    f = open('energy.dat', 'a')
    f.write(f"{kinetic} {potential} {kinetic+potential}\n")
    print(f"{istep} {kinetic} {potential} {kinetic+potential}")

    f = open('movie.xyz', 'a')
    f.write(f"{prebo.nat}\n\n")
    for iat in range(prebo.nat):
        string = f"{prebo.mol.elements[iat]}"
        for isp in range(3):
            string += f" {prebo.pos[iat, isp]*0.529177}"
        string += "\n"
        f.write(string)
        #print(string)


