import sys
sys.path.append('/home/jkha/01_PROJECTS/07_direct_dynamics/code/gemini')
import numpy as np
from pyscf import gto, lib, scf, mcscf, csf_fci, fci
from pyscf.tools import molden
from pre_bo import pre_BO
import scipy
import math, os

#os.environ["OMP_NUM_THREADS"] = "10"

# Control
dt = 1.0
nstep = 100
freq = 1
spin = 0

# System
# ethylene
mol = gto.M(
    #atom="../../../01_casscf/01_ciopt/opt.xyz",
    atom="./opt.xyz",
    #basis='6-31G**',
    basis='sto-3g',
    unit='Angstrom',
    verbose=0
)
ne_cas = (1, 1)
no_cas = 2

# water
#mol = gto.M(
#    atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
#    basis='sto-3g',
#    #basis='6-31g',
#    #unit='Angstrom',
#    verbose=0
#)
##ne_cas = (5, 5)
##no_cas = 7
#ne_cas = (2, 2)
#no_cas = 3

# LiH
#mol = gto.M(
#    atom='Li 1.0 0.0 0.0; H 0 0 0',
#    basis='sto-3g',
#    unit='Bohr',
#    verbose=0
#)
#ne_cas = (2, 2)
#no_cas = 6

# ammonia
#mol = gto.M(
#    atom="./ammonia.xyz",
#    basis='sto-3g',
#    unit='Angstrom',
#    verbose=0
#)
##ne_cas = (5, 5)
##no_cas = 8
#ne_cas = (2, 2)
#no_cas = 3

print(mol.nao)

# Initialization
coords0 = mol.atom_coords(unit='Bohr')
pb = pre_BO(mol, ne_cas, no_cas)
pb.csf_coeff[0] = 1.0

print("ncore=", pb.ncore)
print("ncas=", pb.no_cas)
print("nvirt=", pb.nvirt)

# HF
pb.mf = scf.RHF(pb.mol)
pb.mf.run(conv_tol=1e-8)
pb.mo_coeff = np.copy(pb.mf.mo_coeff)

# CASSCF
cas_scf = mcscf.CASSCF(pb.mf, pb.no_cas, pb.ne_cas)
weights = np.ones(pb.ncsf)/pb.ncsf
#cas_scf.fcisolver = csf_fci.csf_solver(pb.mol, smult=1) # fci solver with CSF
cas_scf.fcisolver = fci.solver(pb.mol, singlet=True)
cas_scf.state_average_(weights)
cas_scf.fcisolver.nroots = pb.ncsf
cas_scf.kernel()
pb.mo_coeff = np.copy(cas_scf.mo_coeff)

molden.from_mo(pb.mol, "initial.molden", pb.mo_coeff)





# Lowdin
#s_tmp = pb.mol.intor('int1e_ovlp')
#s_vals, s_vecs = scipy.linalg.eigh(s_tmp)
#pb.mo_coeff = s_vecs @ np.diag(1.0/np.sqrt(s_vals)) @ s_vecs.T

pb.mc.mo_coeff = np.copy(pb.mo_coeff)
pb.mc.mol = np.copy(pb.mol)
pb.mo_coeff_old = np.copy(pb.mo_coeff)
pb.mol_old = pb.mol.copy()

pb.get_int_ao()
pb.get_V_csf()
pb.get_grad_ao()
pb.get_grad_coeff()
#pb.get_grad_coeff_spacewise()
pb.get_int_mo()
#pb.get_D_csf()
pb.get_dV_csf()
f = open('movie.xyz', 'w')
f.write(f"{pb.nat}\n\n")
for iat in range(pb.nat):
    string = f"{pb.mol.elements[iat]}"
    for isp in range(3):
        string += f" {pb.pos[iat, isp]*0.529177}"
    string += "\n"
    f.write(string)

kinetic = 0.5 * np.einsum('A, Ac ->', pb.mass, pb.vel ** 2)
#potential = pb.V_nuc
#potential = pb.V_csf[1, 1]
potential = pb.V_core
#potential = pb.V_csf[1, 1] + pb.V_core
#potential += np.einsum('I, IJ, J ->', pb.csf_coeff.conj(), pb.V_csf, pb.csf_coeff).real

f = open('energy.dat', 'w')
f.write(f"0.0 {kinetic} {potential} {kinetic+potential}\n")
print(f"0.0 {kinetic} {potential} {kinetic+potential}")

#rforce = -pb.dV_nuc
rforce = -pb.dV_core
#rforce = -pb.dV_csf[:, :, 1, 1]
#rforce = -pb.dV_csf[:, :, 1, 1] - pb.dV_core
f_com = np.sum(rforce, axis=0)
print(f_com)
#rforce -= f_com


for istep in range(nstep):
    # Backup
    pb.mol_old = pb.mol.copy()
    pb.mo_coeff_old = np.copy(pb.mo_coeff)
    pb.grad_coeff_old = np.copy(pb.grad_coeff)
    
    # Half step, update position and mol
    pb.vel += 0.5 * dt * rforce / np.column_stack([pb.mass] * 3)
    pb.pos += dt * pb.vel
    pb.mol.set_geom_(pb.pos, unit='Bohr')
    
    # Update mo coeff
    pb.get_int_ao()
    # Update existing mf and mc instead of re-creating
    #pb.mf.mol = pb.mol
    #pb.mf.init_guess = 'mo'
    #pb.mf.mo_coeff = np.copy(pb.mo_coeff_old)
    #pb.mf.run(conv_tol=1e-8)
    #pb.mo_coeff = np.copy(pb.mf.mo_coeff)
    # Obtain lowdin orbital
    s_tmp = pb.mol.intor('int1e_ovlp')
    #s_sqrt = scipy.linalg.sqrtm(s_tmp)
    #pb.mo_coeff = scipy.linalg.inv(s_sqrt)
    s_vals, s_vecs = scipy.linalg.eigh(s_tmp)
    pb.mo_coeff = s_vecs @ np.diag(1.0/np.sqrt(s_vals)) @ s_vecs.T
    pb.align_mo_coeff(pb.mol_old, pb.mo_coeff_old)
    #pb.align_mo_coeff_spacewise(pb.mol_old, pb.mo_coeff_old)
    
    # Calculate properties at updated geometry.
    pb.get_grad_ao()
    pb.mc.mol = pb.mol
    pb.mc.mo_coeff = np.copy(pb.mo_coeff)
    pb.get_V_csf()
    pb.get_grad_coeff()
    #pb.get_grad_coeff_spacewise()
    pb.get_int_mo()
    #pb.get_D_csf()
    pb.get_dV_csf()
    #rforce = -pb.dV_nuc
    #rforce = -pb.dV_csf[:, :, 1, 1]
    rforce = -pb.dV_core
    #rforce = -pb.dV_csf[:, :, 1, 1] - pb.dV_core
    f_com = np.sum(rforce, axis=0)
    print(f_com)
    #rforce -= f_com
    #pb.vel += 0.5 * dt * pb.force / np.column_stack([pb.mass] * 3)
    pb.vel += 0.5 * dt * rforce / np.column_stack([pb.mass] * 3)
    kinetic = 0.5 * np.einsum('A, Ac ->', pb.mass, pb.vel ** 2)
    #potential = pb.V_nuc
    #potential = pb.V_csf[1, 1]
    potential = pb.V_core
    #potential = pb.V_csf[1, 1] + pb.V_core
    #potential += np.einsum('I, IJ, J ->', pb.csf_coeff.conj(), pb.V_csf, pb.csf_coeff).real
    if ((istep + 1) % freq == 0): 
        f = open('energy.dat', 'a')
        f.write(f"{(istep+1)*dt} {kinetic} {potential} {kinetic+potential}\n")
        print(f"{(istep+1)*dt} {kinetic} {potential} {kinetic+potential}")

        f = open('movie.xyz', 'a')
        f.write(f"{pb.nat}\n\n")
        for iat in range(pb.nat):
            string = f"{pb.mol.elements[iat]}"
            for isp in range(3):
                string += f" {pb.pos[iat, isp]*0.529177}"
            string += "\n"
            f.write(string)
            #print(string)

molden.from_mo(pb.mol, "final.molden", pb.mo_coeff)
