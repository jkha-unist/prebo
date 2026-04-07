import sys
import os
sys.path.append('/home/jkha/01_PROJECTS/07_direct_dynamics/code/gemini')
import numpy as np
from pyscf import gto, scf, mcscf, fci, csf_fci
from pre_bo import pre_BO
import scipy
import math

# Simulation Control
dt = 0.1          # Nuclear time step (au)
nstep = 2000       # Number of steps
n_elec_steps = 10  # Electronic sub-steps per nuclear step
freq = 10          # Saving frequency

# LiH
mol = gto.M(
    atom='Li 1.5 0.0 0.0; H 0 0 0',
    basis='6-31g',
    unit='Bohr',
    symmetry=True,
    spin=0,
    verbose=0
)
ne_cas = (2, 2)
no_cas = mol.nao

# Initialize pre_BO object
pb = pre_BO(mol, ne_cas, no_cas)

# Separate CASCI object for FCI logging to avoid affecting dynamics
mc_fci = mcscf.CASCI(pb.mol, no_cas, ne_cas)
mc_fci.fcisolver = csf_fci.csf_solver(pb.mol, smult=1)
ncsf_cut = 5
mc_fci.fcisolver.nroots = ncsf_cut

# --- Initial State Initialization ---
pb.mf = scf.RHF(pb.mol)
pb.mf.run(conv_tol=1e-8)
pb.mo_coeff = np.copy(pb.mf.mo_coeff)

# SA-CASSCF for initial orbitals
#nroots = 3
#cas_scf = mcscf.CASSCF(pb.mf, pb.no_cas, pb.ne_cas)
#cas_scf.fcisolver = fci.solver(pb.mol, singlet=True)
#cas_scf.fcisolver.nroots = nroots
#weights = np.ones(nroots) / nroots
#cas_scf.state_average_(weights)
#cas_scf.kernel()
#pb.mo_coeff = np.copy(cas_scf.mo_coeff)

# FCI calculation for initial state
mc_fci.mo_coeff = np.copy(pb.mo_coeff)
mc_fci.kernel()
fci_ci_vec = np.array(pb.transformer.vec_det2csf(mc_fci.ci))

# 3. Setup Step 0 Frame
pb.mol_old = mol.copy()
pb.mo_coeff_old = np.copy(pb.mo_coeff)

# Set Initial Electronic state (e.g., pure Ground State CSF)
istate = 1
pb.csf_coeff[:] = fci_ci_vec[istate, :] + 0.0j

# Get properties at t0
pb.mc.mo_coeff = np.copy(pb.mo_coeff)
pb.get_int_ao()
pb.get_V_csf()
pb.get_grad_ao()
pb.get_grad_coeff_continuous(block_orth=False)
pb.get_int_mo()
pb.get_D_csf()
pb.get_dV_csf()
pb.calculate_force()
rforce = pb.force.copy()

# --- Dynamics Loop ---
print("# Time Kinetic Potential Total Populations...")
energy_file = open("energy.dat", "w")
fci_file = open("fci.dat", "w")
pop_file = open("pop.dat", "w")
pop_bo_file = open("pop_bo.dat", "w")
movie_file = open("movie.xyz", "w")

# Write headers
output_e = f"#Time Kinetic_e Potential_e Total_e Norm\n"
output_f = f"#Time " + " ".join([f"E_{i}" for i in range(ncsf_cut)]) + "\n"
energy_file.write(output_e)
fci_file.write(output_f)
pop_file.write("# Time " + " ".join([f"Pop_{i}" for i in range(pb.ncsf)]) + "\n")
pop_bo_file.write("# Time " + " ".join([f"Pop_{i}" for i in range(ncsf_cut)]) + "\n")

for istep in range(nstep + 1):
    # 1. Output current state
    kinetic = 0.5 * np.sum(pb.mass[:, None] * pb.vel**2)
    potential = np.einsum('I, IJ, J ->', pb.csf_coeff.conj(), pb.V_csf, pb.csf_coeff).real + pb.V_core
    total_e = kinetic + potential
    
    if istep % freq == 0:
        if istep > 0:
            # Perform FCI logging using the SEPARATE mc_fci object
            mc_fci.mol = pb.mol
            mc_fci.mo_coeff = pb.mo_coeff
            mc_fci.kernel()
            fci_ci_vec = np.array(pb.transformer.vec_det2csf(mc_fci.ci))
        
        pops_bo = np.absolute(np.einsum('iI, I->i', fci_ci_vec[:, :], pb.csf_coeff[:]))**2
        
        output_e = f"{istep*dt} {kinetic:.8f} {potential:.8f} {total_e:.8f} {np.sum(np.abs(pb.csf_coeff)**2)}"
        output_f = f"{istep*dt} " + " ".join([f"{e:.8f}" for e in mc_fci.e_tot])
        pop_str = " ".join([f"{p:.6f}" for p in np.abs(pb.csf_coeff)**2])
        pop_bo_str = " ".join([f"{p:.6f}" for p in pops_bo])
        
        energy_file.write(output_e + "\n")
        fci_file.write(output_f + "\n")
        pop_file.write(f"{istep*dt} {pop_str}\n")
        pop_bo_file.write(f"{istep*dt} {pop_bo_str}\n")
        
        print(output_e)
        
        movie_file.write(f"{pb.nat}\n\n")
        for iat in range(pb.nat):
            movie_file.write(f"{pb.mol.elements[iat]} {pb.pos[iat,0]*0.529177:.6f} {pb.pos[iat,1]*0.529177:.6f} {pb.pos[iat,2]*0.529177:.6f}\n")
        energy_file.flush()
        fci_file.flush()
        pop_file.flush()
        pop_bo_file.flush()
        movie_file.flush()

    if istep == nstep: break

    pb.backup_elec_state()
    pb.propagate_mo_coeff(dt=dt, block_orth=False)
    
    # Velocity Verlet Step 1
    pb.vel += 0.5 * dt * rforce / pb.mass[:, None]
    pb.pos += dt * pb.vel
    if pb.mol.symmetry:
        pb.pos[:, 1:] = 0.0
        pb.vel[:, 1:] = 0.0
    pb.mol.set_geom_(pb.pos, unit='Bohr')
    pb.mol.build()
    
    # Properties at t + dt
    pb.get_int_ao()
    pb.get_V_csf()
    pb.get_grad_ao()
    pb.get_grad_coeff_continuous(block_orth=False)
    pb.get_int_mo()
    pb.get_D_csf()
    pb.get_dV_csf()
    
    # Electronic propagation
    pb.propagate_elec_expm(dt)
    
    # Calculate force for Step 2
    pb.calculate_force()
    rforce = pb.force.copy()
    
    # Velocity Verlet Step 2
    pb.vel += 0.5 * dt * rforce / pb.mass[:, None]

    pb.mol_old = pb.mol.copy()
    pb.mo_coeff_old = np.copy(pb.mo_coeff)

energy_file.close()
pop_file.close()
movie_file.close()
