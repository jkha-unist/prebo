import sys
import os
# Ensure root directory is in path
root_dir = '/home/jkha/01_PROJECTS/07_direct_dynamics/code/gemini'
if root_dir not in sys.path:
    sys.path.append(root_dir)
import numpy as np
from pyscf import gto, scf, mcscf, fci, csf_fci, symm
from pre_bo_symm import pre_BO
import scipy
import math

# Simulation Control
dt = 0.1          # Nuclear time step (au)
nstep = 2000      # Number of steps
n_elec_steps = 10  # Electronic sub-steps per nuclear step
freq = 10         # Saving frequency

# LiH
mol = gto.M(
    atom='Li 1.5 0.0 0.0; H 0 0 0',
    basis='6-31g',
    unit='Bohr',
    cart=True,
    symmetry="Coov",
    spin=0,
    verbose=0
)
ne_cas = (1, 1)
no_cas = 5

# Detect AO mask for A1 irrep
irrep_ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, np.eye(mol.nao))
ao_mask = (irrep_ids == 0)
print(f"Full AO count: {mol.nao}")
print(f"Truncated AO count (A1 only): {np.sum(ao_mask)}")

# Initialize pre_BO object with truncation
pb = pre_BO(mol, ne_cas, no_cas, ao_mask=ao_mask)

# --- Initial State Initialization ---
# HF on full basis first
mf_full = scf.RHF(mol)
mf_full.run(conv_tol=1e-8)

# SA-CASSCF for initial orbitals on FULL basis
cas_scf = mcscf.CASSCF(mf_full, no_cas, ne_cas)
cas_scf.fcisolver = fci.solver(mol, singlet=True)
cas_scf.fcisolver.nroots = 5
weights = np.ones(5) / 5
cas_scf.state_average_(weights)
cas_scf.kernel()

mo_full_init = np.copy(cas_scf.mo_coeff)

# Identify which of these full MOs are A1
irrep_ids_mo = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_full_init)
mo_a1_full = mo_full_init[:, irrep_ids_mo == 0]

# Now project these A1 MOs into our truncated AO space
pb.mo_coeff = mo_a1_full[pb.ao_mask, :pb.nmo]

# Re-orthonormalize in truncated space to be safe
s_mo = pb.mo_coeff.T @ pb.s_ao @ pb.mo_coeff
w, v = scipy.linalg.eigh(s_mo)
pb.mo_coeff = pb.mo_coeff @ v @ np.diag(1.0/np.sqrt(w)) @ v.T

# Setup CASCI for logging/initial CI (without symmetry)
mol_no_sym = mol.copy()
mol_no_sym.symmetry = False
mc_casci = mcscf.CASCI(mol_no_sym, no_cas, ne_cas)
mc_casci.fcisolver = csf_fci.csf_solver(mol_no_sym, smult=1)
ncsf_cut = 5
mc_casci.fcisolver.nroots = ncsf_cut

# For initial CI, use full-basis representation of our truncated MOs
def get_full_mo(mo_trunc, mask, nao_full):
    mo_full = np.zeros((nao_full, mo_trunc.shape[1]))
    mo_full[mask, :] = mo_trunc
    return mo_full

mo_full_ci = get_full_mo(pb.mo_coeff, pb.ao_mask, mol.nao)
mc_casci.mo_coeff = mo_full_ci
mc_casci.kernel()
casci_ci_vec = np.array(pb.transformer.vec_det2csf(mc_casci.ci))

# 3. Setup Step 0 Frame
pb.mol_old = mol.copy()
pb.mo_coeff_old = np.copy(pb.mo_coeff)

# Set Initial Electronic state (e.g., pure Ground State CSF)
istate = 1
pb.csf_coeff[:] = casci_ci_vec[istate, :] + 0.0j

# Get properties at t0
pb.mc.mo_coeff = mo_full_ci
pb.get_int_ao()
pb.get_V_csf()
pb.get_grad_ao()
pb.get_td_D_csf(block_orth=True, finite_difference=True, dt=dt)
pb.get_dV_csf_2()
pb.calculate_force_2()
rforce = pb.force.copy()

# --- Dynamics Loop ---
print("# Time Kinetic Potential Total Populations...")
energy_file = open("energy.dat", "w")
casci_file = open("casci.dat", "w")
pop_file = open("pop.dat", "w")
pop_bo_file = open("pop_bo.dat", "w")
movie_file = open("movie.xyz", "w")

# Write headers
output_e = f"#Time Kinetic_e Potential_e Total_e Norm\n"
output_f = f"#Time " + " ".join([f"E_{i}" for i in range(ncsf_cut)]) + "\n"
energy_file.write(output_e)
casci_file.write(output_f)
pop_file.write("# Time " + " ".join([f"Pop_{i}" for i in range(pb.ncsf)]) + "\n")
pop_bo_file.write("# Time " + " ".join([f"Pop_{i}" for i in range(ncsf_cut)]) + "\n")

for istep in range(nstep + 1):
    # 1. Output current state
    kinetic = 0.5 * np.sum(pb.mass[:, None] * pb.vel**2)
    potential = np.einsum('I, IJ, J ->', pb.csf_coeff.conj(), pb.V_csf, pb.csf_coeff).real + pb.V_core
    total_e = kinetic + potential

    if istep % freq == 0:
        if istep > 0:
            # Perform CASCI logging using the SEPARATE mc_casci object
            mc_casci.mol = pb.mol
            mo_full_ci = get_full_mo(pb.mo_coeff, pb.ao_mask, mol.nao)
            mc_casci.mo_coeff = mo_full_ci
            mc_casci.kernel()
            casci_ci_vec = np.array(pb.transformer.vec_det2csf(mc_casci.ci))

        # Calculate CSF populations
        pops = np.abs(pb.csf_coeff)**2
        pops_bo = np.absolute(np.einsum('iI, I->i', casci_ci_vec[:, :], pb.csf_coeff[:]))**2

        output_e = f"{istep*dt} {kinetic:.8f} {potential:.8f} {total_e:.8f} {np.sum(pops)}"
        output_f = f"{istep*dt} " + " ".join([f"{e:.8f}" for e in mc_casci.e_tot])
        pop_str = " ".join([f"{p:.6f}" for p in pops])
        pop_bo_str = " ".join([f"{p:.6f}" for p in pops_bo])
        energy_file.write(output_e + "\n")
        casci_file.write(output_f + "\n")
        pop_file.write(f"{istep*dt} {pop_str}\n")
        pop_bo_file.write(f"{istep*dt} {pop_bo_str}\n")

        print(output_e)

        movie_file.write(f"{pb.nat}\n\n")
        for iat in range(pb.nat):
            movie_file.write(f"{pb.mol.elements[iat]} {pb.pos[iat,0]*0.529177:.6f} {pb.pos[iat,1]*0.529177:.6f} {pb.pos[iat,2]*0.529177:.6f}\n")
        energy_file.flush()
        casci_file.flush()
        pop_file.flush()
        pop_bo_file.flush()
        movie_file.flush()

    if istep == nstep: break

    pb.backup_elec_state()

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
    # local ao, lowdin
    s_tmp = pb.s_ao
    s_vals, s_vecs = scipy.linalg.eigh(s_tmp)
    pb.mo_coeff = s_vecs @ np.diag(1.0/np.sqrt(s_vals)) @ s_vecs.T
    pb.align_mo_coeff_spacewise(pb.mol_old, pb.mo_coeff_old)

    pb.get_V_csf()
    pb.get_grad_ao()
    pb.get_td_D_csf(block_orth=True, finite_difference=True, dt=dt)
    pb.get_dV_csf_2()

    # Electronic propagation
    pb.propagate_elec(dt, nsteps=n_elec_steps)

    # Calculate force for Step 2
    pb.calculate_force_2()
    rforce = pb.force.copy()

    # Velocity Verlet Step 2
    pb.vel += 0.5 * dt * rforce / pb.mass[:, None]

    pb.mol_old = pb.mol.copy()
    pb.mo_coeff_old = np.copy(pb.mo_coeff)

energy_file.close()
pop_file.close()
movie_file.close()
