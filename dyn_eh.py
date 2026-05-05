import sys
import os
# Ensure root directory is in path
root_dir = '/home/jkha/01_PROJECTS/07_direct_dynamics/code/prebo'
if root_dir not in sys.path:
    sys.path.append(root_dir)
import numpy as np
from pyscf import gto, scf, mcscf, fci, csf_fci, symm
from pre_bo_1D import pre_BO_1D
import scipy
import math

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

# For initial CI, use full-basis representation of our truncated MOs
def get_full_mo(mo_trunc, mask, nao_full):
    mo_full = np.zeros((nao_full, mo_trunc.shape[1]))
    mo_full[mask, :] = mo_trunc
    return mo_full

# Simulation Control
dt = 1.0         # Nuclear time step (au)
nstep = 1000       # Number of steps
n_elec_steps = 100  # Electronic sub-steps per nuclear step
freq = 1          # Saving frequency
ntraj = 100
istate = 2

# Sample initial q from Normal(mean, sigma)
r_eq = 2.9243918278204553 # Bohr
omega = 0.00766228138128107# a.u.
m0, m1 = 6.941, 1.00794
M = m0 + m1
mu = m0*m1 / M * 1822.8884856192
w0, w1 = m1/M, -m0/M

q_mean = r_eq
q_sigma = np.sqrt(1.0 / mu / omega)
q_initial = np.random.normal(q_mean, q_sigma, ntraj)

k_mean = 0.0
k_sigma = 0.5 / q_sigma
v_initial = np.random.normal(k_mean, k_sigma, ntraj) / mu

for itraj in range(ntraj):
    pos = q_initial[itraj]
    vel = v_initial[itraj]
    R0 = w0 * pos # Li
    R1 = w1 * pos # H
    # LiH
    mol = gto.M(
        atom=f"Li {R0} 0.0 0.0; H {R1} 0 0",
        basis='sto-3g',
        unit='Bohr',
        symmetry="Coov",
        spin=0,
        verbose=0
    )
    ne_cas = (2, 2)
    no_cas = 4 # fci with sigma AOs
    
    # Detect AO mask for A1 irrep
    irrep_ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, np.eye(mol.nao))
    ao_mask = (irrep_ids == 0)
    print(f"Trajectory {itraj} start")
    
    # Initialize pre_BO object with truncation
    pb = pre_BO_1D(mol, ne_cas, no_cas, ao_mask=ao_mask)
    
    # --- Initial State Initialization ---
    # HF on full basis first to match reference starting point exactly
    mf_full = scf.RHF(mol)
    mf_full.run(conv_tol=1e-8)
    mo_full_init = np.copy(mf_full.mo_coeff)
    
    # Identify which of these full MOs are A1
    irrep_ids_mo = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_full_init)
    mo_a1_full = mo_full_init[:, irrep_ids_mo == 0]
    
    # Now project these 7 A1 MOs into our truncated AO space
    # In a perfect world, mo_a1_full only has non-zero values on A1 AOs.
    pb.mo_coeff = np.copy(mo_a1_full[pb.ao_mask, :])
    
    # Re-orthonormalize in truncated space to be safe
    s_mo = pb.mo_coeff.T @ pb.s_ao @ pb.mo_coeff
    w, v = scipy.linalg.eigh(s_mo)
    pb.mo_coeff = pb.mo_coeff @ v @ np.diag(1.0/np.sqrt(w)) @ v.T
    
    # Setup CASCI for initial state CI
    # Disable symmetry in CASCI to avoid internal re-symmetrization of truncated MOs
    #mol_no_sym = mol.copy()
    #mol_no_sym.symmetry = False
    mc_fci = mcscf.CASCI(mol, no_cas, ne_cas)
    mc_fci.fcisolver = csf_fci.csf_solver(mol, smult=1)
    ncsf_cut = 4
    mc_fci.fcisolver.nroots = ncsf_cut
    
    mo_full_ci = get_full_mo(pb.mo_coeff, pb.ao_mask, mol.nao)
    mc_fci.mo_coeff = np.copy(mo_full_ci)
    mc_fci.kernel()
    fci_ci_vec = np.array(pb.transformer.vec_det2csf(mc_fci.ci))
    
    # 3. Setup Step 0 Frame
    pb.mol_old = mol.copy()
    pb.mo_coeff_old = np.copy(pb.mo_coeff)
    pb.align_mo_coeff(pb.mol_old, pb.mo_coeff_old, local=True)
    
    # Set Initial Electronic state (e.g., pure Excited State CSF)
    pb.csf_coeff[:] = fci_ci_vec[istate, :] + 0.0j
    
    # Get properties at t0
    pb.mc.mo_coeff = np.copy(mo_full_ci)
    pb.get_int_ao()
    pb.get_V_csf()
    pb.get_grad_ao()
    pb.get_grad_coeff(local=True)
    #pb.get_int_mo()
    pb.get_D_csf()
    pb.get_dV_csf()
    pb.calculate_force()
    rforce = pb.force.copy()
    
    # --- Dynamics Loop ---
    print("# Time Kinetic Potential Total Populations...")
    energy_file = open(f"energy_{itraj}.dat", "w")
    fci_file = open(f"fci_{itraj}.dat", "w")
    pop_file = open(f"pop_csf_{itraj}.dat", "w")
    pop_bo_file = open(f"pop_bo_{itraj}.dat", "w")
    movie_file = open(f"movie.xyz_{itraj}", "w")
    
    # Write headers
    output_e = f"#Time Kinetic_e Potential_e Total_e Norm\n"
    output_f = f"#Time " + " ".join([f"E_{i}" for i in range(ncsf_cut)]) + "\n"
    energy_file.write(output_e)
    fci_file.write(output_f)
    pop_file.write("# Time " + " ".join([f"Pop_{i}" for i in range(pb.ncsf)]) + "\n")
    pop_bo_file.write("# Time " + " ".join([f"Pop_{i}" for i in range(ncsf_cut)]) + "\n")
    
    for istep in range(nstep + 1):
        # 1. Output current state
        kinetic = 0.5 * np.sum(pb.m_eff * vel ** 2)
        potential = np.einsum('I, IJ, J ->', pb.csf_coeff.conj(), pb.V_csf, pb.csf_coeff).real + pb.V_core
        total_e = kinetic + potential
        
        if istep % freq == 0:
            if istep > 0:
                # Perform FCI logging using the SEPARATE mc_fci object
                mc_fci.reset(pb.mol)
                mc_fci.mol = pb.mol
                mo_full_ci = get_full_mo(pb.mo_coeff, pb.ao_mask, mol.nao)
                mc_fci.mo_coeff = mo_full_ci
                mc_fci.kernel()
                fci_ci_vec = np.array(pb.transformer.vec_det2csf(mc_fci.ci))
            
            # Calculate CSF populations
            pops = np.abs(pb.csf_coeff)**2
            pops_bo = np.absolute(np.einsum('iI, I->i', fci_ci_vec[:, :], pb.csf_coeff[:]))**2
            
            output_e = f"{istep*dt} {kinetic:.8f} {potential:.8f} {total_e:.8f} {np.sum(pops)}"
            output_f = f"{istep*dt} " + " ".join([f"{e:.8f}" for e in mc_fci.e_tot])
            pop_str = " ".join([f"{p:.6f}" for p in pops])
            pop_bo_str = " ".join([f"{p:.6f}" for p in pops_bo])
            energy_file.write(output_e + "\n")
            fci_file.write(output_f + "\n")
            pop_file.write(f"{istep*dt} {pop_str}\n")
            pop_bo_file.write(f"{istep*dt} {pop_bo_str}\n")
            
            print(output_e)
            
            R0 = w0 * pos # Li
            R1 = w1 * pos # H
            movie_file.write(f"2\n\n")
            movie_file.write(f"Li {R0:.6f} 0.0 0.0 \n")
            movie_file.write(f"H {R1:.6f} 0.0 0.0 \n")
            energy_file.flush()
            fci_file.flush()
            pop_file.flush()
            pop_bo_file.flush()
            movie_file.flush()
    
        if istep == nstep: break
    
        # Velocity Verlet Step 1
        vel += 0.5 * dt * rforce / pb.m_eff
        pos += dt * vel
        R0 = w0 * pos # Li
        R1 = w1 * pos # H
        coords = np.array([[R0, 0.0, 0.0], [R1, 0.0, 0.0]])
        pb.mol.set_geom_(coords, unit='Bohr')
        pb.mol.build()
        
        # Properties at t + dt
        pb.get_int_ao()
        # local ao, lowdin
        s_tmp = pb.s_ao
        s_vals, s_vecs = scipy.linalg.eigh(s_tmp)
        pb.mo_coeff = s_vecs @ np.diag(1.0/np.sqrt(s_vals)) @ s_vecs.T
        pb.align_mo_coeff(pb.mol_old, pb.mo_coeff_old, local=True)
        
        pb.mc.reset(pb.mol)
        pb.get_V_csf()
        pb.get_grad_ao()
        pb.get_grad_coeff(local=True)
        #pb.get_int_mo()
        pb.get_D_csf()
        pb.get_dV_csf()
        
        # Electronic propagation
        c = pb.csf_coeff
        
        #dtau = dt / n_elec_steps
        #v_old, v_new = pb.V_csf_old, pb.V_csf
        #d_old, d_new = pb.D_csf_old, pb.D_csf
        #for i in range(n_elec_steps):
        #    t = (i + 0.5) / n_elec_steps
        #    vt = v_old * (1-t) + v_new * t
        #    dt_mat = d_old * (1-t) + d_new * t
        #    # i*cdot = V*c - i*vel*D*c
        #    cdot = -1j * vt @ c - vel * dt_mat @ c
        #    c += dtau * cdot
        #    c /= np.linalg.norm(c)
        #
        M = -1j * pb.V_csf - vel * pb.D_csf
        U = scipy.linalg.expm(M * dt)
        c = U @ c
    
        pb.csf_coeff = c
        
        # Calculate force for Step 2
        pb.calculate_force()
        rforce = pb.force.copy()
        
        # Velocity Verlet Step 2
        vel += 0.5 * dt * rforce / pb.m_eff
    
        pb.mol_old = pb.mol.copy()
        pb.mo_coeff_old = np.copy(pb.mo_coeff)
    
    energy_file.close()
    pop_file.close()
    movie_file.close()
