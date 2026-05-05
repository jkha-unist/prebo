import sys
import os
import numpy as np
from pyscf import gto, scf, symm, mcscf, csf_fci
# Add parent directory to path to find pre_bo_1D and ct_prebo_1D
root_dir = '/home/jkha/01_PROJECTS/07_direct_dynamics/code/prebo'
if root_dir not in sys.path:
    sys.path.append(root_dir)

from pre_bo_1D import pre_BO_1D, get_full_mo
from ct_prebo_1D import CTv2_PreBO_1D


# --- Simulation Parameters ---
ntrajs = 200
dt = 2.0
nsteps = 500
nesteps = 100
ne_cas = (2, 2)
ncsf_cut = 4

# Seed for reproducibility
np.random.seed(42)

# --- Initialize Trajectories ---
calculators = []
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
q_initial = np.random.normal(q_mean, q_sigma, ntrajs)

k_mean = 0.0
k_sigma = 0.5 / q_sigma
v_initial = np.random.normal(k_mean, k_sigma, ntrajs) / mu

print(f"Sampled initial q: {q_initial}")
print(f"Sampled initial v: {v_initial}")

for i in range(ntrajs):
    q = q_initial[i]
    R0 = w0 * q # Li
    R1 = w1 * q # H
    # Initial geometry (COM at origin not strictly needed yet, driver will handle)
    mol = gto.M(
        atom=f'Li {R0} 0 0; H {R1} 0 0',
        basis='sto-3g',
        unit='Bohr',
        cart=True,
        symmetry="Coov",
        spin=0,
        verbose=0
    )
    
    # Get AO mask for Sigma orbitals while symmetry is on
    irrep_ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, np.eye(mol.nao))
    ao_mask = (irrep_ids == 0)
    no_cas = np.sum(ao_mask)
    
    # Disable symmetry for dynamics
    #mol.symmetry = False
    
    # Initialize calculator
    pb = pre_BO_1D(mol, ne_cas, no_cas, ao_mask=ao_mask)
    
    # Initial MOs from HF (needed for alignment later)
    mf = scf.RHF(mol)
    mf.run(conv_tol=1e-8)
    mo_full = mf.mo_coeff
    # Label MOs
    irrep_ids_mo = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_full)
    
    pb.mo_coeff = mo_full[pb.ao_mask, :][:, irrep_ids_mo == 0]

    pb.mol_old = pb.mol.copy()
    pb.mo_coeff_old = np.copy(pb.mo_coeff)
    
    # Ensure orthogonality in truncated space
    s_mo = pb.mo_coeff.T @ pb.s_ao @ pb.mo_coeff
    w, v = np.linalg.eigh(s_mo)
    pb.mo_coeff = pb.mo_coeff @ v @ np.diag(1.0/np.sqrt(w)) @ v.T
    mo_full_ci = get_full_mo(pb.mo_coeff, pb.ao_mask, mol.nao)
    
    nroots = ncsf_cut
    
    pb.mc.fcisolver.nroots = nroots # OVERRIDE
    pb.mc.mo_coeff = np.copy(mo_full_ci)
    pb.mc.kernel()

    fci_ci_vec = np.array(pb.transformer.vec_det2csf(pb.mc.ci))
    pb.csf_coeff[:] = fci_ci_vec[istate, :] + 0.0j
    
    calculators.append(pb)

# --- Setup and Run CTv2 Driver ---
# Path to results
#os.chdir('/home/jkha/01_PROJECTS/07_direct_dynamics/code/prebo/ct_dyn/')

dyn = CTv2_PreBO_1D(
    q=q_initial,
    v=v_initial,
    calculators=calculators,
    dt=dt,
    nsteps=nsteps,
    nesteps=nesteps,
    l_crunch=True,
    t_pc=0,
    t_cons=2,
    l_etot0=True,
    t_pot=1
)
print("Starting CTv2 1D dynamics...")
dyn.run(freq=5)
print("Dynamics completed.")
