import numpy as np
from pyscf import gto, lib
from pre_bo import pre_BO
import math, os
import pickle

os.environ["OMP_NUM_THREADS"] = "10"

# =============================================================================
# 1. CONFIGURATION & SETUP
# =============================================================================
npt = 100           # Number of points from center to edge
xmax = 1.0
dx = xmax / npt               # Step size (dimensionless or Bohr, depends on g/h units)
grid_size = 2 * npt + 1 # Total grid dimension (e.g., 201)
#xmax = dx * npt        # Total length: 2*L

# Simulation parameters
ne_cas = (1, 1)
no_cas = 2
spin = 0

# CI
mol = gto.M(
    #atom="../../../01_casscf/01_ciopt/opt.xyz",
    atom="./opt.xyz",
    basis='6-31G**',
    unit='Angstrom',
    verbose=0
)
coords0 = mol.atom_coords(unit='Bohr')

init_pre_bo = pre_BO(mol, ne_cas, no_cas)
ncsf = init_pre_bo.ncsf

#g_vec = np.loadtxt("../../../01_casscf/02_scan/pes/g.dat")
#h_vec = np.loadtxt("../../../01_casscf/02_scan/pes/h.dat")
g_vec = np.loadtxt("./g.dat")
h_vec = np.loadtxt("./h.dat")

grid_size = 2 * npt + 1
mols_grid = np.empty((grid_size, grid_size), dtype=object)
for i in range(grid_size):
    for j in range(grid_size):
        mols_grid[i, j] = mol.copy()
mos_grid  = np.zeros((grid_size, grid_size,  mol.nao, mol.nao)) 
V  = np.zeros((grid_size, grid_size, ncsf, ncsf)) 
E  = np.zeros((grid_size, grid_size, ncsf)) 
E_casci  = np.zeros((grid_size, grid_size, ncsf)) 
V_core  = np.zeros((grid_size, grid_size)) 
D = np.zeros((grid_size, grid_size, mol.natm, 3, ncsf, ncsf))
D_g = np.zeros((grid_size, grid_size, ncsf, ncsf))
D_h = np.zeros((grid_size, grid_size, ncsf, ncsf))
D_z = np.zeros((grid_size, grid_size, ncsf, ncsf))

# Walker
#def solve_point(ix, iy, mols_grid, mos_grid, V, E, D, old_ix=None, old_iy=None):
def solve_point(ix, iy, old_ix=None, old_iy=None):
    """
    Calculates point (ix, iy).
    Args:
        ix, iy: Current target index
        mols_grid, mos_grid: The storage arrays (passed explicitly)
        old_ix, old_iy: The reference neighbor index (if None, assumes Origin)
    """

    #mol = mols_grid[ix, iy].copy()

    # Calculate displacement from center (npt)
    x_disp = (ix - npt) * dx
    y_disp = (iy - npt) * dx

    # Calculate new coordinates: R_new = R_0 + x*g + y*h
    # Note: Ensure g_vec/h_vec are broadcastable to coords0 shape
    disp = x_disp * g_vec + y_disp * h_vec
    new_coords = coords0 + disp

    # Define the molecule for this specific grid point
    #mol.set_geom_(new_coords, unit='Bohr')
    mols_grid[ix, iy].set_geom_(new_coords, unit='Bohr')

    # Initialize pre_BO
    driver = pre_BO(mols_grid[ix, iy], ne_cas, no_cas)

    # Propagate MO
    if old_ix is not None:
        # Retrieve reference data from the "old" neighbor in the grid
        mol_old = mols_grid[old_ix, old_iy].copy()
        mo_old  = mos_grid[old_ix, old_iy].copy()
        
        # Obtain HF orbital
        driver.mf.init_guess = 'mo'
        driver.mf.mo_coeff = mo_old
        driver.mf.run(conv_tol=1e-8)
        driver.mo_coeff = np.copy(driver.mf.mo_coeff)

        # Run diabatic alignment
        driver.align_mo_coeff(mol_old=mol_old, mo_old=mo_old)

    else: # The first point, i.e. calculation at the CI
        #old_chk_file = "../../..//01_casscf/01_ciopt/opt.chk"
        old_chk_file = "./opt.chk"
        driver.mo_coeff = lib.chkfile.load(old_chk_file, 'mcscf/mo_coeff')
        

    # 5. Calculate MO integrals
    driver.get_int_mo()
    
    # 6. Calculate CSF matrix elements
    driver.mc.mo_coeff = np.copy(driver.mo_coeff)
    driver.get_V_csf()
    driver.get_D_csf()

    # Store the molecule and mo so neighbors can use them later
    #mols_grid[ix, iy] = mol.copy()
    mos_grid[ix, iy]  = driver.mo_coeff.copy()

    # Store data
    V_core[ix, iy] = driver.e_core
    V[ix, iy, :, :] = driver.V_csf[:, :]
    
    eigval, eigvec = np.linalg.eig(V[ix, iy])
    idx = np.argsort(eigval)
    E[ix, iy, :] = eigval[idx]
    
    E_casci[ix, iy, :] = driver.mc.kernel()[0]
    
    D[ix, iy, :, :, :, :] = driver.D_csf[:, :, :, :]

print(">>> Starting Scan")

# Center
solve_point(npt, npt) 

# Spine
for i in range(1, npt + 1):
    solve_point(npt + i, npt, old_ix=npt + i - 1, old_iy=npt)
    solve_point(npt - i, npt, old_ix=npt - i + 1, old_iy=npt)
    
print("Spine done")

# Ribs
for iix in range(grid_size):
    for j in range(1, npt + 1):
        solve_point(iix, npt + j, old_ix=iix, old_iy=npt + j - 1)
        solve_point(iix, npt - j, old_ix=iix, old_iy=npt - j + 1)
    print(f"Rib {iix} done")

for ix in range(grid_size):
    for iy in range(grid_size):
        d_norm = np.einsum('RqIJ, RqIJ -> IJ', D[ix, iy, :, :, :, :], D[ix, iy, :, :, :, :])
        D_g[ix, iy, :, :] = np.einsum('RqIJ,Rq -> IJ', D[ix, iy, :, :, :, :], g_vec[:, :])
        D_h[ix, iy, :, :] = np.einsum('RqIJ,Rq -> IJ', D[ix, iy, :, :, :, :], h_vec[:, :])
        D_z[ix, iy, :, :] = np.sqrt(d_norm[:, :] - D_g[ix, iy, :, :] ** 2 - D_h[ix, iy, :, :] ** 2)

grid = np.linspace(-xmax, xmax, 2*npt+1)
f_e = open("E.dat", "w")
f_v = open("V.dat", "w")
f_d = open("D.dat", "w")
f_e.write(f"#g   h")
f_v.write(f"#g   h   V_core")
f_d.write(f"#g   h")

for icsf in range(ncsf):
    f_e.write(f"     E({icsf})")
    for jcsf in range(icsf, ncsf):
        f_v.write(f"     V({icsf},{jcsf})")
        f_d.write(f"     D({icsf},{jcsf},g)     D({icsf},{jcsf},h)     D({icsf},{jcsf},z)")
for icsf in range(ncsf):
    f_e.write(f"     E_casci({icsf})")

f_e.write("\n")
f_v.write("\n")
f_d.write("\n")

for i, x in enumerate(grid):
    for j, y in enumerate(grid):
        f_e.write(f"{x:7.3f}   {y:7.3f}")
        f_v.write(f"{x:7.3f}   {y:7.3f}   {V_core[i, j]:.10f}")
        f_d.write(f"{x:7.3f}   {y:7.3f}")
        for ist in range(ncsf):
            f_e.write(f"  {E[i, j, ist]:.10f}")
            for jst in range(ist, ncsf):
                f_v.write(f" {V[i, j, ist, jst]:.10f}")
                f_d.write(f" {D_g[i, j, ist, jst]:.10f} {D_h[i, j, ist, jst]:.10f} {D_z[i, j, ist, jst]:.10f}")
        for ist in range(ncsf):
            f_e.write(f"  {E_casci[i, j, ist]:.10f}")
        f_e.write("\n")
        f_v.write("\n")
        f_d.write("\n")
    f_e.write("\n")
    f_v.write("\n")
    f_d.write("\n")
f_e.close()
f_v.close()
f_d.close()

fn_mol_mo = "mol_mo.pkl"
with open(fn_mol_mo, "wb") as f_mol_mo:
    pickle.dump({"npt":npt, "xmax":xmax, "mols": mols_grid, "mos": mos_grid}, f_mol_mo)

