# prebo: Pre-Born-Oppenheimer Direct Dynamics Simulation

`prebo` is a pre-Born-Oppenheimer (pre-BO) direct dynamics simulation code for molecular systems. It performs nuclear dynamics on diabatic potential energy surfaces constructed from Configuration State Functions (CSFs) using [PySCF](https://pyscf.org/) as the electronic structure backend.

## Overview

The code performs Ehrenfest-like dynamics where the electronic state is expanded in a basis of CSFs. It handles the evolution of both nuclear positions and electronic coefficients, including the tracking and propagation of Molecular Orbitals (MOs) to ensure consistent diabatic states across the trajectory.

## Key Features

- **Electronic Structure**: CASCI calculations using a CSF basis via the `csf_fci` module (available in the [pyscf-forge](https://github.com/pyscf/pyscf-forge) extension).
- **MO Tracking/Alignment**:
  - **Discrete**: Löwdin-tracked MOs via inter-geometry overlap and polar decomposition.
  - **Continuous**: Time propagation of MO coefficients via matrix exponential or sub-stepped integration.
- **Gradients and Couplings**:
  - Analytical nuclear gradients of the CSF Hamiltonian (Hellmann-Feynman + Pulay terms).
  - Analytical MO gradients using Sylvester equations.
  - Derivative couplings and time-derivative couplings between CSFs.
- **Dynamics**: Velocity Verlet integration for nuclear motion.

## Dependencies

- Python 3
- [PySCF](https://github.com/pyscf/pyscf)
- [pyscf-forge](https://github.com/pyscf/pyscf-forge) (for the `csf_fci` module)
- NumPy
- SciPy

## Installation

1. Ensure you have Python 3 installed.
2. Install the required dependencies:
   ```bash
   pip install numpy scipy pyscf
   ```
3. Install the `pyscf-forge` extension to enable CSF-based CASCI:
   ```bash
   pip install git+https://github.com/pyscf/pyscf-forge.git
   ```

## Usage

Dynamics simulations are driven by `dyn.py` scripts located in various subdirectories (e.g., `lih_active/`, `lih_fci/`). These scripts configure the simulation parameters and drive the Velocity Verlet loop.

To run a simulation, navigate to a directory containing a `dyn.py` script and execute:
```bash
python dyn.py
```

### Output Files
The simulation generates several output files:
- `energy.dat`: Potential, kinetic, and total energies over time.
- `movie.xyz`: Trajectory coordinates in XYZ format.
- `pop.dat` / `pop_bo.dat`: Population evolution of the electronic states.

## Key Conventions

- **Units**: All internal coordinates and properties are in **atomic units** (Bohr, Hartree, atomic mass units × 1822.89).
- **Tensor Operations**: Einstein summation (`np.einsum`) is used extensively for tensor contractions.
- **Coordinate Gradients**: Gradient arrays generally follow the shape `(nat, 3, ...)`, where `nat` is the number of atoms.

## Project Structure

- `pre_bo.py`: Core `pre_BO` class containing electronic structure and dynamics logic (integrals, CSF Hamiltonian, MO alignment, gradients, propagation).
- `pre_bo_symm.py`: A variant of the `pre_BO` class that incorporates point group symmetry (via PySCF's `symm` module).
- `lih_active/`, `lih_fci/`: Directories containing simulation setups for LiH using different active spaces and method variants (discrete vs. continuous propagation, with or without Hellmann-Feynman forces).
- `lih_active_symm/`, `lih_fci_symm/`: Simulation setups for LiH that utilize molecular symmetry.

## Architecture

The `pre_BO` class in `pre_bo.py` is the central component, handling:
1. **Integrals**: AO/MO overlap, 1e, 2e integrals and their gradients.
2. **CSF Hamiltonian**: Building the diabatic Hamiltonian matrix and its nuclear gradients.
3. **MO Management**: Alignment and tracking of MOs across geometries.
4. **Propagation**: Both MO and electronic coefficient propagation.
5. **Forces**: Calculation of the Ehrenfest force for nuclear dynamics.
