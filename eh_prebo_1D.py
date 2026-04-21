import numpy as np
import scipy.linalg
from pre_bo_1D import pre_BO_1D, get_full_mo
from pyscf import gto
import os, time

class Eh_PreBO_1D:
    def __init__(self, calculators, q, v, dt=10.0, nsteps=1000, nesteps=20, l_hff=False, l_tdnac=False):
        """
        Reduced CTv2 driver for 1D.
        Expects properties from pre_BO_1D (all scalars/NSTxNST matrices).
        l_hff: Use Hellmann-Feynman force.
        l_tdnac: Use time-derivative coupling.
        """
        self.ntrajs = len(calculators)
        self.dt, self.nsteps, self.nesteps = dt, nsteps, nesteps
        self.l_hff = l_hff
        self.l_tdnac = l_tdnac
        
        self.calcs = calculators
        self.nst = self.calcs[0].ncsf
        self.m_eff = self.calcs[0].m_eff # scalar reduced mass
        m0, m1 = self.calcs[0].mass[0], self.calcs[0].mass[1]
        M = m0 + m1
        w0, w1 = m1/M, -m0/M
        self.nst_pair = int(self.nst * (self.nst - 1) / 2)
        
        # 1D State (scalars)
        self.R = np.zeros((self.ntrajs, 1))
        self.V = np.zeros((self.ntrajs, 1))
        self.V_old = np.zeros_like(self.V)
        self.C = np.zeros((self.ntrajs, self.nst), dtype=np.complex128)
        
        # Initialization
        self.R[:, 0] = np.copy(q)
        self.V[:, 0] = np.copy(v)
        self.V_old[:, 0] = np.copy(v)
        for i in range(self.ntrajs):
            self.C[i, :] = np.copy(self.calcs[i].csf_coeff)
            self.calcs[i].mol_old = self.calcs[i].mol.copy()
            self.calcs[i].mo_coeff_old = np.copy(self.calcs[i].mo_coeff)

        # Ehrenfest variables
        self.rho = np.zeros((self.ntrajs, self.nst))
        
        self.energy_files = [open(f"energy_{i}.dat", "w") for i in range(self.ntrajs)]
        self.rho_csf_files = [open(f"rho_csf_{i}.bin", "wb") for i in range(self.ntrajs)]
        self.rho_bo_files = [open(f"rho_bo_{i}.bin", "wb") for i in range(self.ntrajs)]
        self.e_fci_files = [open(f"e_fci_{i}.bin", "wb") for i in range(self.ntrajs)]
        self.v_csf_files = [open(f"v_csf_{i}.bin", "wb") for i in range(self.ntrajs)]
    
    def calculate_properties_hff(self, itraj):
        calc = self.calcs[itraj]
        # Backup for interpolation
        calc.V_csf_old = np.copy(calc.V_csf) if calc.V_csf is not None else None
        self.V_old[itraj] = self.V[itraj]

        # Update geometry with COM weights for physical overlap
        q = self.R[itraj, 0]
        m1, m2 = calc.mass[0], calc.mass[1]
        M = m1 + m2
        w0, w1 = m2/M, -m1/M
        R_cart = np.zeros((2, 3))
        R_cart[0, 0] = w0 * q
        R_cart[1, 0] = w1 * q
        calc.mol.set_geom_(R_cart, unit='Bohr')
        calc.mol.build()
        
        calc.get_int_ao()
        s_tmp = calc.s_ao
        s_vals, s_vecs = scipy.linalg.eigh(s_tmp)
        calc.mo_coeff = s_vecs @ np.diag(1.0/np.sqrt(s_vals)) @ s_vecs.T
        calc.align_mo_coeff(calc.mol_old, calc.mo_coeff_old, symmetrize=True)
        
        calc.get_grad_coeff()
        calc.get_V_csf()
        calc.get_dV_csf()
        calc.get_D_csf()

        calc.mc.reset(calc.mol)
        mo_full = get_full_mo(calc.mo_coeff, calc.ao_mask, calc.mol.nao)
        calc.mc.mo_coeff = np.copy(mo_full)
        calc.mc.kernel()
        
        calc.csf_coeff = np.copy(self.C[itraj])
        calc.calculate_force()

    def calculate_properties(self, itraj):
        calc = self.calcs[itraj]
        # Backup for interpolation
        calc.V_csf_old = np.copy(calc.V_csf) if calc.V_csf is not None else None
        calc.D_csf_old = np.copy(calc.D_csf) if calc.D_csf is not None else None
        if self.l_tdnac:
            calc.td_D_csf_old = np.copy(calc.td_D_csf) if hasattr(calc, 'td_D_csf') else None
            
        self.V_old[itraj] = self.V[itraj]

        # Update geometry with COM weights for physical overlap
        q = self.R[itraj, 0]
        m1, m2 = calc.mass[0], calc.mass[1]
        M = m1 + m2
        w0, w1 = m2/M, -m1/M
        R_cart = np.zeros((2, 3))
        R_cart[0, 0] = w0 * q
        R_cart[1, 0] = w1 * q
        calc.mol.set_geom_(R_cart, unit='Bohr')
        calc.mol.build()
        
        calc.get_int_ao()
        s_tmp = calc.s_ao
        s_vals, s_vecs = scipy.linalg.eigh(s_tmp)
        calc.mo_coeff = s_vecs @ np.diag(1.0/np.sqrt(s_vals)) @ s_vecs.T
        calc.align_mo_coeff(calc.mol_old, calc.mo_coeff_old, symmetrize=True)
        
        calc.get_grad_coeff()
        calc.get_V_csf()
        
        if self.l_hff:
            calc.get_dV_csf_2()
        else:
            calc.get_dV_csf()
            
        calc.get_D_csf()
        
        if self.l_tdnac:
            calc.v_q = self.V[itraj, 0]
            calc.get_td_D_csf()

        calc.mc.reset(calc.mol)
        mo_full = get_full_mo(calc.mo_coeff, calc.ao_mask, calc.mol.nao)
        calc.mc.mo_coeff = np.copy(mo_full)
        calc.mc.kernel()
        
        calc.csf_coeff = np.copy(self.C[itraj])
        if self.l_hff:
            calc.calculate_force_2()
        else:
            calc.calculate_force()

    def propagate_elec(self, itraj):
        calc = self.calcs[itraj]
        dtau = self.dt / self.nesteps
        c = self.C[itraj]
        
        if self.l_tdnac:
            M_old = -1j * calc.V_csf_old - calc.td_D_csf_old
            M_new = -1j * calc.V_csf - calc.td_D_csf
        else:
            v_old, v_new = (calc.V_csf_old if calc.V_csf_old is not None else calc.V_csf), calc.V_csf
            d_old, d_new = (calc.D_csf_old if calc.D_csf_old is not None else calc.D_csf), calc.D_csf
            vel_old, vel_new = self.V_old[itraj, 0], self.V[itraj, 0]
            
        for i in range(self.nesteps):
            t_frac = (i + 0.5) / self.nesteps
            def get_cdot(c_curr, t):
                if self.l_tdnac:
                    M_t = M_old * (1 - t) + M_new * t
                    return M_t @ c_curr - (k_mat @ np.abs(c_curr)**2) * c_curr
                else:
                    vt = v_old * (1-t) + v_new * t
                    dt = d_old * (1-t) + d_new * t
                    velt = vel_old * (1-t) + vel_new * t
                    # i*cdot = V*c - i*v_q*D*c
                    return -1j * vt @ c_curr - velt * dt @ c_curr

            k1 = get_cdot(c, i/self.nesteps)
            k2 = get_cdot(c + 0.5*dtau*k1, (i+0.5)/self.nesteps)
            k3 = get_cdot(c + 0.5*dtau*k2, (i+0.5)/self.nesteps)
            k4 = get_cdot(c + dtau*k3, (i+1)/self.nesteps)
            c += (dtau/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            c /= np.linalg.norm(c)
        self.C[itraj] = c

    def log_results(self, istep):
        dt = self.dt
        for i in range(self.ntrajs):
            calc = self.calcs[i]
            ekin = 0.5 * self.m_eff * self.V[i, 0]**2
            epot = calc.V_core + np.einsum('I, IJ, J', self.C[i].conj(), calc.V_csf, self.C[i]).real
            q = self.R[i, 0]
            # ASCII: Time, Bondlength, Ekin, Epot, Etot
            self.energy_files[i].write(f"{istep*dt:15.8f} {q:15.8f} {ekin:15.8f} {epot:15.8f} {ekin+epot:15.8f}\n")
            self.energy_files[i].flush()
            
            # Diagonalize for BO populations and FCI energies
            #e_fci, v_adi = np.linalg.eigh(calc.V_csf)
            #c_bo = v_adi.T.conj() @ self.C[i]
            e_fci = np.copy(calc.mc.e_tot)
            v_adi = np.array(calc.transformer.vec_det2csf(calc.mc.ci))
            c_bo = v_adi @ self.C[i]

            rho_bo = np.abs(c_bo)**2
            rho_csf = np.abs(self.C[i])**2
            
            # Binary Logging (Float64)
            self.rho_csf_files[i].write(rho_csf.tobytes())
            self.rho_bo_files[i].write(rho_bo.tobytes())
            self.e_fci_files[i].write(e_fci.tobytes())
            self.v_csf_files[i].write(calc.V_csf.tobytes())
            
            self.rho_csf_files[i].flush()
            self.rho_bo_files[i].flush()
            self.e_fci_files[i].flush()
            self.v_csf_files[i].flush()

    def run(self, freq=10):
        dt = self.dt
        for i in range(self.ntrajs): self.calculate_properties(i)
        for istep in range(self.nsteps + 1):
            if istep % freq == 0:
                if istep % 10 == 0: print(f"Step {istep}")
                self.log_results(istep)
            if istep == self.nsteps: break
            
            for i in range(self.ntrajs):
                f_total = self.calcs[i].force
                self.V[i, 0] += 0.5 * dt * f_total / self.m_eff
                self.R[i, 0] += dt * self.V[i, 0]
            for i in range(self.ntrajs):
                self.calcs[i].mol_old = self.calcs[i].mol.copy(); self.calcs[i].mo_coeff_old = np.copy(self.calcs[i].mo_coeff)
                self.calculate_properties(i)
            for i in range(self.ntrajs): self.propagate_elec(i)
            for i in range(self.ntrajs):
                f_total = self.calcs[i].force
                self.V[i, 0] += 0.5 * dt * f_total / self.m_eff
