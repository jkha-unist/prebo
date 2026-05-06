from pyscf import gto, scf, mcscf, fci, nac, csf_fci, ao2mo, grad, symm
from pyscf.tools import mo_mapping, molden
from functools import reduce
import scipy
import numpy as np

def get_full_mo(mo_trunc, mask, nao_full):
    mo_full = np.zeros((nao_full, mo_trunc.shape[1]))
    mo_full[mask, :] = mo_trunc
    return mo_full

class pre_BO_1D(object):
    """
    Reduced-dimensionality pre_BO for diatomic molecules.
    All properties are calculated w.r.t internal coordinate q = x0 - x1.
    By definition, d/dq = d/dx0 (atom 0 moves, atom 1 fixed).
    """
    def __init__(self, mol, ne_cas, no_cas, nmo=None, spin=0, ao_mask=None):
        self.mol = mol 
        self.nat = mol.natm
        self.mass = gto.mole.atom_mass_list(mol, isotope_avg=True) * 1822.8884853324
        # m_eff is reduced mass
        self.m_eff = (self.mass[0] * self.mass[1]) / (self.mass[0] + self.mass[1])
        
        self.w = np.array([self.mass[1], -self.mass[0]]) / (self.mass[0] + self.mass[1])
        
        self.mol_old = mol.copy()
        
        if ao_mask is None and mol.symmetry:
            irrep_ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, np.eye(mol.nao))
            self.ao_mask = (irrep_ids == 0)
        else:
            self.ao_mask = ao_mask

        if self.ao_mask is not None:
            self.trunc_idx = np.where(self.ao_mask)[0]
            self.nao = len(self.trunc_idx)
            full_slices = self.mol.aoslice_by_atom()
            self.aoslices = []
            current_trunc_idx = 0
            for iat in range(self.nat):
                p0, p1 = full_slices[iat][2:4]
                num_kept = np.sum(self.ao_mask[p0:p1])
                self.aoslices.append((full_slices[iat][0], full_slices[iat][1], current_trunc_idx, current_trunc_idx + num_kept))
                current_trunc_idx += num_kept
        else:
            self.nao = mol.nao
            self.trunc_idx = np.arange(self.nao)
            self.aoslices = self.mol.aoslice_by_atom()

        self.nmo = nmo if nmo else self.nao
        self.ne_cas, self.no_cas, self.spin = ne_cas, no_cas, spin
        
        self.get_int_ao()
        self.mf = scf.RHF(self.mol)
        self.mo_coeff = np.zeros((self.nao, self.nmo))
        self.mo_coeff_old = np.zeros_like(self.mo_coeff)
        self.u = np.eye((self.nao))
        self.grad_coeff = np.zeros((self.nao, self.nmo)) # dC/dq
        
        self.transformer = csf_fci.CSFTransformer(self.no_cas, self.ne_cas[0], self.ne_cas[1], smult=2*spin+1)
        self.ncsf = self.transformer.ncsf
        self.mc = mcscf.CASCI(self.mol, no_cas, ne_cas)
        self.mc.fcisolver = csf_fci.csf_solver(self.mol, smult=2*spin+1)
        self.mc.fcisolver.nroots = self.ncsf
        self.ncore = self.mc.ncore
        self.nvirt = self.nmo - self.ncore - self.no_cas # number of core spatial orbital
        
        self.V_core = 0.
        self.V_csf = np.zeros((self.ncsf, self.ncsf))
        self.dV_core = 0.0 
        self.dV_csf = np.zeros((self.ncsf, self.ncsf)) 
        self.D_csf = np.zeros((self.ncsf, self.ncsf)) 
        self.td_D_csf = np.zeros((self.ncsf, self.ncsf))
        self.csf_coeff = np.zeros((self.ncsf), dtype=np.complex128)
        self.force = 0.0 
        self.v_q = 0.0
        
        self.V_csf_old = np.zeros((self.ncsf, self.ncsf))
        self.D_csf_old = np.zeros((self.nat, 3, self.ncsf, self.ncsf))
        self.td_D_csf_old = np.zeros((self.ncsf, self.ncsf))

    def get_int_ao(self):
        self.s_ao = self.mol.intor('int1e_ovlp')[self.trunc_idx, :][:, self.trunc_idx]
        self.kin_ao = self.mol.intor('int1e_kin')[self.trunc_idx, :][:, self.trunc_idx]
        self.nuc_ao = self.mol.intor('int1e_nuc')[self.trunc_idx, :][:, self.trunc_idx]
        self.v_ao = self.mol.intor('int2e', aosym='s1').transpose(0, 2, 1, 3)
        self.v_ao = self.v_ao[self.trunc_idx, :, :, :][:, self.trunc_idx, :, :][:, :, self.trunc_idx, :][:, :, :, self.trunc_idx]
        
        # d/dq = w0 * d/dx0 + w1 * d/dx1
        d_full = self.mol.intor('int1e_ipovlp').transpose(0,2,1)[0, self.trunc_idx, :][:, self.trunc_idx]
        self.d_ao = np.zeros((self.nao, self.nao))
        for iat in range(self.nat):
            p0, p1 = self.aoslices[iat][2:4]
            self.d_ao[:, p0:p1] = -self.w[iat] * d_full[:, p0:p1]

    def get_grad_ao(self):
        # d/dq = d/dx0
        #p0_0, p1_0 = self.aoslices[0][2:4]
        # PySCF ip intors are <nabla mu | h | nu>.
        # d/dx0 <mu|h|nu> = <d/dx0 mu|h|nu> + <mu|h|d/dx0 nu> + <mu|d/dx0 h|nu>
        # get_grad_ao only provides the primitive pieces.
        dkin_raw = self.mol.intor('int1e_ipkin')[0, self.trunc_idx, :][:, self.trunc_idx]
        dnuc_raw = self.mol.intor('int1e_ipnuc')[0, self.trunc_idx, :][:, self.trunc_idx]
        dv_raw = self.mol.intor('int2e_ip1').transpose(0, 1, 3, 2, 4)[0, self.trunc_idx, :, :, :][:, self.trunc_idx, :, :][:, :, self.trunc_idx, :][:, :, :, self.trunc_idx]
        
        self.dkin_ao = np.zeros((self.nao, self.nao))
        self.dnuc_ao = np.zeros((self.nao, self.nao))
        self.drinv_ao = np.zeros((self.nao, self.nao))
        self.dv_ao = np.zeros((self.nao, self.nao, self.nao, self.nao))
        for iat in range(self.nat):
            p0, p1 = self.aoslices[iat][2:4]
            self.dkin_ao[p0:p1, :] = -self.w[iat] * dkin_raw[p0:p1, :]
            self.dnuc_ao[p0:p1, :] = -self.w[iat] * dnuc_raw[p0:p1, :]
            self.dv_ao[p0:p1, :, :, :] = -self.w[iat] * dv_raw[p0:p1, :, :, :]
            with self.mol.with_rinv_at_nucleus(iat):
                drinv_iat = self.mol.intor('int1e_iprinv', comp=3)[0, self.trunc_idx, :][:, self.trunc_idx] * (-self.mol.atom_charge(iat))
                self.drinv_ao += self.w[iat] * drinv_iat
    
    def align_mo_coeff_simple(self, mol_old, mo_old):
        
        mo = np.copy(self.mo_coeff)
        #idx, s = mo_mapping.mo_map(mol_old, mo_old, self.mol, self.mo_coeff)
        s = gto.intor_cross('int1e_ovlp', mol_old, self.mol)[self.trunc_idx, :][:, self.trunc_idx]
        s = reduce(np.dot, (mo_old.T, s, mo))
        
        # detect swap
        col_ind = np.argmax(np.abs(s), axis=1) # i-th element is the index of new MO corresponding to the i-th old MO
        row_ind = np.arange(self.nao)
       
        # detect sign flip
        phases = np.sign(s[row_ind, col_ind])
        phases[phases == 0] = 1.0  # Prevent zeroing out orbitals if exact 0 overlap occurs, although it is highly unlikely.

        self.mo_coeff = mo[:, col_ind] * phases # bring new MO to i-th column (bring new index --> old index)

    def align_mo_coeff(self, mol_old, mo_old, diag_block_0=False, symmetrize=False, local=False):
        
        if (local):
            o_ao = np.copy(self.s_ao)
            symmetrize = False
            diag_block_0 = False
        else:
            o_ao = gto.intor_cross('int1e_ovlp', self.mol, mol_old)[self.trunc_idx, :][:, self.trunc_idx]

        if (diag_block_0):
            for iat in range(self.nat):
                p0, p1 = self.aoslices[iat][2:4]
                for io in range(p0, p1):
                    for jo in range(io+1, p1):
                        o_ao[io, jo] = 0.0
                        o_ao[jo, io] = 0.0
        if (symmetrize):
            o_ao = 0.5*(o_ao + o_ao.transpose(1,0))

        o = self.mo_coeff.T @ o_ao @ mo_old
        m_vals, m_vecs = scipy.linalg.eigh(o @ o.T)
        m_sqrt_inv = m_vecs @ np.diag(1.0 / np.sqrt(m_vals)) @ m_vecs.T
        self.u = m_sqrt_inv @ o
        self.mo_coeff = self.mo_coeff @ self.u

    def get_grad_coeff(self, local=False):
        
        s = np.copy(self.s_ao)
        s_vals, s_vecs = scipy.linalg.eigh(s)
        s_inv = s_vecs @ np.diag(1.0 / s_vals) @ s_vecs.T
        c_local = s_vecs @ np.diag(1.0 / np.sqrt(s_vals)) @ s_vecs.T
        
        if (local):
            o_ao = np.copy(self.s_ao)
        else:
            o_ao = gto.intor_cross('int1e_ovlp', self.mol, self.mol_old)[self.trunc_idx, :][:, self.trunc_idx]
        
        #if (diag_block_0):
        #    for iat in range(self.nat):
        #        p0, p1 = self.aoslices[iat][2:4]
        #        for io in range(p0, p1):
        #            for jo in range(io+1, p1):
        #                o_ao[io, jo] = 0.0
        #                o_ao[jo, io] = 0.0

        o = c_local @ o_ao @ self.mo_coeff_old
        m_vals, m_vecs = scipy.linalg.eigh(o @ o.T)
        m_inv = m_vecs @ np.diag(1.0 / m_vals) @ m_vecs.T
        m_sqrt_inv = m_vecs @ np.diag(1.0 / np.sqrt(m_vals)) @ m_vecs.T
        u = m_sqrt_inv @ o

        dns = self.d_ao + self.d_ao.transpose(1, 0)
        rhs_s = - s_inv @ dns @ s_inv
        dc_local = scipy.linalg.solve_sylvester(c_local, c_local, rhs_s)
        
        if (local):
            dno_ao = dns
        else:
            do_ao_raw = gto.intor_cross('int1e_ipovlp', self.mol, self.mol_old)[0, self.trunc_idx, :][:, self.trunc_idx]
            dno_ao = np.zeros((self.nao, self.nao))
            for iat in range(self.nat):
                p0, p1 = self.aoslices[iat][2:4]
                dno_ao[p0:p1, :] = -self.w[iat] * do_ao_raw[p0:p1, :]
        
        dno = dc_local @ o_ao @ self.mo_coeff_old + c_local @ dno_ao @ self.mo_coeff_old
        dnm = dno @ o.T + o @ dno.T
        rhs_m = - m_inv @ dnm @ m_inv
        dm_sqrt_inv = scipy.linalg.solve_sylvester(m_sqrt_inv, m_sqrt_inv, rhs_m)
        dnu = m_sqrt_inv @ dno + dm_sqrt_inv @ o
        self.grad_coeff = dc_local @ u + c_local @ dnu
    
    def align_mo_coeff_spacewise(self, mol_old, mo_old, block_orth=True, ncore=None, no_cas=None):
        
        mo = np.copy(self.mo_coeff)
        if (ncore==None):
            o0 = self.ncore
        else:
            o0 = ncore
        
        if (no_cas==None):
            o1 = o0 + self.no_cas
        else:
            o1 = o0 + no_cas

        nvirt = self.nmo - o1

        o_ao = gto.intor_cross('int1e_ovlp', self.mol, mol_old)[self.trunc_idx, :][:, self.trunc_idx]
        o = self.mo_coeff.T @ o_ao @ mo_old
        
        # Core
        o_core = reduce(np.dot, (mo.T, o_ao, mo_old[:, :o0]))
        if o0 > 0:
            m = o_core.T @ o_core
            m_vals, m_vecs = scipy.linalg.eigh(m)
            m_sqrt_inv = m_vecs @ np.diag(1.0 / np.sqrt(m_vals)) @ m_vecs.T
            u_core = o_core @ m_sqrt_inv
            self.mat_core = [m_vals, m_vecs]
        else:
            u_core = np.zeros((self.nao, 0))
            self.mat_core = [np.array([]), np.array([])]
        
        # Active
        tmp_act = reduce(np.dot, (mo.T, o_ao, mo_old[:, o0:o1]))
        o_act = np.copy(tmp_act)
        if block_orth and o0 > 0:
            o_act -= u_core @ u_core.T @ tmp_act
        
        m_act = o_act.T @ o_act
        m_vals_act, m_vecs_act = scipy.linalg.eigh(m_act)
        m_sqrt_inv_act = m_vecs_act @ np.diag(1.0 / np.sqrt(m_vals_act)) @ m_vecs_act.T
        u_act = o_act @ m_sqrt_inv_act
        self.mat_act = [m_vals_act, m_vecs_act]
        
        # Virtual
        tmp_virt = reduce(np.dot, (mo.T, o_ao, mo_old[:, o1:]))
        o_virt = np.copy(tmp_virt)
        if block_orth:
            if o0 > 0: o_virt -= u_core @ u_core.T @ tmp_virt
            if self.no_cas > 0: o_virt -= u_act @ u_act.T @ tmp_virt
        
        if nvirt > 0:
            m_virt = o_virt.T @ o_virt
            m_vals_virt, m_vecs_virt = scipy.linalg.eigh(m_virt)
            m_sqrt_inv_virt = m_vecs_virt @ np.diag(1.0 / np.sqrt(m_vals_virt)) @ m_vecs_virt.T
            u_virt = o_virt @ m_sqrt_inv_virt
            self.mat_virt = [m_vals_virt, m_vecs_virt]
        else:
            u_virt = np.zeros((self.nao, 0))
            self.mat_virt = [np.array([]), np.array([])]

        self.u = np.column_stack((u_core, u_act, u_virt))
        
        mo_new =  mo @ self.u
        self.mo_coeff = np.copy(mo_new)

    def get_V_csf(self):
        self.mc.mo_coeff = np.copy(self.mo_coeff)
        o0, o1 = self.ncore, self.ncore + self.no_cas
        h_ao = self.kin_ao + self.nuc_ao
        self.h_cas  = np.einsum('kp,kl,lq -> pq', self.mo_coeff[:, o0:o1], h_ao, self.mo_coeff[:, o0:o1], optimize=True)
        self.h_cas += 2.0 * np.einsum('kp, la, mq, na, klmn -> pq', self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.v_ao, optimize=True)
        self.h_cas -= np.einsum('kp, lq, ma, na, klmn -> pq', self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao, optimize=True)
        self.v_cas = np.einsum('kp, lq, mr, ns, klmn -> pqrs', self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.v_ao, optimize=True).transpose(0,2,1,3)
        self.V_csf = self.mc.fcisolver.pspace(self.h_cas, self.v_cas, self.no_cas, self.ne_cas, npsp=self.ncsf)[1]
        self.V_core = self.mol.enuc
        self.V_core += 2.0 * np.einsum('ka,kl,la -> ', self.mo_coeff[:, :o0], h_ao, self.mo_coeff[:, :o0], optimize=True)
        self.V_core += 2.0 * np.einsum('ka, lb, ma, nb, klmn -> ', self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao, optimize=True)
        self.V_core -= np.einsum('ka, la, mb, nb, klmn -> ', self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao, optimize=True)

    def get_dV_csf(self):
        self.mc.mo_coeff = np.copy(self.mo_coeff)
        dV_nuc_raw = np.copy(grad.rhf.grad_nuc(self.mol)[:, 0])
        self.dV_core = 0.0
        for iat in range(self.nat):
            self.dV_core += self.w[iat] * dV_nuc_raw[iat]
        
        o0, o1 = self.ncore, self.ncore + self.no_cas
        
        # strictly pre_bo_symm.py logic for d/dx0:
        dh_tmp  = np.einsum('kp,kl,lq -> pq', self.grad_coeff[:, o0:o1], (self.kin_ao+self.nuc_ao), self.mo_coeff[:, o0:o1], optimize=True)
        dh_tmp += np.einsum('kp, kl, lq -> pq', self.mo_coeff[:, o0:o1], self.dkin_ao+self.dnuc_ao, self.mo_coeff[:, o0:o1], optimize=True)
        dh_tmp += np.einsum('kp, kl, lq -> pq', self.mo_coeff[:, o0:o1], self.drinv_ao, self.mo_coeff[:, o0:o1], optimize=True)
        
        dh_tmp += 2.0 * np.einsum('kp, la, mq, na, klmn -> pq', self.grad_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.v_ao, optimize=True)
        dh_tmp += 2.0 * np.einsum('ka, lp, ma, nq, klmn -> pq', self.grad_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.v_ao, optimize=True)
        dh_tmp += 2.0 * np.einsum('kp, la, mq, na, klmn -> pq', self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.dv_ao, optimize=True)
        dh_tmp += 2.0 * np.einsum('ka, lp, ma, nq, klmn -> pq', self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.dv_ao, optimize=True)
        dh_tmp -= np.einsum('kp, lq, ma, na, klmn -> pq', self.grad_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao, optimize=True)
        dh_tmp -= np.einsum('ka, la, mp, nq, klmn -> pq', self.grad_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.v_ao, optimize=True)
        dh_tmp -= np.einsum('kp, lq, ma, na, klmn -> pq', self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.dv_ao, optimize=True)
        dh_tmp -= np.einsum('ka, la, mp, nq, klmn -> pq', self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.dv_ao, optimize=True)
        
        dh_mo = dh_tmp + dh_tmp.transpose(1, 0)
        dv_tmp  = np.einsum('kp, lq, mr, ns, klmn -> pqrs', self.grad_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.v_ao, optimize=True)
        dv_tmp += np.einsum('kp, lq, mr, ns, klmn -> pqrs', self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.dv_ao, optimize=True)
        dv_cas = dv_tmp + dv_tmp.transpose(1, 0, 3, 2) + dv_tmp.transpose(2, 3, 0, 1) + dv_tmp.transpose(3, 2, 1, 0)
        
        self.dV_csf = self.mc.fcisolver.pspace(dh_mo, dv_cas.transpose(0, 2, 1, 3), self.no_cas, self.ne_cas, npsp=self.ncsf)[1]

        self.dV_core += 4.0 * np.einsum('ka,kl,la -> ', self.grad_coeff[:, :o0], (self.kin_ao+self.nuc_ao), self.mo_coeff[:, :o0], optimize=True)
        self.dV_core += 4.0 * np.einsum('ka, kl, la -> ', self.mo_coeff[:, :o0], self.dkin_ao+self.dnuc_ao, self.mo_coeff[:, :o0], optimize=True)
        self.dV_core += 4.0 * np.einsum('ka, kl, la -> ', self.mo_coeff[:, :o0], self.drinv_ao, self.mo_coeff[:, :o0], optimize=True)
        self.dV_core += 8.0 * np.einsum('ka, lb, ma, nb, klmn -> ', self.grad_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao, optimize=True)
        self.dV_core += 8.0 * np.einsum('ka, lb, ma, nb, klmn -> ', self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.dv_ao, optimize=True)
        self.dV_core -= 4.0 * np.einsum('ka, la, mb, nb, klmn -> ', self.grad_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao, optimize=True)
        self.dV_core -= 4.0 * np.einsum('ka, la, mb, nb, klmn -> ', self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.dv_ao, optimize=True)

    def get_D_csf(self):
        d_mo = self.mo_coeff.T @ self.d_ao @ self.mo_coeff
        d_mo += self.mo_coeff.T @ self.s_ao @ self.grad_coeff
        self.D_csf = self.mc.fcisolver.pspace(d_mo[self.ncore:self.ncore+self.no_cas, self.ncore:self.ncore+self.no_cas], np.zeros_like(self.v_cas), self.no_cas, self.ne_cas, npsp=self.ncsf)[1]
        self.D_csf = 0.5 * (self.D_csf - self.D_csf.T)

    def get_td_mo_coeff(self, block_orth=True):
        # d/dt = v_q * d/dq
        dns_q = self.d_ao + self.d_ao.transpose(1, 0)
        #p0_0, p1_0 = self.aoslices[0][2:4]
        #d_full = self.mol.intor('int1e_ipovlp').transpose(0,2,1)[0, self.trunc_idx, :][:, self.trunc_idx]
        #dns_q = np.zeros((self.nao, self.nao))
        #dns_q[:, p0_0:p1_0] += d_full[:, p0_0:p1_0]
        #dns_q[p0_0:p1_0, :] += d_full[:, p0_0:p1_0].T
        
        ds_dt = self.v_q * dns_q
        td_ao = self.v_q * self.d_ao
        
        m_dot = self.mo_coeff.T @ ds_dt @ self.mo_coeff
        
        k_mat = np.zeros((self.nmo, self.nmo))
        if (block_orth):
            o0, o1 = self.ncore, self.ncore + self.no_cas
            k_mat[0:o0, 0:o0] = -0.5 * m_dot[0:o0, 0:o0]
            k_mat[o0:o1, o0:o1] = -0.5 * m_dot[o0:o1, o0:o1]
            k_mat[o1:, o1:] = -0.5 * m_dot[o1:, o1:]
            k_mat[0:o0, o0:o1] = -m_dot[0:o0, o0:o1]
            k_mat[0:o0, o1:] = -m_dot[0:o0, o1:]
            k_mat[o0:o1, o1:] = -m_dot[o0:o1, o1:]
        else:
            k_mat = -0.5 * m_dot
        
        return k_mat, td_ao

    def get_td_D_csf(self, block_orth=True, finite_difference=False, dt=None):
        zero_h2e = np.zeros(self.v_cas.shape)
        if (finite_difference):
            _, td_ao = self.get_td_mo_coeff(block_orth=block_orth)
            cdot = (self.mo_coeff - self.mo_coeff_old) / dt
        else:
            k_mat, td_ao = self.get_td_mo_coeff(block_orth=block_orth)
            cdot = self.mo_coeff @ k_mat
        td_mo = np.einsum('kp, kl, lq', self.mo_coeff, td_ao, self.mo_coeff, optimize=True)
        td_mo += np.einsum('kp, kl, lq', self.mo_coeff, self.s_ao, cdot, optimize=True)
        td_mo = 0.5 * (td_mo - td_mo.transpose(1,0))
        self.td_D_csf = self.mc.fcisolver.pspace(td_mo[self.ncore:self.ncore+self.no_cas, self.ncore:self.ncore+self.no_cas], zero_h2e, self.no_cas, self.ne_cas, npsp=self.ncsf)[1]
        self.td_D_csf = 0.5 * (self.td_D_csf - self.td_D_csf.transpose(1, 0))

    def calculate_force(self):
        self.force = -(self.csf_coeff.conj() @ self.dV_csf @ self.csf_coeff).real
        self.force += 2.0 * (self.csf_coeff.conj() @ self.D_csf @ self.V_csf @ self.csf_coeff).real
        self.force -= self.dV_core

    def get_dV_csf_2(self):
        """Hellmann-Feynman Hamiltonian gradient (skips Pulay terms)."""
        self.mc.mo_coeff = np.copy(self.mo_coeff)
        dV_nuc_raw = np.copy(grad.rhf.grad_nuc(self.mol)[:, 0])
        self.dV_core = 0.0
        for iat in range(self.nat):
            self.dV_core += self.w[iat] * dV_nuc_raw[iat]
        
        self.get_grad_ao()
        o0, o1 = self.ncore, self.ncore + self.no_cas
        
        # 1e dh_tmp: only includes operator derivative
        dh_tmp = np.einsum('kp, kl, lq -> pq', self.mo_coeff[:, o0:o1], self.drinv_ao, self.mo_coeff[:, o0:o1], optimize=True)
        dh_mo = dh_tmp + dh_tmp.transpose(1, 0)
        
        # 2e part is zero for HF force
        zero_v = np.zeros((self.no_cas, self.no_cas, self.no_cas, self.no_cas))
        
        self.dV_csf = self.mc.fcisolver.pspace(dh_mo, zero_v, self.no_cas, self.ne_cas, npsp=self.ncsf)[1]

        # V_core update: 2 * h_aa operator derivative
        self.dV_core += 4.0 * np.einsum('ka, kl, la -> ', self.mo_coeff[:, :o0], self.drinv_ao, self.mo_coeff[:, :o0], optimize=True)

    def calculate_force_2(self):
        """Hellmann-Feynman force calculation."""
        self.force = -(self.csf_coeff.conj() @ self.dV_csf @ self.csf_coeff).real
        self.force -= self.dV_core
