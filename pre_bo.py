from pyscf import gto, scf, mcscf, fci, nac, csf_fci, ao2mo, grad, symm
from pyscf.tools import mo_mapping, molden
from functools import reduce
import scipy
import numpy as np
import pickle
import math
import time

class pre_BO(object):
    
    def __init__(self, mol, ne_cas, no_cas, nmo=None, spin=0):
        
        # System
        self.mol = mol # molecule object of pyscf
        self.nat = mol.natm # number of atoms
        self.pos = mol.atom_coords(unit='Bohr') # nuclear position
        self.vel = np.zeros((self.nat, 3)) # nuclear velocity
        self.mass = gto.mole.atom_mass_list(mol, isotope_avg=True) * 1822.8884853324 # nuclear mass
        
        self.mol_old = mol.copy() # backup molecule
        
        self.nao = mol.nao # number of total spatial orbitals
        if (nmo == None):
            self.nmo = self.nao
        else:
            self.nmo = nmo
        self.ne_cas = ne_cas # number of active alpha and beta electrons. tuple (alpha, beta)
        self.no_cas = no_cas # number of active spatial orbitals
        self.spin = spin # S
        self.aoslices = self.mol.aoslice_by_atom() 
        
        # AO integrals
        self.s_ao = self.mol.intor('int1e_ovlp') # AO overlap (nao, nao)
        self.kin_ao = self.mol.intor('int1e_kin') # AO 1e kinetic energy integral
        self.nuc_ao = self.mol.intor('int1e_nuc') # AO 1e e-n couloumb integral
        self.v_ao = self.mol.intor('int2e', aosym='s1').transpose(0, 2, 1, 3) # AO 2e e-e coulomb integral
        
        # AO integral gradients
        self.d_ao = self.mol.intor('int1e_ipovlp').transpose(0,2,1) # (3, nao, nao)  <\chi_l | \nabla_e \chi_m> --> need to convert to d/dR_nuc which requires ao_slice for indices
        self.dkin_ao = np.zeros((3, self.nao, self.nao)) # gradient of AO 1e integral (3, nao, nao)
        self.dnuc_ao = np.zeros((3, self.nao, self.nao)) # gradient of AO 1e integral (3, nao, nao)
        self.dv_ao = np.zeros((3, self.nao, self.nao, self.nao, self.nao)) # gradient of AO 2e integral (3, nao, nao, nao, nao)
        
        # HF
        self.mf = scf.RHF(self.mol) # HF object
        #self.mf.run(conv_tol=1e-8)
        
        # MO integrals
        #self.mo_coeff[:, :] = self.mf.mo_coeff[:, :] # (nao, nmo) In fact, nao=nmo. Just to clarify the row and column.
        self.mo_coeff = np.zeros((self.nao, self.nmo)) # MO coefficient
        self.mo_coeff_old = np.zeros_like(self.mo_coeff) # backup MO coefficient
        self.grad_coeff = np.zeros((self.nat, 3, self.nao, self.nmo)) # nuclear gradient of MO coefficient (nat, 3, nao, nmo)  
        self.grad_coeff_old = np.zeros((self.nat, 3, self.nao, self.nmo)) # nuclear gradient of MO coefficient (nat, 3, nao, nmo)  
        self.h_mo = np.zeros((self.nmo, self.nmo)) # MO total 1e integral (kin + v_en) (nmo, nmo)
        self.v_mo = np.zeros((self.nmo, self.nmo, self.nmo, self.nmo)) # MO 2e integral  (nmo, nmo, nmo, nmo)
        self.u = np.zeros((self.nmo, self.nmo))
        
        # MO integral gradients 
        self.d_mo = np.zeros((self.nat, 3, self.nmo, self.nmo)) # MO derivative coupling (nat, 3, nmo, nmo)
        self.dh_mo = np.zeros((self.nat, 3, self.nmo, self.nmo)) # nuclear gradient of MO 1e integral (nat, 3, nmo, nmo)
        self.dv_mo = np.zeros((self.nat, 3, self.nmo, self.nmo, self.nmo, self.nmo)) # nuclear gradient of MO 2e integral (nat, 3, nmo, nmo, nmo, nmo)
        #self.g_mo = np.zeros((self.nat, self.nmo, self.nmo)) # MO scalar coupling (nat, nmo, nmo)
        #self.d_dot_d_mo = np.zeros((self.nat, self.nmo, self.nmo, self.nmo, self.nmo)) # MO dd term (nat, 3, nmo, nmo)
        
        # CASCI
        #ne_cas_sum = ne_cas[0] + ne_cas[1] # number of electrons
        self.transformer = csf_fci.CSFTransformer(self.no_cas, self.ne_cas[0], self.ne_cas[1], smult=2*spin+1)
        self.ncsf = self.transformer.ncsf

        self.mc = mcscf.CASCI(self.mol, no_cas, ne_cas) # CASCI object
        self.mc.fcisolver = csf_fci.csf_solver(self.mol, smult=2*spin+1) # fci solver with CSF
        #self.mc.mo_coeff[:, :] = self.mo_coeff[:, :]
        self.mc.fcisolver.nroots = self.ncsf # number of states
        self.ncore = self.mc.ncore # number of core spatial orbital
        self.nvirt = self.nmo - self.ncore - self.no_cas # number of core spatial orbital
        self.h_cas = np.zeros((self.no_cas, self.no_cas)) # 1e active
        self.v_cas = np.zeros((self.no_cas, self.no_cas)) # TODO 2e active, wrong dimension..!!!!!! It will be overwritten but.. need to be fixed. I think it was no_cas*(no_cas+1)/2 , no_cas*(no_cas+1)/2.. double check
        self.dh_cas = np.zeros((self.nat, 3, self.no_cas, self.no_cas)) # 1e active
        self.dv_cas = np.zeros((self.nat, 3, self.no_cas, self.no_cas, self.no_cas, self.no_cas)) # TODO 2e active, wrong dimension..!!!!!! It will be overwritten but.. need to be fixed. I think it was no_cas*(no_cas+1)/2 , no_cas*(no_cas+1)/2.. double check
        
        self.mat_core = [np.ones((self.ncore)), np.eye(self.ncore)]
        self.mat_act = [np.ones((self.no_cas)), np.eye(self.no_cas)]
        self.mat_virt = [np.ones((self.nvirt)), np.eye(self.nvirt)]
        
        # CSF properties
        self.V_core = 0. # core energy + Vnn
        self.V_csf = np.zeros((self.ncsf, self.ncsf)) # H_BO CSF matrix 
        self.dV_core = np.zeros((self.nat, 3))
        self.dV_csf = np.zeros((self.nat, 3, self.ncsf, self.ncsf)) # nuclear gradient of H_BO CSF matrix
        self.D_csf = np.zeros((self.nat, 3, self.ncsf, self.ncsf)) # derivative coupling between CSFs
        self.td_D_csf = np.zeros((self.ncsf, self.ncsf)) # derivative coupling between CSFs

        self.V_nuc = 0. # V_nn
        self.dV_nuc = np.zeros((self.nat, 3))
        
        # Dynamics variables
        self.csf_coeff = np.zeros((self.ncsf), dtype=np.complex128) # time-dependent CSF coefficients
        self.force = np.zeros((self.nat, 3))
        
        # Old state variables for electronic propagation
        self.vel_old = np.zeros((self.nat, 3))
        self.V_csf_old = np.zeros((self.ncsf, self.ncsf))
        self.D_csf_old = np.zeros((self.nat, 3, self.ncsf, self.ncsf))
        self.td_D_csf_old = np.zeros((self.ncsf, self.ncsf))

        # Continuous propagation variables
        self.dns = np.zeros((self.nat, 3, self.nao, self.nao))
        self.dns_old = np.zeros((self.nat, 3, self.nao, self.nao))
        #self.dS_dt = np.zeros((self.nao, self.nao))
        #self.M_nu = np.zeros((self.nat, 3, self.norb, self.norb))
        #self.M_dot = np.zeros((self.norb, self.norb))
        #self.X_nu = np.zeros((self.nat, 3, self.norb, self.norb))
        #self.K_mat = np.zeros((self.norb, self.norb))
        #self.dC_dt = np.zeros((self.norb, self.norb))
        #self.L_mat = np.zeros((self.norb, self.norb))
        #self.H_static_grad = np.zeros((self.nat, 3))


    def backup_elec_state(self):
        """
        Backup the current electronic state variables (velocity, Hamiltonian, non-adiabatic couplings)
        at the beginning of the nuclear step. This is required for interpolating the matrices
        during the sub-stepped electronic propagation.
        """
        self.vel_old = np.copy(self.vel)
        self.mo_coeff_old = np.copy(self.mo_coeff)
        self.V_csf_old = np.copy(self.V_csf)
        self.D_csf_old = np.copy(self.D_csf)
        self.td_D_csf_old = np.copy(self.td_D_csf)
        self.dns_old = np.copy(self.dns)

    def get_td_mo_coeff(self, block_orth=True):
        
        ds_ao_raw = -self.mol.intor('int1e_ipovlp')
        ds_dt = np.zeros((self.nao, self.nao))
        td_ao = np.zeros((self.nao, self.nao))
        
        for iat in range(self.nat):
            shl0, shl1, p0, p1 = self.aoslices[iat]
            dns = np.zeros((3, self.nao, self.nao))
            dns[:, p0:p1, :] = ds_ao_raw[:, p0:p1, :]
            td_ao += np.einsum('c, ckl -> kl', self.vel[iat], dns)
            dns[:, :, p0:p1] += ds_ao_raw[:, p0:p1, :].transpose(0, 2, 1)
            ds_dt += np.einsum('c, ckl -> kl', self.vel[iat], dns)
            
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
    

    def get_int_ao(self):

        self.s_ao = self.mol.intor('int1e_ovlp') # AO overlap (nao, nao)
        self.kin_ao = self.mol.intor('int1e_kin') # AO 1e kinetic energy integral
        self.nuc_ao = self.mol.intor('int1e_nuc') # AO 1e e-n couloumb integral
        self.v_ao = self.mol.intor('int2e', aosym='s1').transpose(0, 2, 1, 3) # AO 2e e-e coulomb integral
        self.d_ao = self.mol.intor('int1e_ipovlp').transpose(0,2,1) # (3, nao, nao)  <\chi_l | \nabla_e \chi_m> --> need to convert to d/dR_nuc which requires ao_slice for indices
    
    def get_grad_ao(self):
        
        self.dkin_ao = self.mol.intor('int1e_ipkin')
        self.dnuc_ao = self.mol.intor('int1e_ipnuc')
        self.dv_ao = self.mol.intor('int2e_ip1').transpose(0, 1, 3, 2, 4)
    
    def get_int_mo(self):
        
        #C = self.mo_coeff
        
        # Duplicative, since get_h1cas and get_h2cas calculate these.

        # 1e
        #self.h_mo = np.einsum('kl,kp,lq->pq', (self.nuc_ao + self.kin_ao), C, C)
        # 2e
        #self.v_mo = ao2mo.full(self.v_ee_ao, C)
        
        # d & g
        #d_ao_tmp = np.zeros((3, self.norb, self.norb))
        #g_ao_tmp = np.zeros((self.norb, self.norb))
        #dS_tmp = np.zeros((3, self.norb, self.norb)) # \nabla_\nu S 
        #d2S_tmp = np.zeros((self.norb, self.norb)) # \nabla_\nu^2 S

        # d
        for iat in range(self.nat):
            shl0, shl1, p0, p1 = self.aoslices[iat]
            self.d_mo[iat, :, :, :]  = np.einsum('kp,ckl,lq->cpq', self.mo_coeff[:, :], -self.d_ao[:, :, p0:p1], self.mo_coeff[p0:p1, :], optimize=True)
            self.d_mo[iat, :, :, :] += np.einsum('kp,kl,clq->cpq', self.mo_coeff[:, :], self.s_ao[:, :], self.grad_coeff[iat, :, :, :], optimize=True)
            #self.d_mo[iat, :, :, :] -= 0.5 * np.einsum('kp,alk,lq->apq', self.mo_coeff[p0:p1, :], -self.d_ao[:, :, p0:p1], self.mo_coeff[:, :], optimize=True)
        
        # g    
        #S_inv = scipy.linalg.inv(self.s_ao)
        #self.d_dot_d_mo[:, :, :, :, :] = np.einsum('acpr,acqs->aprqs', self.d_mo, self.d_mo) # pyscf follows chemist's notation
        #tmp_dense = np.zeros((self.norb, self.norb))
        #for iat in range(self.nat):
        #    start, stop = ao_slice[iat, 0], ao_slice[iat, 1]
        #    
        #    tmp_dense[:, :] = 0.
        #    tmp_dense[:, start:stop] = (-2.0 * self.kin_ao[:, start:stop]) # g_ao
        #    
        #    d2S_tmp[:, :] = 0.
        #    d2S_tmp[:, start:stop] = (-2.0 * self.kin_e_ao[:, start:stop]) 
        #    d2S_tmp[start:stop, :] += (-2.0 * self.kin_e_ao[:, start:stop].T) 
        #    d2S_tmp[start:stop, start:stop] = 0.0 # already zero but for stability
        #    tmp_dense += -0.5 * d2S_tmp # \nabla^2 S

        #    tmp_dense[:, :] -= 0.25 * np.einsum('ckl,lm,cnm->kn', -self.d_ao[:, :, start:stop], S_inv[start:stop, start:stop], -self.d_ao[:, :, start:stop])
        #    tmp_dense[:, start:stop] -= 0.25 * np.einsum('ckl,lm,cmn->kn', -self.d_ao[:, :, start:stop], S_inv[start:stop, :], -self.d_ao[:, :, start:stop])
        #    tmp_dense[start:stop, :] += 0.75 * np.einsum('clk,lm,cnm->kn', -self.d_ao[:, :, start:stop], S_inv[:, start:stop], -self.d_ao[:, :, start:stop])
        #    tmp_dense[start:stop, start:stop] += 0.75 * np.einsum('clk,lm,cmn->kn', -self.d_ao[:, :, start:stop], S_inv[:, :], -self.d_ao[:, :, start:stop])

        #    #g_mo_test[iat, :, :] = np.einsum('kp,kl,lq->pq', C, tmp_dense, C)
        #    self.g_mo[iat, :, :] = np.einsum('kp,kl,lq->pq', C, tmp_dense, C)

        # TEST Hermiticity of g_mo. Result: Hermitian
        #print(np.sum(self.g_mo - self.g_mo.transpose((0,2,1))))
        #print(np.sum(np.absolute(self.g_mo)))
    
    
    def get_csf(self):
        pass

    def get_V_csf(self):
        
        # TODO: make our own version of get_h1cas/_h2cas to use self.h_ao and self_v_ao, mc.get_h1cas/_h2cas calculates AO integrals again..
        self.mc.mo_coeff = np.copy(self.mo_coeff)
        self.mc.mol = self.mol.copy()
        self.V_nuc = self.mol.enuc
        self.h_cas, self.V_core = self.mc.get_h1cas(self.mo_coeff) # get active 1e integral and core energy + Vnn
        self.v_cas = self.mc.get_h2cas(self.mo_coeff) # get active 2e integral
        self.V_csf = self.mc.fcisolver.pspace(self.h_cas, self.v_cas, self.no_cas, self.ne_cas, npsp=self.ncsf)[1] # get CSF Hamiltonian matrix
        #print(self.h_cas)
        #print(self.V_core)
        #print(self.v_cas)
    
    def get_V_csf_2(self):
        
        self.mc.mo_coeff = np.copy(self.mo_coeff)
        self.mc.mol = self.mol.copy()
        
        o0 = self.ncore
        o1 = self.ncore + self.no_cas
        
        # 1e
        # h_{pq}
        self.h_cas  = np.einsum('kp,kl,lq -> pq', self.mo_coeff[:, o0:o1], (self.kin_ao+self.nuc_ao)[:, :], self.mo_coeff[:, o0:o1], optimize=True)
        # \sum_a 2(v_{paqa}-v_{pqaa})
        self.h_cas += 2.0 * np.einsum('kp, la, mq, na, klmn -> pq', self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
        self.h_cas -= np.einsum('kp, lq, ma, na, klmn -> pq', self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
        
        # 2e
        # v_{pqrs}
        self.v_cas = np.einsum('kp, lq, mr, ns, klmn -> pqrs', self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.v_ao[:, :, :, :], optimize=True)
        #self.v_cas = ao2mo.restore('4', self.v_cas.transpose(0,2,1,3), self.no_cas)
        self.v_cas = self.v_cas.transpose(0,2,1,3)

        self.V_csf = self.mc.fcisolver.pspace(self.h_cas, self.v_cas, self.no_cas, self.ne_cas, npsp=self.ncsf)[1] # get CSF Hamiltonian matrix
        
        # V_{const}
        # V_{nn}
        self.V_core = self.mc.energy_nuc()
        # \sum_a 2h_{aa}
        self.V_core += 2.0 * np.einsum('ka,kl,la -> ', self.mo_coeff[:, :o0], (self.kin_ao+self.nuc_ao)[:, :], self.mo_coeff[:, :o0], optimize=True)
        # \sum_{ab} 2(v_{abab}-v_{aabb})
        self.V_core += 2.0 * np.einsum('ka, lb, ma, nb, klmn -> ', self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
        self.V_core -= np.einsum('ka, la, mb, nb, klmn -> ', self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
    
    def get_D_csf(self):
        
        zero_h2e = np.zeros(self.v_cas.shape) # zero arrray to use `fcisolver.pspace`
        for iat in range(self.nat):
            for isp in range(3):
                self.D_csf[iat, isp, :, :] = self.mc.fcisolver.pspace(self.d_mo[iat, isp, self.ncore:self.ncore+self.no_cas, self.ncore:self.ncore+self.no_cas], zero_h2e, self.no_cas, self.ne_cas, npsp=self.ncsf)[1]
        self.D_csf = 0.5 * (self.D_csf - self.D_csf.transpose(0, 1, 3, 2))
    
    def get_td_D_csf(self, block_orth=True, finite_difference=False, dt=None):
        
        zero_h2e = np.zeros(self.v_cas.shape) # zero arrray to use `fcisolver.pspace`
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

    def get_G_csf(self):
        # TODO
        # 1. Full 1e RI dev: 0.5 * (<nab|nab> + \sum dd)^p_q
        # 2-1. Active space 1e RI dev: 0.5 * (<nab p|nab q> + \sum_r d^p_r d^r_q + 2.0*\sum_a d^p_ad^a_q)^p_q
        # 2-2. V_cons RI dev: \sum_a(<nab a|nab a> + \sum_b d^a_b d^b_a)
        pass

    def get_dV_csf(self):
        
        self.mc.mo_coeff = np.copy(self.mo_coeff)
        self.mc.mol = self.mol.copy()
        
        # nuclear gradients
        self.dV_nuc = grad.rhf.grad_nuc(self.mol)
        self.dV_core = np.copy(self.dV_nuc)

        # electronic gradients
        self.get_grad_ao()
        
        s_inv = scipy.linalg.inv(np.copy(self.s_ao))
        # TODO: active space 
        o0 = self.ncore
        o1 = self.ncore + self.no_cas
        for iat in range(self.nat):
            shl0, shl1, p0, p1 = self.aoslices[iat]
            #self.grad_coeff[iat, :, :, :] = - 0.5 * np.einsum('kl, clm, mp -> ckp', s_inv[:, :], -self.d_ao[:, :, p0:p1], self.mo_coeff[p0:p1, :], optimize=True)
            #self.grad_coeff[iat, :, :, :] += - 0.5 * np.einsum('kl, cml, mp -> ckp', s_inv[:, p0:p1], -self.d_ao[:, :, p0:p1], self.mo_coeff[:, :], optimize=True)
            
            # 1e
            # \nabla C terms: \alpha
            dh_tmp  = np.einsum('ckp,kl,lq -> cpq', self.grad_coeff[iat, :, :, o0:o1], (self.kin_ao+self.nuc_ao)[:, :], self.mo_coeff[:, o0:o1], optimize=True)
            # AO <\nabla | h | >  terms: \beta
            dh_tmp += np.einsum('kp, ckl, lq -> cpq', self.mo_coeff[p0:p1, o0:o1], -(self.dkin_ao+self.dnuc_ao)[:, p0:p1, :], self.mo_coeff[:, o0:o1], optimize=True)
            # AO < |\nabla V_nuc | > terms: \delta
            with self.mol.with_rinv_at_nucleus(iat):
                drinv_tmp = self.mol.intor('int1e_iprinv', comp=3)
                drinv_tmp *= -self.mol.atom_charge(iat)
            dh_tmp += np.einsum('kp, ckl, lq -> cpq', self.mo_coeff[:, o0:o1], drinv_tmp[:, :, :], self.mo_coeff[:, o0:o1], optimize=True)
            # effective 1e from 2e: \sum_a 2*((\kappa + \lambda)_{paqa}+(\kappa + \lambda)_{apaq}), due to symmetry summing transpose is same as multiplying 2 to (paqa+apaq)
            # core coulomb
            dh_tmp += 2.0 * np.einsum('ckp, la, mq, na, klmn -> cpq', self.grad_coeff[iat, :, :, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
            dh_tmp += 2.0 * np.einsum('cka, lp, ma, nq, klmn -> cpq', self.grad_coeff[iat, :, :, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.v_ao[:, :, :, :], optimize=True)
            dh_tmp += 2.0 * np.einsum('kp, la, mq, na, cklmn -> cpq', self.mo_coeff[p0:p1, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], -self.dv_ao[:, p0:p1, :, :, :], optimize=True)
            dh_tmp += 2.0 * np.einsum('ka, lp, ma, nq, cklmn -> cpq', self.mo_coeff[p0:p1, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], -self.dv_ao[:, p0:p1, :, :, :], optimize=True)
            # core exchange
            dh_tmp -= np.einsum('ckp, lq, ma, na, klmn -> cpq', self.grad_coeff[iat, :, :, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
            dh_tmp -= np.einsum('cka, la, mp, nq, klmn -> cpq', self.grad_coeff[iat, :, :, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.v_ao[:, :, :, :], optimize=True)
            dh_tmp -= np.einsum('kp, lq, ma, na, cklmn -> cpq', self.mo_coeff[p0:p1, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], -self.dv_ao[:, p0:p1, :, :, :], optimize=True)
            dh_tmp -= np.einsum('ka, la, mp, nq, cklmn -> cpq', self.mo_coeff[p0:p1, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], -self.dv_ao[:, p0:p1, :, :, :], optimize=True)
            ## active exchange
            #dh_tmp -= 0.5 * np.einsum('ckp, lq, mr, nr, klmn -> cpq', self.grad_coeff[iat, :, :, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.v_ao[:, :, :, :], optimize=True)
            #dh_tmp -= 0.5 * np.einsum('ckr, lr, mp, nq, klmn -> cpq', self.grad_coeff[iat, :, :, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.v_ao[:, :, :, :], optimize=True)
            #dh_tmp -= 0.5 * np.einsum('kp, lq, mr, nr, cklmn -> cpq', self.mo_coeff[p0:p1, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], -self.dv_ao[:, p0:p1, :, :, :], optimize=True)
            #dh_tmp -= 0.5 * np.einsum('kr, lr, mp, nq, cklmn -> cpq', self.mo_coeff[p0:p1, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], -self.dv_ao[:, p0:p1, :, :, :], optimize=True)
            # total
            dh_mo = dh_tmp + dh_tmp.transpose(0, 2, 1)
            self.dh_cas[iat] = np.copy(dh_mo)

            # 2e
            # \nabla C terms: \kappa_{pqrs}
            dv_tmp  = np.einsum('ckp, lq, mr, ns, klmn -> cpqrs', self.grad_coeff[iat, :, :, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.v_ao[:, :, :, :], optimize=True)
            # AO <\nabla, | , > terms: \lambda_{pqrs}
            dv_tmp += np.einsum('kp, lq, mr, ns, cklmn -> cpqrs', self.mo_coeff[p0:p1, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], -self.dv_ao[:, p0:p1, :, :, :], optimize=True)
            # Total
            dv_mo = dv_tmp + dv_tmp.transpose(0, 2, 1, 4, 3) + dv_tmp.transpose(0, 3, 4, 1, 2) + dv_tmp.transpose(0, 4, 3, 2, 1)
            self.dv_cas[iat] = np.copy(dv_mo)
            
            # effective const from 1e: 2*(\alpha_{aa} + \beta_{aa} + \delta_{aa}), due to symmetry summing transpose is same as multiplying 2
            self.dV_core[iat, :] += 4.0 * np.einsum('cka,kl,la -> c', self.grad_coeff[iat, :, :, :o0], (self.kin_ao+self.nuc_ao)[:, :], self.mo_coeff[:, :o0], optimize=True)
            self.dV_core[iat, :] += 4.0 * np.einsum('ka, ckl, la -> c', self.mo_coeff[p0:p1, :o0], -(self.dkin_ao+self.dnuc_ao)[:, p0:p1, :], self.mo_coeff[:, :o0], optimize=True)
            self.dV_core[iat, :] += 4.0 * np.einsum('ka, ckl, la -> c', self.mo_coeff[:, :o0], drinv_tmp[:, :, :], self.mo_coeff[:, :o0], optimize=True)

            # effective const from 2e: \kappa + \lambda, due to symmetry, summing transpose is same as multiplying 4
            # coulomb: 2 * (4*(\kappa+\lambda)_{abab})
            self.dV_core[iat, :] += 8.0 * np.einsum('cka, lb, ma, nb, klmn -> c', self.grad_coeff[iat, :, :, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
            self.dV_core[iat, :] += 8.0 * np.einsum('ka, lb, ma, nb, cklmn -> c', self.mo_coeff[p0:p1, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], -self.dv_ao[:, p0:p1, :, :, :], optimize=True)
            # exchange - (4*(\kappa+\lambda)_{aabb})
            self.dV_core[iat, :] -= 4.0 * np.einsum('cka, la, mb, nb, klmn -> c', self.grad_coeff[iat, :, :, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
            self.dV_core[iat, :] -= 4.0 * np.einsum('ka, la, mb, nb, cklmn -> c', self.mo_coeff[p0:p1, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], -self.dv_ao[:, p0:p1, :, :, :], optimize=True)
            
            for isp in range(3):
                self.dV_csf[iat, isp, :, :] = self.mc.fcisolver.pspace(dh_mo[isp, :, :], dv_mo.transpose(0,1,3,2,4)[isp, :, :, :, :], self.no_cas, self.ne_cas, npsp=self.ncsf)[1] # get CSF Hamiltonian matrix
                
                #dv_mo_tmp = ao2mo.restore('4', dv_mo[isp].transpose(0, 2, 1, 3), self.no_cas)
                #self.dV_csf[iat, isp, :, :] = self.mc.fcisolver.pspace(dh_mo[isp, :, :], dv_mo_tmp, self.no_cas, self.ne_cas)[1] # get CSF Hamiltonian matrix
                
    
    def get_dV_csf_2(self):
        
        self.mc.mo_coeff = np.copy(self.mo_coeff)
        self.mc.mol = self.mol.copy()

        # nuclear gradients
        self.dV_core = grad.rhf.grad_nuc(self.mol)

        # electronic gradients
        self.get_grad_ao()
        
        # TODO: active space 
        o0 = self.ncore
        o1 = self.ncore + self.no_cas
        zero_v = np.zeros((self.no_cas, self.no_cas, self.no_cas, self.no_cas))
        for iat in range(self.nat):
            shl0, shl1, p0, p1 = self.aoslices[iat]
            with self.mol.with_rinv_at_nucleus(iat):
                drinv_tmp = self.mol.intor('int1e_iprinv', comp=3)
                drinv_tmp *= -self.mol.atom_charge(iat)
            dh_tmp = np.einsum('kp, ckl, lq -> cpq', self.mo_coeff[:, o0:o1], drinv_tmp[:, :, :], self.mo_coeff[:, o0:o1], optimize=True)
            dh_mo = dh_tmp + dh_tmp.transpose(0, 2, 1)
            self.dV_core[iat, :] += 4.0 * np.einsum('ka, ckl, la -> c', self.mo_coeff[:, :o0], drinv_tmp[:, :, :], self.mo_coeff[:, :o0], optimize=True)
            
            for isp in range(3):
                self.dV_csf[iat, isp, :, :] = self.mc.fcisolver.pspace(dh_mo[isp, :, :], zero_v, self.no_cas, self.ne_cas, npsp=self.ncsf)[1] # get CSF Hamiltonian matrix
        
    def calculate_force(self):
        self.force = -np.einsum('I, AcIJ, J -> Ac', self.csf_coeff.conj(), self.dV_csf, self.csf_coeff, optimize=True).real
        self.force += 2.0 * np.einsum('I, AcKI, KJ, J -> Ac', self.csf_coeff.conj(), self.D_csf, self.V_csf, self.csf_coeff, optimize=True).real
        self.force -= self.dV_core
    
    def calculate_force_2(self):
        self.force = -np.einsum('I, AcIJ, J -> Ac', self.csf_coeff.conj(), self.dV_csf, self.csf_coeff).real
        self.force -= self.dV_core
    

    def propagate_elec_expm(self, dt, l_tdnac=False):
        """
        Exact Unitary propagation of CSF coefficients (non-interpolated).
        Evaluates C(t + dt) = exp(M * dt) * C(t).
        """
        # Construct the effective coupling matrix M at the current geometry
        M = -1j * self.V_csf
        if (l_tdnac):
            M -= 1.0 * self.td_D_csf
        else:
            M -= np.einsum('Ac, AcIJ -> IJ', self.vel, self.D_csf, optimize=True)
        
        # Compute exact unitary propagator
        U = scipy.linalg.expm(M * dt)
        
        # Update the electronic state
        self.csf_coeff = U @ self.csf_coeff

    def propagate_elec(self, dt, nsteps=100, l_tdnac=False):
        """
        Integrates the electronic TDSE over a single nuclear step `dt` 
        by taking `nsteps` interpolated sub-steps using RK4.
        Requires self.backup_elec_state() to be called at the previous nuclear step.
        """
        # Construct M_old and M_new matrices (atomic units: hbar = 1)
        if (l_tdnac):
            M_old = -1j * self.V_csf_old - self.td_D_csf_old
            M_new = -1j * self.V_csf - self.td_D_csf
        else:
            # M = -i * H_BO - sum_nu (R_dot_nu * D_nu)
            M_old = -1j * self.V_csf_old - np.einsum('Ac, AcIJ -> IJ', self.vel_old, self.D_csf_old, optimize=True)
            M_new = -1j * self.V_csf - np.einsum('Ac, AcIJ -> IJ', self.vel, self.D_csf, optimize=True)
        
        C = self.csf_coeff.copy()
        dtau = dt / nsteps
        
        for i in range(nsteps):
            t1 = i / nsteps
            t2 = (i + 0.5) / nsteps
            t3 = (i + 1.0) / nsteps
            
            M1 = M_old * (1.0 - t1) + M_new * t1
            M2 = M_old * (1.0 - t2) + M_new * t2
            M3 = M_old * (1.0 - t3) + M_new * t3
            
            k1 = M1 @ C
            k2 = M2 @ (C + 0.5 * dtau * k1)
            k3 = M2 @ (C + 0.5 * dtau * k2)
            k4 = M3 @ (C + dtau * k3)
            
            C += (dtau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            
        # Enforce strict norm conservation (optional but recommended for long MD)
        #norm = np.linalg.norm(C)
        self.csf_coeff = np.copy(C) #/ norm

    def align_mo_coeff_simple(self, mol_old, mo_old):
        
        mo = np.copy(self.mo_coeff)
        #idx, s = mo_mapping.mo_map(mol_old, mo_old, self.mol, self.mo_coeff)
        s = gto.intor_cross('int1e_ovlp', mol_old, self.mol)
        s = reduce(np.dot, (mo_old.T, s, mo))
        
        # detect swap
        col_ind = np.argmax(np.abs(s), axis=1) # i-th element is the index of new MO corresponding to the i-th old MO
        row_ind = np.arange(self.nao)
       
        # detect sign flip
        phases = np.sign(s[row_ind, col_ind])
        phases[phases == 0] = 1.0  # Prevent zeroing out orbitals if exact 0 overlap occurs, although it is highly unlikely.

        self.mo_coeff = mo[:, col_ind] * phases # bring new MO to i-th column (bring new index --> old index)
    
    #####################################################################################################
    # MO alignment and corresponding gradient without separation of space, discrete
    #####################################################################################################
    def align_mo_coeff(self, mol_old, mo_old, diag_block_0=False):
        
        mo = np.copy(self.mo_coeff)
        if self.mol.symmetry:
            mo = symm.symmetrize_orb(self.mol, mo, s=self.s_ao)
        
        o_ao = gto.intor_cross('int1e_ovlp', self.mol, mol_old)
        if (diag_block_0):
            for iat in range(self.nat):
                p0, p1 = self.aoslices[iat][2:4]
                for io in range(p0, p1):
                    for jo in range(io+1, p1):
                        o_ao[io, jo] = 0.0
                        o_ao[jo, io] = 0.0
        
        o = reduce(np.dot, (mo.T, o_ao, mo_old))
        m = o @ o.T
        m_vals, m_vecs = scipy.linalg.eigh(m)
        m_sqrt_inv = m_vecs @ np.diag(1.0 / np.sqrt(m_vals)) @ m_vecs.T
        u = m_sqrt_inv @ o
        mo_new =  mo @ u
        self.mo_coeff = np.copy(mo_new)
        
    def get_grad_coeff(self, diag_block_0=False): # For FCI..
        """
        Calculate analytical nuclear gradient of Tracked Lowdin Molecular Orbitals.
        Implements equations 15-21 from Pre_BO_direct_dynamics.pdf.
        """
        # 1. AO Overlap and its nuclear gradient
        s = np.copy(self.s_ao)

        # ds_ao_raw[c, mu, nu] = <nabla_c mu | nu>
        # Nuclear gradient d/dR_A = -nabla_e for basis functions on A.
        ds_ao_raw = -self.mol.intor('int1e_ipovlp')
        
        # 2. Local Lowdin Orbitals: C_local = S^-1/2 
        s_vals, s_vecs = scipy.linalg.eigh(s)
        sq_s = np.sqrt(s_vals)
        s_sqrt = s_vecs @ np.diag(sq_s) @ s_vecs.T
        s_inv = s_vecs @ np.diag(1.0 / s_vals) @ s_vecs.T
        c_local = s_vecs @ np.diag(1.0 / sq_s) @ s_vecs.T
        if self.mol.symmetry:
            c_local = symm.symmetrize_orb(self.mol, c_local, s=s)
            #irrep_ids = symm.label_orb_symm(self.mol, self.mol.irrep_id, self.mol.symm_orb, c_local)
            #c_local = c_local[:, irrep_ids==0]
            #print(c_local.shape)
            
        
        # 3. Inter-geometry overlap matrix: O = C_local^T * O_AO * C
        # O_AO = <chi(R(t)) | chi(R(t-dt))>
        o_ao = gto.intor_cross('int1e_ovlp', self.mol, self.mol_old)
        if (diag_block_0):
            for iat in range(self.nat):
                p0, p1 = self.aoslices[iat][2:4]
                for io in range(p0, p1):
                    for jo in range(io+1, p1):
                        o_ao[io, jo] = 0.0
                        o_ao[jo, io] = 0.0
        # C_local is symmetric, so C_local^T = C_local
        o = reduce(np.dot, (c_local.T, o_ao, self.mo_coeff_old))
        
        # 4. M = O * O^T
        m = o @ o.T
        m_vals, m_vecs = scipy.linalg.eigh(m)
        sq_m = np.sqrt(m_vals)
        m_inv = m_vecs @ np.diag(1.0 / m_vals) @ m_vecs.T
        m_sqrt = m_vecs @ np.diag(sq_m) @ m_vecs.T
        m_sqrt_inv = m_vecs @ np.diag(1.0 / sq_m) @ m_vecs.T
        
        # 5. Tracking Matrix U = M^-1/2 * O 
        u = m_sqrt_inv @ o

        # 6. Precompute Gradient of O_AO:<nabla_nu chi(R(t)) | chi(R(t-dt))> & <chi(R(t)) | \nabla chi(R(t-dt))>
        do_ao_raw_left = -gto.intor_cross('int1e_ipovlp', self.mol, self.mol_old)
        #do_ao_raw_right = -gto.intor_cross('int1e_ipovlp', self.mol_old, self.mol).transpose(0, 2, 1)
            
        orthnorm_test = np.zeros((self.nat, 3))

        for iat in range(self.nat):
            shl0, shl1, p0, p1 = self.aoslices[iat]
            
            # nabla S for this atom
            dns = np.zeros((3, self.nao, self.nao))
            dns[:, p0:p1, :] = ds_ao_raw[:, p0:p1, :]
            dns[:, :, p0:p1] += ds_ao_raw[:, p0:p1, :].transpose(0, 2, 1)
            #print("dns")
            #print(dns[:, p0:p1, p0:p1])
            
            # nabla O_AO for this atom
            dno_ao = np.zeros((3, self.nao, self.nao))
            dno_ao[:, p0:p1, :] = do_ao_raw_left[:, p0:p1, :] 
            if (diag_block_0):
                for io in range(p0, p1):
                    for jo in range(io+1, p1):
                        dno_ao[:, io, jo] = 0.0
                        dno_ao[:, jo, io] = 0.0
            #dno_ao[:, :, p0:p1] += do_ao_raw_right[:, :, p0:p1]
            
            for idim in range(3):
                # Eq 17 context: Sylvester for dC_local = d(S^-1/2)
                # C_local * dC_local + dC_local * C_local = dS^-1 = - S^-1 * dS * S^-1
                #rhs_s = - reduce(np.dot, (s_inv, dns[idim], s_inv))
                rhs_s = - s_inv @ dns[idim] @ s_inv
                dc_local = scipy.linalg.solve_sylvester(c_local, c_local, rhs_s)
                #print("dc_local:", dc_local)
                
                # Eq 21: Gradient of inter-geometry overlap O
                # dO = dC_local * O_AO * C' + C_local * dO_AO * C'
                dno = dc_local @ o_ao @ self.mo_coeff_old + c_local @ dno_ao[idim] @ self.mo_coeff_old #+ c_local @ o_ao @ self.grad_coeff_old[iat, idim]
                #print("dno:", dno)
                
                # Eq 20: Gradient of metric M = O * O^T
                # dM = dO * O^T + O * dO^T
                dnm = dno @ o.T + o @ dno.T
                #print("dnm:", dnm)
                
                # Eq 19: Sylvester for dM^-1/2
                # M^-1/2 * dM_sqrt_inv + dM_sqrt_inv * M^-1/2 = dM^-1 = - M^-1 * dM * M^-1
                #rhs_m = - reduce(np.dot, (m_inv, dnm, m_inv))
                rhs_m = - m_inv @ dnm @ m_inv
                dm_sqrt_inv = scipy.linalg.solve_sylvester(m_sqrt_inv, m_sqrt_inv, rhs_m)
                #print("dm_sqrt_inv:", dm_sqrt_inv)
                
                # Eq 18: Gradient of tracking matrix U
                # dU = M^-1/2 * dO + dM_sqrt_inv * O
                dnu = m_sqrt_inv @ dno + dm_sqrt_inv @ o
                #print("dnu", dnu)
                
                # Eq 15: Final MO gradient
                # dC = dC_local * U + C_local * dU
                self.grad_coeff[iat, idim, :, :] = dc_local @ u + c_local @ dnu
                #print("grad_coeff", self.grad_coeff[iat, idim, :, :])
        
            #orthnorm_test[iat] = np.einsum('ckp,kl,lq->c', self.grad_coeff[iat, :, :, :], s[:, :], self.mo_coeff[:, :])
            #orthnorm_test[iat] += np.einsum('kp,kl,clq->c', self.mo_coeff[:, :], s[:, :], self.grad_coeff[iat, :, :, :])
            #orthnorm_test[iat] += np.einsum('kp,ckl,lq->c', self.mo_coeff[:, :], dns[:, :, :], self.mo_coeff[:, :])
            #print(orthnorm_test[iat])
            #print("-----------------------------------------------------------------------")
            
        #orthnorm_com_test = np.einsum('A, Ac->c', self.mass[:], orthnorm_test[:, :]) / np.sum(self.mass[:])
        #print("com:", orthnorm_com_test)
        #orthnorm_sum_test = np.sum(orthnorm_test[:, :], axis=0)
        #print("sum:", orthnorm_sum_test)


    #####################################################################################################
    # MO propagation and corresponding gradient with CAS approximation, continuous
    #####################################################################################################
    def propagate_mo_coeff_expm(self, dt, block_orth=True):
        """
        Propagate MO coefficients forward using the exact Unitary Propagator (non-interpolated).
        C(t+dt) = C(t) * exp(K * dt).
        """
        k_mat, _ = self.get_td_mo_coeff(block_orth=block_orth)

        # Unitary propagation via matrix exponential
        u_mo = scipy.linalg.expm(k_mat * dt)
        self.mo_coeff = self.mo_coeff @ u_mo
        
        # Re-orthogonalize to prevent numerical drift
        s = self.mol.intor('int1e_ovlp')
        overlap_mo = self.mo_coeff.T @ s @ self.mo_coeff
        val, vec = scipy.linalg.eigh(overlap_mo)
        self.mo_coeff = self.mo_coeff @ (vec @ np.diag(1.0/np.sqrt(val)) @ vec.T)
        if self.mol.symmetry:
            self.mo_coeff = symm.symmetrize_orb(self.mol, self.mo_coeff, s=s)

    def propagate_mo_coeff(self, dt, nsteps=100, block_orth=True):
        """
        Propagate MO coefficients forward using an interpolated RK4 integrator.
        Ensures the transport matrix K is consistent with the changing nuclear trajectory.
        """
        o0, o1 = self.ncore, self.ncore + self.no_cas
        
        def get_K(C, vel, dns):
            # Construct K = build_triangular(C^T * (sum vel * dns) * C)
            ds_dt = np.einsum('Ac, Acmn -> mn', vel, dns)
            m_dot = C.T @ ds_dt @ C
            k = np.zeros((self.nmo, self.nmo))
            if (block_orth):
                k[0:o0, 0:o0] = -0.5 * m_dot[0:o0, 0:o0]
                k[o0:o1, o0:o1] = -0.5 * m_dot[o0:o1, o0:o1]
                k[o1:, o1:] = -0.5 * m_dot[o1:, o1:]
                k[0:o0, o0:o1] = -m_dot[0:o0, o0:o1]
                k[0:o0, o1:] = -m_dot[0:o0, o1:]
                k[o0:o1, o1:] = -m_dot[o0:o1, o1:]
            else:
                k = -0.5 * m_dot
            return k

        C = self.mo_coeff.copy()
        dtau = dt / nsteps
        
        for i in range(nsteps):
            # Linear interpolation of nuclear parameters
            t1 = i / nsteps
            t2 = (i + 0.5) / nsteps
            t3 = (i + 1.0) / nsteps
            
            v1, d1 = self.vel_old * (1-t1) + self.vel * t1, self.dns_old * (1-t1) + self.dns * t1
            v2, d2 = self.vel_old * (1-t2) + self.vel * t2, self.dns_old * (1-t2) + self.dns * t2
            v3, d3 = self.vel_old * (1-t3) + self.vel * t3, self.dns_old * (1-t3) + self.dns * t3
            
            k1 = get_K(C, v1, d1)
            deriv1 = C @ k1
            
            k2 = get_K(C + 0.5 * dtau * deriv1, v2, d2)
            deriv2 = (C + 0.5 * dtau * deriv1) @ k2
            
            k3 = get_K(C + 0.5 * dtau * deriv2, v2, d2)
            deriv3 = (C + 0.5 * dtau * deriv2) @ k3
            
            k4 = get_K(C + dtau * deriv3, v3, d3)
            deriv4 = (C + dtau * deriv3) @ k4
            
            C += (dtau / 6.0) * (deriv1 + 2.0 * deriv2 + 2.0 * deriv3 + deriv4)
            
        self.mo_coeff = C
        # Re-orthogonalize to prevent numerical drift
        s = self.mol.intor('int1e_ovlp')
        overlap_mo = self.mo_coeff.T @ s @ self.mo_coeff
        val, vec = scipy.linalg.eigh(overlap_mo)
        self.mo_coeff = self.mo_coeff @ (vec @ np.diag(1.0/np.sqrt(val)) @ vec.T)
        if self.mol.symmetry:
            self.mo_coeff = symm.symmetrize_orb(self.mol, self.mo_coeff, s=s)
    
    def get_grad_coeff_continuous(self, block_orth=True):
        """
        Calculate MO coefficient gradients using the continuous connection matrix X^nu.
        X^nu is constructed to satisfy orthonormality and subspace preservation.
        """
        if self.mol.symmetry:
            self.mo_coeff = symm.symmetrize_orb(self.mol, self.mo_coeff, s=self.s_ao)
        self.get_grad_ao()
        ds_ao_raw = -self.mol.intor('int1e_ipovlp')
        
        o0, o1 = self.ncore, self.ncore + self.no_cas
        
        self.dns.fill(0.0)
        for iat in range(self.nat):
            shl0, shl1, p0, p1 = self.aoslices[iat]
            # 1. Calculate nabla S for this atom
            self.dns[iat, :, p0:p1, :] = ds_ao_raw[:, p0:p1, :]
            self.dns[iat, :, :, p0:p1] += ds_ao_raw[:, p0:p1, :].transpose(0, 2, 1)
            
            for idim in range(3):
                # 2. Project into MO basis: M^nu = C^T (nabla_nu S) C
                m_nu = self.mo_coeff.T @ self.dns[iat, idim] @ self.mo_coeff
                
                # 3. Build upper-triangular X^nu
                x_nu = np.zeros((self.nmo, self.nmo))
                if (block_orth):
                    # Diagonal blocks: -0.5 * M
                    x_nu[0:o0, 0:o0] = -0.5 * m_nu[0:o0, 0:o0]
                    x_nu[o0:o1, o0:o1] = -0.5 * m_nu[o0:o1, o0:o1]
                    x_nu[o1:, o1:] = -0.5 * m_nu[o1:, o1:]
                    # Off-diagonal blocks: -M (Core -> Active -> Virtual)
                    x_nu[0:o0, o0:o1] = -m_nu[0:o0, o0:o1]
                    x_nu[0:o0, o1:] = -m_nu[0:o0, o1:]
                    x_nu[o0:o1, o1:] = -m_nu[o0:o1, o1:]
                else:
                    x_nu = -0.5 * m_nu
                
                # 4. grad_coeff = C * X^nu
                self.grad_coeff[iat, idim] = self.mo_coeff @ x_nu
    
    
    #####################################################################################################
    # Spacewise alignment and corresponding gradient with CAS approximation, discrete
    #####################################################################################################
    def align_mo_coeff_spacewise(self, mol_old, mo_old, block_orth=True):
        
        mo = np.copy(self.mo_coeff)
        if self.mol.symmetry:
            mo = symm.symmetrize_orb(self.mol, mo, s=self.s_ao)
        o0, o1 = self.ncore, self.ncore+self.no_cas
        o_ao = gto.intor_cross('int1e_ovlp', self.mol, mol_old)
        
        # There were bugs in calculating o_core, tmp_act, tmp_virt !!!!!!!!!!!!!!
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
        
        if self.nvirt > 0:
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

    def get_grad_coeff_spacewise(self, block_orth=True):
        """
        Calculate analytical nuclear gradient of Tracked Lowdin Molecular Orbitals.
        Implements equations 15-21 from Pre_BO_direct_dynamics.pdf.
        """
        o0, o1 = self.ncore, self.ncore+self.no_cas

        # 1. AO Overlap and its nuclear gradient
        s = np.copy(self.s_ao)
        # ds_ao_raw[c, mu, nu] = <nabla_c mu | nu>
        # Nuclear gradient d/dR_A = -nabla_e for basis functions on A.
        ds_ao_raw = -self.mol.intor('int1e_ipovlp')
        
        # 2. Local Lowdin Orbitals: C_local = S^-1/2 
        s_vals, s_vecs = scipy.linalg.eigh(s)
        sq_s = np.sqrt(s_vals)
        s_sqrt = s_vecs @ np.diag(sq_s) @ s_vecs.T
        s_inv = s_vecs @ np.diag(1.0 / s_vals) @ s_vecs.T
        c_local = s_vecs @ np.diag(1.0 / sq_s) @ s_vecs.T
        if self.mol.symmetry:
            c_local = symm.symmetrize_orb(self.mol, c_local, s=s)
            self.mc
        
        # 3. Inter-geometry overlap matrix: O = C_local^T * O_AO * C
        # O_AO = <chi(R(t)) | chi(R(t-dt))>
        o_ao = gto.intor_cross('int1e_ovlp', self.mol, self.mol_old)
        # C_local is symmetric, so C_local^T = C_local
        o_mo = reduce(np.dot, (c_local.T, o_ao, self.mo_coeff_old))
        
        # 4. M = O * O^T
        if o0 > 0:
            m_vals, m_vecs = self.mat_core[0], self.mat_core[1]
            sq_m = np.sqrt(m_vals)
            m_c_inv = m_vecs @ np.diag(1.0 / m_vals) @ m_vecs.T
            m_c_sqrt_inv = m_vecs @ np.diag(1.0 / sq_m) @ m_vecs.T
        
        m_vals, m_vecs = self.mat_act[0], self.mat_act[1]
        sq_m = np.sqrt(m_vals)
        m_a_inv = m_vecs @ np.diag(1.0 / m_vals) @ m_vecs.T
        m_a_sqrt_inv = m_vecs @ np.diag(1.0 / sq_m) @ m_vecs.T
        
        if self.nvirt > 0:
            m_vals, m_vecs = self.mat_virt[0], self.mat_virt[1]
            sq_m = np.sqrt(m_vals)
            m_v_inv = m_vecs @ np.diag(1.0 / m_vals) @ m_vecs.T
            m_v_sqrt_inv = m_vecs @ np.diag(1.0 / sq_m) @ m_vecs.T
        
        u = np.copy(self.u)
        
        # 6. Precompute Gradient of O_AO:<nabla_nu chi(R(t)) | chi(R(t-dt))> & <chi(R(t)) | \nabla chi(R(t-dt))>
        do_ao_raw_left = -gto.intor_cross('int1e_ipovlp', self.mol, self.mol_old)
        #do_ao_raw_right = -gto.intor_cross('int1e_ipovlp', self.mol_old, self.mol).transpose(0, 2, 1)

        for iat in range(self.nat):
            shl0, shl1, p0, p1 = self.aoslices[iat]
            
            # nabla S for this atom
            ds = np.zeros((3, self.nao, self.nao))
            ds[:, p0:p1, :] = ds_ao_raw[:, p0:p1, :]
            ds[:, :, p0:p1] += ds_ao_raw[:, p0:p1, :].transpose(0, 2, 1)
            
            # nabla O_AO for this atom
            do_ao = np.zeros((3, self.nao, self.nao))
            do_ao[:, p0:p1, :] = do_ao_raw_left[:, p0:p1, :] 
            
            for idim in range(3):
                # Gradient of local MO coefficient, S^{-1/2}
                rhs_s = - s_inv @ ds[idim] @ s_inv
                dc_local = scipy.linalg.solve_sylvester(c_local, c_local, rhs_s)
                # Gradient of local time overlap
                do_mo = dc_local @ o_ao @ self.mo_coeff_old + c_local @ do_ao[idim] @ self.mo_coeff_old
                
                # Core
                if o0 > 0:
                    dm_core = do_mo[:, :o0].T @ o_mo[:, :o0] + o_mo[:, :o0].T @ do_mo[:, :o0]
                    rhs_m = - m_c_inv @ dm_core @ m_c_inv
                    dm_sqrt_inv = scipy.linalg.solve_sylvester(m_c_sqrt_inv, m_c_sqrt_inv, rhs_m)
                    du_core = do_mo[:, :o0] @ m_c_sqrt_inv + o_mo[:, :o0] @ dm_sqrt_inv
                    self.grad_coeff[iat, idim, :, :o0] = dc_local @ u[:, :o0] + c_local @ du_core
                else:
                    du_core = np.zeros((self.nao, 0))

                # Active
                o_act = o_mo[:, o0:o1] 
                do_act = np.copy(do_mo[:, o0:o1])
                
                if(block_orth and o0 > 0):
                    o_act -= u[:, :o0] @ u[:, :o0].T @ o_mo[:, o0:o1]
                    do_act -= du_core @ u[:, :o0].T @ o_mo[:, o0:o1]
                    do_act -= u[:, :o0] @ du_core.T @ o_mo[:, o0:o1]
                    do_act -= u[:, :o0] @ u[:, :o0].T @ do_mo[:, o0:o1]

                dm_act = do_act.T @ o_act + o_act.T @ do_act
                rhs_m = - m_a_inv @ dm_act @ m_a_inv
                dm_sqrt_inv = scipy.linalg.solve_sylvester(m_a_sqrt_inv, m_a_sqrt_inv, rhs_m)
                du_act = do_act @ m_a_sqrt_inv + o_act @ dm_sqrt_inv
                self.grad_coeff[iat, idim, :, o0:o1] = dc_local @ u[:, o0:o1] + c_local @ du_act

                # Virtual
                if self.nvirt > 0:
                    o_virt = o_mo[:, o1:] 
                    do_virt = np.copy(do_mo[:, o1:])
                    
                    if (block_orth):
                        if o0 > 0:
                            o_virt -= u[:, :o0] @ u[:, :o0].T @ o_mo[:, o1:]
                            do_virt -= du_core @ u[:, :o0].T @ o_mo[:, o1:]
                            do_virt -= u[:, :o0] @ du_core.T @ o_mo[:, o1:]
                            do_virt -= u[:, :o0] @ u[:, :o0].T @ do_mo[:, o1:]
                        
                        o_virt -= u[:, o0:o1] @ u[:, o0:o1].T @ o_mo[:, o1:]
                        do_virt -= du_act @ u[:, o0:o1].T @ o_mo[:, o1:]
                        do_virt -= u[:, o0:o1] @ du_act.T @ o_mo[:, o1:]
                        do_virt -= u[:, o0:o1] @ u[:, o0:o1].T @ do_mo[:, o1:]
                    
                    dm_virt = do_virt.T @ o_virt + o_virt.T @ do_virt
                    rhs_m = - m_v_inv @ dm_virt @ m_v_inv
                    dm_sqrt_inv = scipy.linalg.solve_sylvester(m_v_sqrt_inv, m_v_sqrt_inv, rhs_m)
                    du_virt = do_virt @ m_v_sqrt_inv + o_virt @ dm_sqrt_inv
                    self.grad_coeff[iat, idim, :, o1:] = dc_local @ u[:, o1:] + c_local @ du_virt

    ###################################
    # Exact factorization terms
    ###################################
    # We need information of multiple trajectories............
    def calculate_qmom(self):
        pass
    
    def calculate_qmom(self):
        pass

    def calculate_enc(self):
        pass
    

