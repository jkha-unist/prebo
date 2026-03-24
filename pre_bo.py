from pyscf import gto, scf, mcscf, fci, nac, csf_fci, ao2mo, grad
from pyscf.tools import mo_mapping, molden
from functools import reduce
import scipy
import numpy as np
import pickle
import math
import time

class pre_BO(object):
    
    def __init__(self, mol, ne_cas, no_cas, spin=0):
        
        # System
        self.mol = mol # molecule object of pyscf
        self.nat = mol.natm # number of atoms
        self.pos = mol.atom_coords(unit='Bohr') # nuclear position
        self.vel = np.zeros((self.nat, 3)) # nuclear velocity
        self.mass = gto.mole.atom_mass_list(mol, isotope_avg=True) * 1822.8884853324 # nuclear mass
        
        self.mol_old = mol.copy() # backup molecule
        
        self.norb = mol.nao # number of total spatial orbitals
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
        self.d_ao = self.mol.intor('int1e_ipovlp').transpose((0,2,1)) # (3, nao, nao)  <\chi_l | \nabla_e \chi_m> --> need to convert to d/dR_nuc which requires ao_slice for indices
        self.dkin_ao = np.zeros((3, self.norb, self.norb)) # gradient of AO 1e integral (3, nao, nao)
        self.dnuc_ao = np.zeros((3, self.norb, self.norb)) # gradient of AO 1e integral (3, nao, nao)
        self.dv_ao = np.zeros((3, self.norb, self.norb, self.norb, self.norb)) # gradient of AO 2e integral (3, nao, nao, nao, nao)
        
        # HF
        self.mf = scf.RHF(self.mol) # HF object
        #self.mf.run(conv_tol=1e-8)
        
        # MO integrals
        #self.mo_coeff[:, :] = self.mf.mo_coeff[:, :] # (nao, nmo) In fact, nao=nmo. Just to clarify the row and column.
        self.mo_coeff = np.zeros((self.norb, self.norb)) # MO coefficient
        self.mo_coeff_old = np.zeros_like(self.mo_coeff) # backup MO coefficient
        self.grad_coeff = np.zeros((self.nat, 3, self.norb, self.norb)) # nuclear gradient of MO coefficient (nat, 3, nao, nmo)  
        self.grad_coeff_old = np.zeros((self.nat, 3, self.norb, self.norb)) # nuclear gradient of MO coefficient (nat, 3, nao, nmo)  
        self.h_mo = np.zeros((self.norb, self.norb)) # MO total 1e integral (kin + v_en) (nmo, nmo)
        self.v_mo = np.zeros((self.norb, self.norb, self.norb, self.norb)) # MO 2e integral  (nmo, nmo, nmo, nmo)
        self.u = np.zeros((self.norb, self.norb))
        
        # MO integral gradients 
        self.d_mo = np.zeros((self.nat, 3, self.norb, self.norb)) # MO derivative coupling (nat, 3, nmo, nmo)
        self.dh_mo = np.zeros((self.nat, 3, self.norb, self.norb)) # nuclear gradient of MO 1e integral (nat, 3, nmo, nmo)
        self.dv_mo = np.zeros((self.nat, 3, self.norb, self.norb, self.norb, self.norb)) # nuclear gradient of MO 2e integral (nat, 3, nmo, nmo, nmo, nmo)
        self.g_mo = np.zeros((self.nat, self.norb, self.norb)) # MO scalar coupling (nat, nmo, nmo)
        self.d_dot_d_mo = np.zeros((self.nat, self.norb, self.norb, self.norb, self.norb)) # MO dd term (nat, 3, nmo, nmo)
        
        # CASCI
        ne_cas_sum = ne_cas[0] + ne_cas[1] # number of electrons
        transformer = csf_fci.CSFTransformer(self.no_cas, self.ne_cas[0], self.ne_cas[1], smult=2*spin+1)
        self.ncsf = transformer.ncsf

        self.mc = mcscf.CASCI(self.mf, no_cas, ne_cas) # CASCI object
        self.mc.fcisolver = csf_fci.csf_solver(self.mol, smult=2*spin+1) # fci solver with CSF
        #self.mc.mo_coeff[:, :] = self.mo_coeff[:, :]
        self.mc.fcisolver.nroots = self.ncsf # number of states
        self.ncore = self.mc.ncore # number of core spatial orbital
        self.nvirt = self.norb - self.ncore - self.no_cas # number of core spatial orbital
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

        self.V_nuc = 0. # V_nn
        self.dV_nuc = np.zeros((self.nat, 3))
        
        # Dynamics variables
        self.csf_coeff = np.zeros((self.ncsf), dtype=np.complex128) # time-dependent CSF coefficients
        self.force = np.zeros((self.nat, 3))

    def get_int_ao(self):

        self.s_ao = self.mol.intor('int1e_ovlp') # AO overlap (nao, nao)
        self.kin_ao = self.mol.intor('int1e_kin') # AO 1e kinetic energy integral
        self.nuc_ao = self.mol.intor('int1e_nuc') # AO 1e e-n couloumb integral
        self.v_ao = self.mol.intor('int2e', aosym='s1').transpose(0, 2, 1, 3) # AO 2e e-e coulomb integral
        self.d_ao = self.mol.intor('int1e_ipovlp').transpose((0,2,1)) # (3, nao, nao)  <\chi_l | \nabla_e \chi_m> --> need to convert to d/dR_nuc which requires ao_slice for indices
    
    def get_grad_ao(self):
        
        self.dkin_ao = self.mol.intor('int1e_ipkin')
        self.dnuc_ao = self.mol.intor('int1e_ipnuc')
        self.dv_ao = self.mol.intor('int2e_ip1').transpose(0, 1, 3, 2, 4)
    
    def get_int_mo(self):
        
        C = self.mo_coeff
        
        # Duplicative, since get_h1cas and get_h2cas calculate these.

        # 1e
        #self.h_mo = np.einsum('kl,kp,lq->pq', (self.nuc_ao + self.kin_ao), C, C)
        # 2e
        #self.v_mo = ao2mo.full(self.v_ee_ao, C)
        
        # d & g
        d_ao_tmp = np.zeros((3, self.norb, self.norb))
        g_ao_tmp = np.zeros((self.norb, self.norb))
        dS_tmp = np.zeros((3, self.norb, self.norb)) # \nabla_\nu S 
        d2S_tmp = np.zeros((self.norb, self.norb)) # \nabla_\nu^2 S

        # d
        for iat in range(self.nat):
            shl0, shl1, p0, p1 = self.aoslices[iat]
            self.d_mo[iat, :, :, :]  = np.einsum('kp,ckl,lq->cpq', self.mo_coeff[:, :], -self.d_ao[:, :, p0:p1], self.mo_coeff[p0:p1, :], optimize=True)
            self.d_mo[iat, :, :, :] += np.einsum('kp,kl,ckq->cpq', self.mo_coeff[:, :], self.s_ao[:, :], self.grad_coeff[iat, :, :, :], optimize=True)
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
        
        s_inv = scipy.linalg.inv(self.s_ao)
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
        self.force = -np.einsum('I, AcIJ, J -> Ac', self.csf_coeff.conj(), self.dV_csf, self.csf_coeff).real
        self.force += 2.0 * np.einsum('I, AcKI, KJ, J -> Ac', self.csf_coeff.conj(), self.D_csf, self.V_csf, self.csf_coeff).real
        self.force -= self.dV_core

    
    def align_mo_coeff_spacewise(self, mol_old, mo_old, block_orth=True):
        
        mo = np.copy(self.mo_coeff)
        
        o0, o1 = self.ncore, self.ncore+self.no_cas
        
        o_ao = gto.intor_cross('int1e_ovlp', self.mol, mol_old)
        
        o_core = reduce(np.dot, (mo, o_ao, mo_old[:, :o0]))
        m = o_core.T @ o_core
        m_vals, m_vecs = scipy.linalg.eigh(m)
        m_sqrt_inv = m_vecs @ np.diag(1.0 / np.sqrt(m_vals)) @ m_vecs.T
        u_core = o_core @ m_sqrt_inv
        self.mat_core = [m_vals, m_vecs]
        
        tmp = reduce(np.dot, (mo, o_ao, mo_old[:, o0:o1]))
        o_act = np.copy(tmp)
        
        if(block_orth):
            o_act -= u_core @ u_core.T @ tmp
        
        m = o_act.T @ o_act
        m_vals, m_vecs = scipy.linalg.eigh(m)
        m_sqrt_inv = m_vecs @ np.diag(1.0 / np.sqrt(m_vals)) @ m_vecs.T
        u_act = o_act @ m_sqrt_inv
        self.mat_act = [m_vals, m_vecs]
        
        tmp = reduce(np.dot, (mo, o_ao, mo_old[:, o1:]))
        o_virt = np.copy(tmp)
        
        if(block_orth):
            o_virt -= u_core @ u_core.T @ tmp
            o_virt -= u_act @ u_act.T @ tmp
        
        m = o_virt.T @ o_virt
        m_vals, m_vecs = scipy.linalg.eigh(m)
        m_sqrt_inv = m_vecs @ np.diag(1.0 / np.sqrt(m_vals)) @ m_vecs.T
        u_virt = o_virt @ m_sqrt_inv
        self.mat_virt = [m_vals, m_vecs]

        self.u = np.column_stack((u_core, u_act, u_virt))
        
        mo_new =  self.mo_coeff @ self.u
        self.mo_coeff = np.copy(mo_new)

        I_test = self.u @ self.u.T
        f = open(f"uuT_{block_orth}.dat","w")
        for i in range(self.norb):
            for j in range(self.norb):
                f.write(f"{i} {j} {I_test[i, j]}\n")
            f.write("\n")
        f.close()
        
        I_test = self.u.T @ self.u
        f = open(f"uTu_{block_orth}.dat","w")
        for i in range(self.norb):
            for j in range(self.norb):
                f.write(f"{i} {j} {I_test[i, j]}\n")
            f.write("\n")
        f.close()
        
        tmp = np.absolute(mo_new.T @ o_ao @ mo_old)
        ovlp = tmp @ tmp.T
        f = open(f"ovlp_after_alignment_{block_orth}.dat","w")
        for i in range(self.norb):
            ovlp[i, i] = 0.0
            for j in range(self.norb):
                f.write(f"{i} {j} {ovlp[i, j]}\n")
            f.write("\n")
        f.close()
        
    
    def get_grad_coeff_spacewise(self, block_orth=True):
        """
        Calculate analytical nuclear gradient of Tracked Lowdin Molecular Orbitals.
        Implements equations 15-21 from Pre_BO_direct_dynamics.pdf.
        """
        o0, o1 = self.ncore, self.ncore+self.no_cas

        # 1. AO Overlap and its nuclear gradient
        s = self.s_ao
        # ds_ao_raw[c, mu, nu] = <nabla_c mu | nu>
        # Nuclear gradient d/dR_A = -nabla_e for basis functions on A.
        ds_ao_raw = -self.mol.intor('int1e_ipovlp')
        
        # 2. Local Lowdin Orbitals: C_local = S^-1/2 
        s_vals, s_vecs = scipy.linalg.eigh(s)
        sq_s = np.sqrt(s_vals)
        s_sqrt = s_vecs @ np.diag(sq_s) @ s_vecs.T
        s_inv = s_vecs @ np.diag(1.0 / s_vals) @ s_vecs.T
        c_local = s_vecs @ np.diag(1.0 / sq_s) @ s_vecs.T
        
        # 3. Inter-geometry overlap matrix: O = C_local^T * O_AO * C
        # O_AO = <chi(R(t)) | chi(R(t-dt))>
        o_ao = gto.intor_cross('int1e_ovlp', self.mol, self.mol_old)
        # C_local is symmetric, so C_local^T = C_local
        o_mo = reduce(np.dot, (c_local, o_ao, self.mo_coeff_old))
        
        # 4. M = O * O^T
        m_vals, m_vecs = self.mat_core[0], self.mat_core[1]
        sq_m = np.sqrt(m_vals)
        m_c_inv = m_vecs @ np.diag(1.0 / m_vals) @ m_vecs.T
        m_c_sqrt = m_vecs @ np.diag(sq_m) @ m_vecs.T
        m_c_sqrt_inv = m_vecs @ np.diag(1.0 / sq_m) @ m_vecs.T
        
        m_vals, m_vecs = self.mat_act[0], self.mat_act[1]
        sq_m = np.sqrt(m_vals)
        m_a_inv = m_vecs @ np.diag(1.0 / m_vals) @ m_vecs.T
        m_a_sqrt = m_vecs @ np.diag(sq_m) @ m_vecs.T
        m_a_sqrt_inv = m_vecs @ np.diag(1.0 / sq_m) @ m_vecs.T
        
        m_vals, m_vecs = self.mat_virt[0], self.mat_virt[1]
        sq_m = np.sqrt(m_vals)
        m_v_inv = m_vecs @ np.diag(1.0 / m_vals) @ m_vecs.T
        m_v_sqrt = m_vecs @ np.diag(sq_m) @ m_vecs.T
        m_v_sqrt_inv = m_vecs @ np.diag(1.0 / sq_m) @ m_vecs.T
        
        u = np.copy(self.u)
        
        # 6. Precompute Gradient of O_AO:<nabla_nu chi(R(t)) | chi(R(t-dt))> & <chi(R(t)) | \nabla chi(R(t-dt))>
        do_ao_raw_left = -gto.intor_cross('int1e_ipovlp', self.mol, self.mol_old)
        do_ao_raw_right = -gto.intor_cross('int1e_ipovlp', self.mol_old, self.mol).transpose(0, 2, 1)

        for iat in range(self.nat):
            shl0, shl1, p0, p1 = self.aoslices[iat]
            
            # nabla S for this atom
            ds = np.zeros((3, self.norb, self.norb))
            ds[:, p0:p1, :] = ds_ao_raw[:, p0:p1, :]
            ds[:, :, p0:p1] += ds_ao_raw[:, p0:p1, :].transpose(0, 2, 1)
            
            # nabla O_AO for this atom
            do_ao = np.zeros((3, self.norb, self.norb))
            do_ao[:, p0:p1, :] = do_ao_raw_left[:, p0:p1, :] 
            #do_ao[:, :, p0:p1] += do_ao_raw_right[:, :, p0:p1]
            
            for idim in range(3):
                
                # Gradient of local MO coefficient, S^{-1/2}
                rhs_s = - s_inv @ ds[idim] @ s_inv
                dc_local = scipy.linalg.solve_sylvester(c_local, c_local, rhs_s)
                # Gradient of local time overlap
                do_mo = dc_local @ o_ao @ self.mo_coeff_old + c_local @ do_ao[idim] @ self.mo_coeff_old #+ c_local @ o_ao @ self.grad_coeff_old[iat, idim]
                
                # Core
                dm_core = do_mo[:, :o0].T @ o_mo[:, :o0] + o_mo[:, :o0].T @ do_mo[:, :o0]
                rhs_m = - m_c_inv @ dm_core @ m_c_inv
                dm_sqrt_inv = scipy.linalg.solve_sylvester(m_c_sqrt_inv, m_c_sqrt_inv, rhs_m)
                du_core = do_mo[:, :o0] @ m_c_sqrt_inv + o_mo[:, :o0] @ dm_sqrt_inv
                self.grad_coeff[iat, idim, :, :o0] = dc_local @ u[:, :o0] + c_local @ du_core

                # Active
                o_act = o_mo[:, o0:o1] 
                do_act = np.copy(do_mo[:, o0:o1])
                
                if(block_orth):
                    
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
                o_virt = o_mo[:, o1:] 
                do_virt = np.copy(do_mo[:, o1:])
                
                if (block_orth):
                    
                    o_virt -= u[:, :o0] @ u[:, :o0].T @ o_mo[:, o1:] + u[:, o0:o1] @ u[:, o0:o1].T @ o_mo[:, o1:]
                    
                    do_virt -= du_core @ u[:, :o0].T @ o_mo[:, o1:]
                    do_virt -= u[:, :o0] @ du_core.T @ o_mo[:, o1:]
                    do_virt -= u[:, :o0] @ u[:, :o0].T @ do_mo[:, o1:]
                    do_virt -= du_act @ u[:, o0:o1].T @ o_mo[:, o1:]
                    do_virt -= u[:, o0:o1] @ du_act.T @ o_mo[:, o1:]
                    do_virt -= u[:, o0:o1] @ u[:, o0:o1].T @ do_mo[:, o1:]
                
                dm_virt = do_virt.T @ o_virt + o_virt.T @ do_virt
                rhs_m = - m_v_inv @ dm_virt @ m_v_inv
                dm_sqrt_inv = scipy.linalg.solve_sylvester(m_v_sqrt_inv, m_v_sqrt_inv, rhs_m)
                du_virt = do_virt @ m_v_sqrt_inv + o_virt @ dm_sqrt_inv
                self.grad_coeff[iat, idim, :, o1:] = dc_local @ u[:, o1:] + c_local @ du_virt

    def get_grad_coeff_symm(self):
        """
        Calculate analytical nuclear gradient of Tracked Lowdin Molecular Orbitals.
        Implements equations 15-21 from Pre_BO_direct_dynamics.pdf.
        """
        # 1. AO Overlap and its nuclear gradient
        s = self.s_ao
        # ds_ao_raw[c, mu, nu] = <nabla_c mu | nu>
        # Nuclear gradient d/dR_A = -nabla_e for basis functions on A.
        ds_ao_raw = -self.mol.intor('int1e_ipovlp')
        
        for iat in range(self.nat):
            shl0, shl1, p0, p1 = self.aoslices[iat]
            
            # nabla S for this atom
            self.grad_coeff[iat, :] = 0.5 * np.einsum('kp, clk, lq -> cpq', self.mo_coeff[p0:p1, :], ds_ao_raw[:, :, p0:p1], self.mo_coeff[:, :])
            self.grad_coeff[iat, :] -= 0.5 * np.einsum('kp, ckl, lq -> cpq', self.mo_coeff[:, :], ds_ao_raw[:, :, p0:p1], self.mo_coeff[p0:p1, :])

    def get_grad_coeff(self):
        """
        Calculate analytical nuclear gradient of Tracked Lowdin Molecular Orbitals.
        Implements equations 15-21 from Pre_BO_direct_dynamics.pdf.
        """
        # 1. AO Overlap and its nuclear gradient
        s = self.s_ao
        # ds_ao_raw[c, mu, nu] = <nabla_c mu | nu>
        # Nuclear gradient d/dR_A = -nabla_e for basis functions on A.
        ds_ao_raw = -self.mol.intor('int1e_ipovlp')
        
        # 2. Local Lowdin Orbitals: C_local = S^-1/2 
        s_vals, s_vecs = scipy.linalg.eigh(s)
        sq_s = np.sqrt(s_vals)
        s_sqrt = s_vecs @ np.diag(sq_s) @ s_vecs.T
        s_inv = s_vecs @ np.diag(1.0 / s_vals) @ s_vecs.T
        c_local = s_vecs @ np.diag(1.0 / sq_s) @ s_vecs.T
        
        # 3. Inter-geometry overlap matrix: O = C_local^T * O_AO * C
        # O_AO = <chi(R(t)) | chi(R(t-dt))>
        o_ao = gto.intor_cross('int1e_ovlp', self.mol, self.mol_old)
        # C_local is symmetric, so C_local^T = C_local
        o = reduce(np.dot, (c_local, o_ao, self.mo_coeff_old))
        
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

        for iat in range(self.nat):
            shl0, shl1, p0, p1 = self.aoslices[iat]
            
            # nabla S for this atom
            dns = np.zeros((3, self.norb, self.norb))
            dns[:, p0:p1, :] = ds_ao_raw[:, p0:p1, :]
            dns[:, :, p0:p1] += ds_ao_raw[:, p0:p1, :].transpose(0, 2, 1)
            
            # nabla O_AO for this atom
            dno_ao = np.zeros((3, self.norb, self.norb))
            dno_ao[:, p0:p1, :] = do_ao_raw_left[:, p0:p1, :] 
            #dno_ao[:, :, p0:p1] += do_ao_raw_right[:, :, p0:p1]
            
            for idim in range(3):
                # Eq 17 context: Sylvester for dC_local = d(S^-1/2)
                # C_local * dC_local + dC_local * C_local = dS^-1 = - S^-1 * dS * S^-1
                #rhs_s = - reduce(np.dot, (s_inv, dns[idim], s_inv))
                rhs_s = - s_inv @ dns[idim] @ s_inv
                dc_local = scipy.linalg.solve_sylvester(c_local, c_local, rhs_s)
                
                # Eq 21: Gradient of inter-geometry overlap O
                # dO = dC_local * O_AO * C' + C_local * dO_AO * C'
                dno = dc_local @ o_ao @ self.mo_coeff_old + c_local @ dno_ao[idim] @ self.mo_coeff_old #+ c_local @ o_ao @ self.grad_coeff_old[iat, idim]
                
                # Eq 20: Gradient of metric M = O * O^T
                # dM = dO * O^T + O * dO^T
                dnm = dno @ o.T + o @ dno.T
                
                # Eq 19: Sylvester for dM^-1/2
                # M^-1/2 * dM_sqrt_inv + dM_sqrt_inv * M^-1/2 = dM^-1 = - M^-1 * dM * M^-1
                #rhs_m = - reduce(np.dot, (m_inv, dnm, m_inv))
                rhs_m = - m_inv @ dnm @ m_inv
                dm_sqrt_inv = scipy.linalg.solve_sylvester(m_sqrt_inv, m_sqrt_inv, rhs_m)
                
                # Eq 18: Gradient of tracking matrix U
                # dU = M^-1/2 * dO + dM_sqrt_inv * O
                dnu = m_sqrt_inv @ dno + dm_sqrt_inv @ o
                
                # Eq 15: Final MO gradient
                # dC = dC_local * U + C_local * dU
                self.grad_coeff[iat, idim, :, :] = dc_local @ u + c_local @ dnu

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
    

    def align_mo_coeff_simple(self, mol_old, mo_old):
        
        mo = np.copy(self.mo_coeff)
        #idx, s = mo_mapping.mo_map(mol_old, mo_old, self.mol, self.mo_coeff)
        s = gto.intor_cross('int1e_ovlp', mol_old, self.mol)
        s = reduce(np.dot, (mo_old.T, s, mo))
        
        # detect swap
        col_ind = np.argmax(np.abs(s), axis=1) # i-th element is the index of new MO corresponding to the i-th old MO
        row_ind = np.arange(self.norb)
       
        # detect sign flip
        phases = np.sign(s[row_ind, col_ind])
        phases[phases == 0] = 1.0  # Prevent zeroing out orbitals if exact 0 overlap occurs, although it is highly unlikely.

        self.mo_coeff = mo[:, col_ind] * phases # bring new MO to i-th column (bring new index --> old index)
    
    def align_mo_coeff(self, mol_old, mo_old):
        
        mo = np.copy(self.mo_coeff)
        o_ao = gto.intor_cross('int1e_ovlp', self.mol, mol_old)
        o = reduce(np.dot, (mo.T, o_ao, mo_old))
        m = o @ o.T
        m_vals, m_vecs = scipy.linalg.eigh(m)
        m_sqrt_inv = m_vecs @ np.diag(1.0 / np.sqrt(m_vals)) @ m_vecs.T
        u = m_sqrt_inv @ o
        mo_new =  self.mo_coeff @ u
        self.mo_coeff = np.copy(mo_new)
        
        I_test = u @ u.T
        f = open("uuT_full.dat","w")
        for i in range(self.norb):
            for j in range(self.norb):
                f.write(f"{i} {j} {I_test[i, j]}\n")
            f.write("\n")
        f.close()
        
        I_test = u.T @ u
        f = open("uTu_full.dat","w")
        for i in range(self.norb):
            for j in range(self.norb):
                f.write(f"{i} {j} {I_test[i, j]}\n")
            f.write("\n")
        f.close()
        
        tmp = np.absolute(mo_new.T @ o_ao @ mo_old)
        ovlp = tmp @ tmp.T
        f = open("ovlp_after_alignment_full.dat","w")
        for i in range(self.norb):
            ovlp[i, i] = 0.0
            for j in range(self.norb):
                f.write(f"{i} {j} {ovlp[i, j]}\n")
            f.write("\n")
        f.close()
    
