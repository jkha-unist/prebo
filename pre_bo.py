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
        ne_cas_sum = ne_cas[0] + ne_cas[1] # number of electrons
        self.ncsf = int((2*spin+1) / (no_cas+1) * math.comb(no_cas+1, int(ne_cas_sum/2-spin)) * math.comb(no_cas+1, int(ne_cas_sum/2+spin+1))) # number of CSFs
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
        self.h_mo = np.zeros((self.norb, self.norb)) # MO total 1e integral (kin + v_en) (nmo, nmo)
        self.v_mo = np.zeros((self.norb, self.norb, self.norb, self.norb)) # MO 2e integral  (nmo, nmo, nmo, nmo)
        
        # MO integral gradients 
        self.d_mo = np.zeros((self.nat, 3, self.norb, self.norb)) # MO derivative coupling (nat, 3, nmo, nmo)
        self.dh_mo = np.zeros((self.nat, 3, self.norb, self.norb)) # nuclear gradient of MO 1e integral (nat, 3, nmo, nmo)
        self.dv_mo = np.zeros((self.nat, 3, self.norb, self.norb, self.norb, self.norb)) # nuclear gradient of MO 2e integral (nat, 3, nmo, nmo, nmo, nmo)
        self.g_mo = np.zeros((self.nat, self.norb, self.norb)) # MO scalar coupling (nat, nmo, nmo)
        self.d_dot_d_mo = np.zeros((self.nat, self.norb, self.norb, self.norb, self.norb)) # MO dd term (nat, 3, nmo, nmo)
        
        # CASCI
        self.mc = mcscf.CASCI(self.mf, no_cas, ne_cas) # CASCI object
        self.mc.fcisolver = csf_fci.csf_solver(self.mol, smult=2*spin+1) # fci solver with CSF
        #self.mc.mo_coeff[:, :] = self.mo_coeff[:, :]
        self.mc.fcisolver.nroots = self.ncsf # number of states
        self.ncore = self.mc.ncore # number of core spatial orbital
        self.h_cas = np.zeros((self.no_cas, self.no_cas)) # 1e active
        self.v_cas = np.zeros((self.no_cas, self.no_cas)) # TODO 2e active, wrong dimension..!!!!!! It will be overwritten but.. need to be fixed. I think it was no_cas*(no_cas+1)/2 , no_cas*(no_cas+1)/2.. double check
        
        # CSF properties
        self.V_core = 0. # core energy + Vnn
        self.V_csf = np.zeros((self.ncsf, self.ncsf)) # H_BO CSF matrix 
        self.dV_core = np.zeros((self.nat, 3))
        self.dV_csf = np.zeros((self.nat, 3, self.ncsf, self.ncsf)) # nuclear gradient of H_BO CSF matrix
        self.D_csf = np.zeros((self.nat, 3, self.ncsf, self.ncsf)) # derivative coupling between CSFs

        # Dynamics variables
        self.csf_coeff = np.zeros((self.ncsf), dtype=np.complex128) # time-dependent CSF coefficients
        self.force = np.zeros((self.nat, 3))

    def get_csf(self):
        pass

    def get_V_csf(self):
        
        # TODO: make our own version of get_h1cas/_h2cas to use self.h_ao and self_v_ao, mc.get_h1cas/_h2cas calculates AO integrals again..
        self.h_cas, self.V_core = self.mc.get_h1cas() # get active 1e integral and core energy + Vnn
        self.v_cas = self.mc.get_h2cas() # get active 2e integral
        self.V_csf = self.mc.fcisolver.pspace(self.h_cas, self.v_cas, self.no_cas, self.ne_cas)[1] # get CSF Hamiltonian matrix
        #print(self.h_cas)
        #print(self.V_core)
        #print(self.v_cas)
    
    def get_V_csf_2(self):
        
        o0 = self.ncore
        o1 = self.ncore + self.no_cas
        
        self.h_cas  = np.einsum('kp,kl,lq -> pq', self.mo_coeff[:, o0:o1], (self.kin_ao+self.nuc_ao)[:, :], self.mo_coeff[:, o0:o1], optimize=True)
        self.h_cas += 2.0 * np.einsum('kp, la, mq, na, klmn -> pq', self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
        self.h_cas -= np.einsum('kp, lq, ma, na, klmn -> pq', self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)

        self.v_cas = np.einsum('kp, lq, mr, ns, klmn -> pqrs', self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.v_ao[:, :, :, :], optimize=True)
        self.v_cas = ao2mo.restore('4', self.v_cas.transpose(0, 2, 1, 3), self.no_cas)

        self.V_core = self.mc.energy_nuc()
        self.V_core += 2.0 * np.einsum('ka,kl,la -> ', self.mo_coeff[:, :o0], (self.kin_ao+self.nuc_ao)[:, :], self.mo_coeff[:, :o0], optimize=True)
        self.V_core += 2.0 * np.einsum('ka, lb, ma, nb, klmn -> ', self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
        self.V_core -= np.einsum('ka, la, mb, nb, klmn -> ', self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
        #print(self.h_cas)
        #print(self.V_core)
        #print(self.v_cas)
    
    def get_D_csf(self):
        
        zero_h2e = np.zeros(self.v_cas.shape) # zero arrray to use `fcisolver.pspace`
        for iat in range(self.nat):
            for isp in range(3):
                self.D_csf[iat, isp, :, :] = self.mc.fcisolver.pspace(self.d_mo[iat, isp, self.ncore:self.ncore+self.no_cas, self.ncore:self.ncore+self.no_cas], zero_h2e, self.no_cas, self.ne_cas)[1]

    def get_G_csf(self):
        # TODO
        # 1. Full 1e RI dev: 0.5 * (<nab|nab> + \sum dd)^p_q
        # 2-1. Active space 1e RI dev: 0.5 * (<nab p|nab q> + \sum_r d^p_r d^r_q + 2.0*\sum_a d^p_ad^a_q)^p_q
        # 2-2. V_cons RI dev: \sum_a(<nab a|nab a> + \sum_b d^a_b d^b_a)
        pass

    def get_dV_csf(self):
        
        # nuclear gradients
        self.dV_core = grad.rhf.grad_nuc(self.mol)

        # electronic gradients
        self.get_grad_ao()
        
        s_inv = scipy.linalg.inv(self.s_ao)
        # TODO: active space 
        o0 = self.ncore
        o1 = self.ncore + self.no_cas
        for iat in range(self.nat):
            shl0, shl1, p0, p1 = self.aoslices[iat]
            self.grad_coeff[iat, :, :, :] = - 0.5 * np.einsum('kl, clm, mp -> ckp', s_inv[:, :], -self.d_ao[:, :, p0:p1], self.mo_coeff[p0:p1, :], optimize=True)
            self.grad_coeff[iat, :, :, :] += - 0.5 * np.einsum('kl, cml, mp -> ckp', s_inv[:, p0:p1], -self.d_ao[:, :, p0:p1], self.mo_coeff[:, :], optimize=True)
            
            # 1e
            # \nabla C terms
            dh_tmp  = np.einsum('ckp,kl,lq -> cpq', self.grad_coeff[iat, :, :, o0:o1], (self.kin_ao+self.nuc_ao)[:, :], self.mo_coeff[:, o0:o1], optimize=True)
            # AO <\nabla | h | >  terms
            dh_tmp += np.einsum('kp, ckl, lq -> cpq', self.mo_coeff[p0:p1, o0:o1], -(self.dkin_ao+self.dnuc_ao)[:, p0:p1, :], self.mo_coeff[:, o0:o1], optimize=True)
            # AO < |\nabla V_nuc | > terms
            with self.mol.with_rinv_at_nucleus(iat):
                drinv_tmp = self.mol.intor('int1e_iprinv', comp=3)
                drinv_tmp *= -self.mol.atom_charge(iat)
            dh_tmp += np.einsum('kp, ckl, lq -> cpq', self.mo_coeff[:, o0:o1], drinv_tmp[:, :, :], self.mo_coeff[:, o0:o1], optimize=True)
            # effective 1e from 2e
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

            # 2e
            # \nabla C terms
            dv_tmp  = np.einsum('ckp, lq, mr, ns, klmn -> cpqrs', self.grad_coeff[iat, :, :, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.v_ao[:, :, :, :], optimize=True)
            # AO <\nabla, | , > terms
            dv_tmp += np.einsum('kp, lq, mr, ns, cklmn -> cpqrs', self.mo_coeff[p0:p1, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], self.mo_coeff[:, o0:o1], -self.dv_ao[:, p0:p1, :, :, :], optimize=True)
            # Total
            dv_mo = dv_tmp + dv_tmp.transpose(0, 2, 1, 4, 3) + dv_tmp.transpose(0, 3, 4, 1, 2) + dv_tmp.transpose(0, 4, 3, 2, 1)
            
            # effective const from 1e
            self.dV_core[iat, :] += 4.0 * np.einsum('cka,kl,la -> c', self.grad_coeff[iat, :, :, :o0], (self.kin_ao+self.nuc_ao)[:, :], self.mo_coeff[:, :o0], optimize=True)
            self.dV_core[iat, :] += 4.0 * np.einsum('ka, ckl, la -> c', self.mo_coeff[p0:p1, :o0], -(self.dkin_ao+self.dnuc_ao)[:, p0:p1, :], self.mo_coeff[:, :o0], optimize=True)
            self.dV_core[iat, :] += 4.0 * np.einsum('ka, ckl, la -> c', self.mo_coeff[:, :o0], drinv_tmp[:, :, :], self.mo_coeff[:, :o0], optimize=True)

            # effective const from 2e
            # coulomb
            self.dV_core[iat, :] += 8.0 * np.einsum('cka, lb, ma, nb, klmn -> c', self.grad_coeff[iat, :, :, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
            self.dV_core[iat, :] += 8.0 * np.einsum('ka, lb, ma, nb, cklmn -> c', self.mo_coeff[p0:p1, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], -self.dv_ao[:, p0:p1, :, :, :], optimize=True)
            # exchange
            self.dV_core[iat, :] -= 4.0 * np.einsum('cka, la, mb, nb, klmn -> c', self.grad_coeff[iat, :, :, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.v_ao[:, :, :, :], optimize=True)
            self.dV_core[iat, :] -= 4.0 * np.einsum('ka, la, mb, nb, cklmn -> c', self.mo_coeff[p0:p1, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], self.mo_coeff[:, :o0], -self.dv_ao[:, p0:p1, :, :, :], optimize=True)
            
            for isp in range(3):
                self.dV_csf[iat, isp, :, :] = self.mc.fcisolver.pspace(dh_mo[isp, :, :], dv_mo.transpose(0,1,3,2,4)[isp, :, :, :, :], self.no_cas, self.ne_cas)[1] # get CSF Hamiltonian matrix
                
                #dv_mo_tmp = ao2mo.restore('4', dv_mo[isp].transpose(0, 2, 1, 3), self.no_cas)
                #self.dV_csf[iat, isp, :, :] = self.mc.fcisolver.pspace(dh_mo[isp, :, :], dv_mo_tmp, self.no_cas, self.ne_cas)[1] # get CSF Hamiltonian matrix
                
    
    def get_dV_csf_2(self):
        
        # nuclear gradients
        self.dV_core = grad.rhf.grad_nuc(self.mol)

        # electronic gradients
        self.get_grad_ao()
        
        s_inv = scipy.linalg.inv(self.s_ao)
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
                self.dV_csf[iat, isp, :, :] = self.mc.fcisolver.pspace(dh_mo[isp, :, :], zero_v, self.no_cas, self.ne_cas)[1] # get CSF Hamiltonian matrix
        


    def calculate_force(self):
        self.force = -np.einsum('I, AcIJ, J -> Ac', self.csf_coeff.conj(), self.dV_csf, self.csf_coeff).real
        #self.force -= -2.0 * np.einsum('I, AcKI, KJ, J -> Ac', self.csf_coeff.conj(), self.D_csf, self.V_csf, self.csf_coeff).real
        self.force -= self.dV_core

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
            self.d_mo[iat, :, :, :]  = 0.5 * np.einsum('kp,akl,lq->apq', self.mo_coeff[:, :], -self.d_ao[:, :, p0:p1], self.mo_coeff[p0:p1, :], optimize=True)
            self.d_mo[iat, :, :, :] -= 0.5 * np.einsum('kp,alk,lq->apq', self.mo_coeff[p0:p1, :], -self.d_ao[:, :, p0:p1], self.mo_coeff[:, :], optimize=True)
        
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
        
    def align_mo_coeff(self, mol_old, mo_old):
        
        mo = np.copy(self.mo_coeff)
        #idx, s = mo_mapping.mo_map(mol_old, mo_old, self.mol, self.mo_coeff)
        s = gto.intor_cross('int1e_ovlp', mol_old, self.mol)
        s = reduce(np.dot, (mo_old.T, s, mo))
        StS = s.T @ s
        StS_sqrt = scipy.linalg.sqrtm(StS)
        V = scipy.linalg.inv(StS_sqrt)
        U =  s @ V
        mo = self.mo_coeff @ U.T
        self.mo_coeff = np.copy(mo)

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

    def align_mo_coeff_lowdin(self, mol_old, mo_old):
        """
        Update self.mo_coeff using Lowdin propagation (symmetric orthonormalization).
        C_new = C_old * (C_old.T * S_new * C_old)^-1/2
        This is the orbital update model assumed by the analytical gradient formulas.
        """
        s_new = self.mol.intor('int1e_ovlp')
        tmp = mo_old.T @ s_new @ mo_old
        val, vec = scipy.linalg.eigh(tmp)
        tmp_inv_sqrt = vec @ np.diag(1.0/np.sqrt(val)) @ vec.T
        self.mo_coeff = mo_old @ tmp_inv_sqrt

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
   
    #def get_grad_mo(self, iat):
    #    
    #    s_inv = scipy.linalg.inv(self.s_ao)
    #    for iat in range(self.nat):
    #        shl0, shl1, p0, p1 = self.aoslices[iat]
    #        self.grad_coeff[iat, :, :, :] = - 0.5 * np.einsum('kl, clm, mp -> ckp', s_inv[:, :], self.d_ao[:, :, p0:p1], self.mo_coeff[p0:p1, :], optimize=True)
    #        self.grad_coeff[iat, :, :, :] += - 0.5 * np.einsum('kl, cml, mp -> ckp', s_inv[:, p0:p1], self.d_ao[:, :, p0:p1], self.mo_coeff[:, :], optimize=True)
    #        # \nabla C terms
    #        self.dh_mo[iat, :, :, :] = np.einsum('ckp,kl,lq', self.grad_coeff[iat, :, :, :], (self.kin_ao+self.nuc_ao)[:, :], self.mo_coeff[:, :])
    #        self.dh_mo[iat, :, :, :] -= np.einsum('kp,kl,clq', self.mo_coeff[:, :], (self.kin_ao+self.nuc_ao)[:, :], self.grad_coeff[iat, :, :, :])
    #        # AO <\nabla | h | >  terms
    #        self.dh_mo[iat, :, :, :] -= np.einsum('kp, ckl, lq -> cpq', self.mo_coeff[p0:p1, :], (self.dkin_ao+self.dnuc_ao)[:, p0:p1, :], self.mo_coeff[:, :], optimize=True)
    #        self.dh_mo[iat, :, :, :] -= np.einsum('kp, clk, lq -> cpq', self.mo_coeff[:, :], (self.dkin_ao+self.dnuc_ao)[:, p0:p1, :], self.mo_coeff[p0:p1, :], optimize=True)
    #        # AO < |\nabla V_nuc | > terms
    #        with self.mol.with_rinv_at_nucleus(iat):
    #            drinv_tmp = self.mol.intor('int1e_iprinv', comp=3)
    #            drinv_tmp *= -self.mol.atom_charge(iat)
    #        self.dh_mo[iat, :, :, :] -= np.einsum('kp, ckl, lq -> cpq', self.mo_coeff[p0:p1, :], drinv_tmp[:, p0:p1, :], self.mo_coeff[:, :], optimize=True)
    #        self.dh_mo[iat, :, :, :] -= np.einsum('kp, clk, lq -> cpq', self.mo_coeff[:, :], drinv_tmp[:, p0:p1, :], self.mo_coeff[p0:p1, :], optimize=True)

    #        self.dv_mo[iat, :, :, :, :, :]  = np.einsum('ckp, lq, mr, ns, klmn -> cpqrs', self.grad_coeff[iat, :, :, :], self.mo_coeff[:, :], self.mo_coeff[:, :], self.mo_coeff[:, :], self.v_ao[:, :, :, :], optimize=True)
    #        self.dv_mo[iat, :, :, :, :, :] += np.einsum('clq, kp, mr, ns, klmn -> cpqrs', self.grad_coeff[iat, :, :, :], self.mo_coeff[:, :], self.mo_coeff[:, :], self.mo_coeff[:, :], self.v_ao[:, :, :, :], optimize=True)
    #        self.dv_mo[iat, :, :, :, :, :] += np.einsum('cmr, lq, kp, ns, klmn -> cpqrs', self.grad_coeff[iat, :, :, :], self.mo_coeff[:, :], self.mo_coeff[:, :], self.mo_coeff[:, :], self.v_ao[:, :, :, :], optimize=True)
    #        self.dv_mo[iat, :, :, :, :, :] += np.einsum('cns, lq, mr, kp, klmn -> cpqrs', self.grad_coeff[iat, :, :, :], self.mo_coeff[:, :], self.mo_coeff[:, :], self.mo_coeff[:, :], self.v_ao[:, :, :, :], optimize=True)

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

