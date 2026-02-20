from pyscf import gto, scf, mcscf, fci, nac, csf_fci, ao2mo
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
        self.csf_coeff = np.zeros((self.ncsf), dtype=np.complex128) # time-dependent CSF coefficients
        
        # AO properties
        self.s_ao = self.mol.intor('int1e_ovlp') # AO overlap (nao, nao)
        self.kin_e_ao = self.mol.intor('int1e_kin') # AO 1e kinetic energy integral
        self.v_en_ao = self.mol.intor('int1e_nuc') # AO 1e e-n couloumb integral
        self.v_ee_ao = self.mol.intor('int2e', aosym='s1') # AO 2e e-e coulomb integral
        self.d_ao = -1.0 * mol.intor('int1e_ipovlp').transpose((0,2,1)) # (3, nao, nao)  <\chi_l | \nabla_e \chi_m> --> need to convert to d/dR_nuc which requires ao_slice for indices
        self.dv_ee_ao = mol.intor('int2e_ip1') # (3, nao, nao, nao, nao) (\nabla k,l|m,n)  --> need to convert to d/dR_nuc
        #self.dv_en_ao = ????
        #self.dkin_e_ao = ????
        
        # HF
        self.mf = scf.RHF(self.mol) # HF object
        #self.mf.run(conv_tol=1e-8)
        
        # MO properties
        #self.mo_coeff[:, :] = self.mf.mo_coeff[:, :] # (nao, nmo) In fact, nao=nmo. Just to clarify the row and column.
        self.mo_coeff = np.zeros((self.norb, self.norb)) # MO coefficient
        self.mo_coeff_old = np.zeros_like(self.mo_coeff) # backup MO coefficient
        self.grad_coeff = np.zeros((self.nat, 3, self.norb, self.norb)) # nuclear gradient of MO coefficient (nat, 3, nao, nmo)  
        self.h_mo = np.zeros((self.norb, self.norb)) # MO total 1e integral (kin + v_en) (nmo, nmo)
        self.v_mo = np.zeros((self.norb, self.norb, self.norb, self.norb)) # MO 2e integral  (nmo, nmo, nmo, nmo)
        self.dh_mo = np.zeros((self.nat, 3, self.norb, self.norb)) # nuclear gradient of MO 1e integral (nat, 3, nmo, nmo)
        self.dv_ee_mo = np.zeros((self.nat, 3, self.norb, self.norb)) # nuclear gradient of MO 2e integral (nat, 3, nmo, nmo)
        self.d_mo = np.zeros((self.nat, 3, self.norb, self.norb)) # MO derivative coupling (nat, 3, nmo, nmo)
        self.g_mo = np.zeros((self.nat, self.norb, self.norb)) # MO scalar coupling (nat, nmo, nmo)
        self.d_dot_d_mo = np.zeros((self.nat, self.norb, self.norb, self.norb, self.norb)) # MO dd term (nat, 3, nmo, nmo)
        
        # CASCI
        self.mc = mcscf.CASCI(self.mf, no_cas, ne_cas) # CASCI object
        self.mc.fcisolver = csf_fci.csf_solver(self.mol, smult=2*spin+1) # fci solver with CSF
        #self.mc.mo_coeff[:, :] = self.mo_coeff[:, :]
        self.mc.fcisolver.nroots = self.ncsf # number of states
        self.ncore = self.mc.ncore # number of core spatial orbital
        self.e_core = 0. # core energy + Vnn
        self.h_cas = np.zeros((self.no_cas, self.no_cas)) # 1e active
        self.v_cas = np.zeros((self.no_cas, self.no_cas)) # TODO 2e active, wrong dimension..!!!!!! It will be overwritten but.. need to be fixed. I think it was no_cas*(no_cas+1)/2 , no_cas*(no_cas+1)/2.. double check
        
        # CSF properties
        self.V_csf = np.zeros((self.ncsf, self.ncsf)) # H_BO CSF matrix 
        self.dV_csf = np.zeros((self.nat, 3, self.ncsf, self.ncsf)) # nuclear gradient of H_BO CSF matrix
        self.D_csf = np.zeros((self.nat, 3, self.ncsf, self.ncsf)) # derivative coupling between CSFs

    def get_csf(self):
        pass

    def get_V_csf(self):
        
        self.h_cas, self.e_core = self.mc.get_h1cas() # get active 1e integral and core energy + Vnn
        self.v_cas = self.mc.get_h2cas() # get active 2e integral
        self.V_csf = self.mc.fcisolver.pspace(self.h_cas, self.v_cas, self.no_cas, self.ne_cas)[1] # get CSF Hamiltonian matrix
    
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
        pass

    def calculate_force(self):
        pass
    
    def get_int_mo(self):
        
        C = self.mo_coeff
        
        
        # Duplicative, since get_h1cas and get_h2cas calculate these.

        # 1e
        #self.h_mo = np.einsum('kl,kp,lq->pq', (self.v_en_ao + self.kin_e_ao), C, C)
        # 2e
        #self.v_mo = ao2mo.full(self.v_ee_ao, C)
        
        # d & g
        d_ao_tmp = np.zeros((3, self.norb, self.norb))
        g_ao_tmp = np.zeros((self.norb, self.norb))
        dS_tmp = np.zeros((3, self.norb, self.norb)) # \nabla_\nu S 
        d2S_tmp = np.zeros((self.norb, self.norb)) # \nabla_\nu^2 S

        S_inv = scipy.linalg.inv(self.s_ao)
        ao_slice = self.mol.aoslice_by_atom()[:, 2:4] # start-AO-id and stop-AO-id for each atom. (0,1 entries are start/stop-"shell"-ids)
        
        # d
        for iat in range(self.nat):
            start, stop = ao_slice[iat, 0], ao_slice[iat, 1]
            self.d_mo[iat, :, :, :] = 0.5 * np.einsum('kp,akl,lq->apq' ,\
                self.mo_coeff[:, :],\
                self.d_ao[:, :, start:stop],\
                self.mo_coeff[start:stop, :])
        self.d_mo = self.d_mo - self.d_mo.transpose((0, 1, 3, 2))
        
        # g    
        #self.d_dot_d_mo[:, :, :, :, :] = np.einsum('acpr,acqs->aprqs', self.d_mo, self.d_mo) # pyscf follows chemist's notation
        #tmp_dense = np.zeros((self.norb, self.norb))
        #for iat in range(self.nat):
        #    start, stop = ao_slice[iat, 0], ao_slice[iat, 1]
        #    
        #    tmp_dense[:, :] = 0.
        #    tmp_dense[:, start:stop] = (-2.0 * self.kin_e_ao[:, start:stop]) # g_ao
        #    
        #    d2S_tmp[:, :] = 0.
        #    d2S_tmp[:, start:stop] = (-2.0 * self.kin_e_ao[:, start:stop]) 
        #    d2S_tmp[start:stop, :] += (-2.0 * self.kin_e_ao[:, start:stop].T) 
        #    d2S_tmp[start:stop, start:stop] = 0.0 # already zero but for stability
        #    tmp_dense += -0.5 * d2S_tmp # \nabla^2 S

        #    tmp_dense[:, :] -= 0.25 * np.einsum('ckl,lm,cnm->kn', self.d_ao[:, :, start:stop], S_inv[start:stop, start:stop], self.d_ao[:, :, start:stop])
        #    tmp_dense[:, start:stop] -= 0.25 * np.einsum('ckl,lm,cmn->kn', self.d_ao[:, :, start:stop], S_inv[start:stop, :], self.d_ao[:, :, start:stop])
        #    tmp_dense[start:stop, :] += 0.75 * np.einsum('clk,lm,cnm->kn', self.d_ao[:, :, start:stop], S_inv[:, start:stop], self.d_ao[:, :, start:stop])
        #    tmp_dense[start:stop, start:stop] += 0.75 * np.einsum('clk,lm,cmn->kn', self.d_ao[:, :, start:stop], S_inv[:, :], self.d_ao[:, :, start:stop])

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

    def get_int_ao(self):

        self.s_ao = self.mol.intor('int1e_ovlp') # AO overlap (nao, nao)
        self.kin_e_ao = self.mol.intor('int1e_kin') # AO 1e kinetic energy integral
        self.v_en_ao = self.mol.intor('int1e_nuc') # AO 1e e-n couloumb integral
        self.v_ee_ao = self.mol.intor('int2e', aosym='s1') # AO 2e e-e coulomb integral
        self.d_ao = -1.0 * mol.intor('int1e_ipovlp').transpose((0,2,1)) # (3, nao, nao)  <\chi_l | \nabla_e \chi_m> --> need to convert to d/dR_nuc which requires ao_slice for indices
        self.dv_ee_ao = mol.intor('int2e_ip1') # (3, nao, nao, nao, nao) (\nabla k,l|m,n)  --> need to convert to d/dR_nuc
        #self.dv_en_ao = ????
        #self.dkin_e_ao = ????
   

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

