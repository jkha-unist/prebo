def hcore_at_atm(mol,atm_id):
  '''
  mol: pyscf.gto.mole.Mole
  PySCF molecular geometry object.

  atm_id:
  Atom w.r.t which the gradient is to be evaluated. Follows the atom ordering of mol, starting at 0.
  e.g.: for 'Li 0 0 0; H 0 0 3.2' atm_id for Li-> 0, for H-> 1.

  returns: numpy.array
  Sum of the three AO contributions to the h_core integral gradient w.r.t to atom atm_id. Still need to be multiplied by MO coefficients 
  and do not include contributions from the gradient of the coefficients.
  '''
  #This import is needed because if you do mf.get_hcore it can return an array with incorrect shape
  from pyscf.grad.rhf import get_hcore  

  aoslices = mol.aoslice_by_atom()
  h1 = get_hcore(mol)
  shl0, shl1, p0, p1 = aoslices[atm_id]
  with mol.with_rinv_at_nucleus(atm_id):
      vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
      vrinv *= -mol.atom_charge(atm_id)
      vrinv[:,p0:p1] += h1[:,p0:p1]

  return vrinv + vrinv.transpose(0,2,1)

  def mo_grad_one(mol,mf,atm_id):
  '''
  mol: pyscf.gto.mole.Mole
  PySCF molecular geometry object.

  atm_id:
  Atom w.r.t which the gradient is to be evaluated. Follows the atom ordering of mol, starting at 0.
  e.g.: for 'Li 0 0 0; H 0 0 3.2' atm_id for Li

  mf: pyscf.scf
  PySCF HF calculation

  return: NumPy.array
  '''
  dAO_dv = hcore_at_atm(mol,atm_id)        # derivative of the AO w.r.t nuclear position atm_id
  dMO_dv = np.einsum('pi,pq,qj->ij', mf.mo_coeff, dAO_dv, mf.mo_coeff ) 

  return dMO_dv
