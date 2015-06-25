"""
RDKit related files are included.
"""
from jchem import *

def find( sm, sg, disp = False, graph = False):
	m = Chem.MolFromSmiles( sm)
	results = m.GetSubstructMatches( Chem.MolFromSmarts( sg))

	if graph:
		show_mol( sm)

	if disp: 
		print 'Results:', results
		print 'Canonical SMILES:', Chem.MolToSmiles( m)

	return results


def prt_sg( m, sg = '[C]', info = ''):
	matched_sg_l = m.GetSubstructMatches( Chem.MolFromSmarts( sg))
	print sg, ':', len(matched_sg_l), '-->', matched_sg_l, info

	return len(matched_sg_l)

def find_subgroups( sm = 'CNN', info = ''):
	"""
	return subgroups for various functional groups
	"""

	# sm = 'CNN' # Acetamide --> 1.58
	show_mol( sm)
	print 'SMILES: {},'.format(sm), info

	m = Chem.MolFromSmiles( sm)

	prt_sg( m, '[a]')
	prt_sg( m, '[A]')
	prt_sg( m, '[#6]')
	prt_sg( m, '[R1]')
	prt_sg( m, '[R2]')
	prt_sg( m, '[r5]')
	prt_sg( m, '[v4]')
	prt_sg( m, '[v3]')
	prt_sg( m, '[X2]')
	prt_sg( m, '[H]')
	prt_sg( m, '[H1]')
	prt_sg( m, '*')
	prt_sg( m, 'CC')
	prt_sg( m, '[#6]~[#6]') # any bond
	prt_sg( m, '[#6]@[#6]') # ring bond
	#prt_sg( m, 'F/?[#6]=C/[Cl]')
	prt_sg( m, 'F/[#6]=C/Cl')
	prt_sg( m, '[!c]')
	prt_sg( m, '[N,#8]')
	prt_sg( m, '[#7,C&+0,+1]')
	print '============================================'	


	sgn_d = dict()
	prt_sg( m, '[C]',   info = 'no subgroup')
	print ''

	prt_sg( m, '[CH3]',   info = 'no subgroup')
	sgn_d[1] = prt_sg( m,  '[*CH3]',  info = '#01, sol += -1.7081') 
	print ''

	prt_sg( m, '[CH2]',   info = 'no subgroup')	
	sgn_d[2] = prt_sg( m,  '[*CH2*]', info = '#02, sol += -0.4991')
	sgn_d[5] = prt_sg( m, '*=[CH2]', info = '#02, sol += -0.4991')
	print ''
	
	prt_sg( m, '[CH1]',   info = 'no subgroup')	
	sgn_d[3] = prt_sg( m,  '*[CH1]*(*)', info = '#02, sol += -0.4991')
	sgn_d[6] = prt_sg( m, '*=[CH1]*', info = '#02, sol += -0.4991')
	print ''

	prt_sg( m, '[CH0]',   info = 'no subgroup')	
	prt_sg( m, '*[CH0]',   info = 'no subgroup')	
	prt_sg( m, '[*CH0]',   info = 'no subgroup')	
	prt_sg( m, '[*CH0*](*)',   info = 'no subgroup')	
	prt_sg( m, '*=[CH0*](*)',   info = 'no subgroup')	
	sgn_d[4] = prt_sg( m,  '*[CH0]*(*)(*)', info = '#02, sol += -0.4991')
	sgn_d[7] = prt_sg( m, '*=[CH0]*(*)', info = '#02, sol += -0.4991')
	sgn_d[8] = prt_sg( m, '*=[CH0]=*', info = '#02, sol += -0.4991')
	sgn_d[9] = prt_sg( m,  '*[CH0]#[CH1]', info = '#02, sol += -0.4991')
	print ''
	
	sgn_d[8] = prt_sg( m, '[NH2]', info = '#49, sol += 0') 
	sgn_d[8] = prt_sg( m, '[NH]',  info = '#50, sol += 0') 
	sgn_d[8] = prt_sg( m, '[OH]') #26 (primary) 0.2711