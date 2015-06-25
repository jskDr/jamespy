import pandas as pd
import numpy as np
import re #regular expression

from rdkit import Chem


import jchem

def gen_14BQ_OH():
    """
    return 1,4BQ species with OH functionals.
    """
    q_smiles_base = {}
    q_smiles_mid = {}
    q_smiles_base['1,4-BQ,2-OH'] = '[H]OC1=C([H])C(=O)C([H])=C([H])C1=O'
    q_smiles_base['1,4-BQ,Full-OH'] = 'OC1=C(O)C(=O)C(O)=C(O)C1=O'
    q_smiles_base['1,4-BQ'] = 'O=C1C=CC(=O)C=C1'

    q_smiles_mid['1,4-BQ'] = 'O=C1C=CC(=O)C=C1'
    q_smiles_mid['1,4-BQ,2-OH'] = 'OC1=CC(=O)C=CC1=O'
    q_smiles_mid['1,4-BQ,2,3-OH'] = 'OC1=C(O)C(=O)C=CC1=O'
    q_smiles_mid['1,4-BQ,2,3,5-OH'] = 'OC1=CC(=O)C(O)=C(O)C1=O'
    q_smiles_mid['1,4-BQ,Full-OH'] = 'OC1=C(O)C(=O)C(O)=C(O)C1=O'    

    return q_smiles_base, q_smiles_mid

def gen_910AQ_SO3H():
    """
    return 9,10AQ species with SO3H functionals.
    """
    q_smiles_base = {}
    q_smiles_mid = {}

    q_smiles_base['9,10AQ'] = 'O=C1C2C=CC=CC2C(=O)C2=C1C=CC=C2'
    q_smiles_base['9,10AQ,1-OH'] = 'OS(=O)(=O)C1=CC=CC2C1C(=O)C1=C(C=CC=C1)C2=O'
    q_smiles_base['9,10AQ,2-OH'] = 'OS(=O)(=O)C1=CC2C(C=C1)C(=O)C1=C(C=CC=C1)C2=O'
    q_smiles_base['9,10AQ,Full-OH'] = 'OS(=O)(=O)C1=C(C(=C(C2C1C(=O)C1=C(C2=O)C(=C(C(=C1S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O'

    q_smiles_mid['9,10AQ'] = 'O=C1C2C=CC=CC2C(=O)C2=C1C=CC=C2'
    q_smiles_mid['9,10AQ,1-OH'] = 'OS(=O)(=O)C1=CC=CC2C1C(=O)C1=C(C=CC=C1)C2=O'
    q_smiles_mid['9,10AQ,2-OH'] = 'OS(=O)(=O)C1=CC2C(C=C1)C(=O)C1=C(C=CC=C1)C2=O'
    q_smiles_mid['9,10AQ,1,2-OH'] = 'OS(=O)(=O)C1=C(C2C(C=C1)C(=O)C1=C(C=CC=C1)C2=O)S(O)(=O)=O'
    q_smiles_mid['9,10AQ,Full-OH'] = 'OS(=O)(=O)C1=C(C(=C(C2C1C(=O)C1=C(C2=O)C(=C(C(=C1S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O)S(O)(=O)=O'

    return q_smiles_base, q_smiles_mid

def gen_smiles_quinone( quinone = '9,10AQ', r_group = 'SO3H'):
    if quinone == '1,4BQ' and r_group == 'OH':
        return gen_14BQ_OH()
    elif quinone == '9,10AQ' and r_group == 'SO3H':
        return gen_910AQ_SO3H()


class AQDS_OH():
    def __init__(self, fname = 'oh_subs_csv.csv'):
        self.pdr = pd.read_csv( fname)

    def get_Frag6_D( self, dfr = None):

        if dfr == None:
            dfr = self.pdr

        ri_vec = [1, 3, 4, 5, 6, 8]
        R = []
        HOH201 = {'H': 0, 'OH': 1}

        for ri in ri_vec:
            s = 'R{}'.format( ri)
            rv = dfr[s].tolist()
            rv_01 = map( lambda x: HOH201[x], rv)
            R.append( rv_01)
        RM = np.mat( R).T

        Frag6_L = []   
        OHSYMB = {'H': '', 'OH': '(O)'}
        for ri in ri_vec:
            s = 'R{}'.format( ri)
            rv = dfr[s].tolist()
            fr_01 = map( lambda x: OHSYMB[x], rv)
            Frag6_L.append( fr_01)
        #print Frag6_L

        Frag6_D = []
        for ii in range( len(Frag6_L[0])):
            Frag6_D.append({})
            
        for ii, frag in enumerate(Frag6_L):
            for ix, fr in enumerate(frag):
                dict_key = '{B%d}' % ii
                Frag6_D[ix][dict_key] = fr

        return Frag6_D

    def _gen_27aqds_with_oh_r1( self, Frag6_D, show = True):
        """
        2,7-AQDS with OH attachment are performed 
        using smiles interpolation
        """
        mol_smiles_list = []
        for ix, mol_symb in enumerate(Frag6_D):
            mol = bq14_oh2 = Chem.MolFromSmiles( 'C1(=O)c2c{B3}c{B4}c(S(=O)(=O)O)c{B5}c2C(=O)c2c{B0}c(S(=O)(=O)O)c{B1}c{B2}c21', replacements=mol_symb)
            mol_smiles = Chem.MolToSmiles( mol)
            mol_smiles_list.append( mol_smiles)

            if show:
                print ix+1, mol_smiles
                jchem.show_mol( mol_smiles)

        return mol_smiles_list

    def gen_27aqds_with_R( self, Frag6_D, r_gr, show = True):
        """
        2,7-AQDS with OH attachment are performed 
        using smiles interpolation
        """
        mol_smiles_list = []
        for ix, mol_symb in enumerate(Frag6_D):
            # r_gr = 'S(=O)(=O)O' #[N+]([O-])=O
            base_smiles = 'C1(=O)c2c{B3}c{B4}c(%s)c{B5}c2C(=O)c2c{B0}c(%s)c{B1}c{B2}c21' % (r_gr, r_gr)
            mol = bq14_oh2 = Chem.MolFromSmiles( base_smiles, replacements=mol_symb)
            mol_smiles = Chem.MolToSmiles( mol)
            mol_smiles_list.append( mol_smiles)

            if show:
                print ix+1, mol_smiles
                jchem.show_mol( mol_smiles)

        return mol_smiles_list

    def gen_27aqds_with_oh( self, Frag6_D, show = True):
        r_gr = 'S(=O)(=O)O' 
        return self.gen_27aqds_with_R( Frag6_D, r_gr, show = show)


    def gen_27aqds_with_no2( self, Frag6_D, show = True):
        r_gr = '[N+]([O-])=O'
        return self.gen_27aqds_with_R( Frag6_D, r_gr, show = show)

class HAQDS_OH():
    def __init__(self, fname = 'oh_subs_csv.csv'):
        self.pdr = pd.read_csv( fname)

    def get_Frag6_D( self, dfr = None):

        if dfr == None:
            dfr = self.pdr

        ri_vec = [1, 3, 4, 5, 6, 8]
        R = []
        HOH201 = {'H': 0, 'OH': 1}

        for ri in ri_vec:
            s = 'R{}'.format( ri)
            rv = dfr[s].tolist()
            rv_01 = map( lambda x: HOH201[x], rv)
            R.append( rv_01)
        RM = np.mat( R).T

        Frag6_L = []   
        OHSYMB = {'H': '', 'OH': '(O)'}
        for ri in ri_vec:
            s = 'R{}'.format( ri)
            rv = dfr[s].tolist()
            fr_01 = map( lambda x: OHSYMB[x], rv)
            Frag6_L.append( fr_01)
        #print Frag6_L

        Frag6_D = []
        for ii in range( len(Frag6_L[0])):
            Frag6_D.append({})
            
        for ii, frag in enumerate(Frag6_L):
            for ix, fr in enumerate(frag):
                dict_key = '{B%d}' % ii
                Frag6_D[ix][dict_key] = fr

        return Frag6_D

    def _gen_27aqds_with_oh_r1( self, Frag6_D, show = True):
        """
        2,7-AQDS with OH attachment are performed 
        using smiles interpolation
        """
        mol_smiles_list = []
        for ix, mol_symb in enumerate(Frag6_D):
            mol = bq14_oh2 = Chem.MolFromSmiles( 'C1(O)c2c{B3}c{B4}c(S(=O)(=O)O)c{B5}c2C(O)c2c{B0}c(S(=O)(=O)O)c{B1}c{B2}c21', replacements=mol_symb)
            mol_smiles = Chem.MolToSmiles( mol)
            mol_smiles_list.append( mol_smiles)

            if show:
                print ix+1, mol_smiles
                jchem.show_mol( mol_smiles)

        return mol_smiles_list

    def gen_27aqds_with_R( self, Frag6_D, r_gr, show = True):
        """
        2,7-AQDS with OH attachment are performed 
        using smiles interpolation
        """
        mol_smiles_list = []
        for ix, mol_symb in enumerate(Frag6_D):
            # r_gr = 'S(=O)(=O)O' #[N+]([O-])=O
            base_smiles = 'C1(O)c2c{B3}c{B4}c(%s)c{B5}c2C(O)c2c{B0}c(%s)c{B1}c{B2}c21' % (r_gr, r_gr)
            mol = Chem.MolFromSmiles( base_smiles, replacements=mol_symb)
            mol_smiles = Chem.MolToSmiles( mol)
            mol_smiles_list.append( mol_smiles)

            if show:
                print ix+1, mol_smiles
                jchem.show_mol( mol_smiles)

        return mol_smiles_list

    def gen_27aqds_with_oh( self, Frag6_D, show = True):
        r_gr = 'S(=O)(=O)O' 
        return self.gen_27aqds_with_R( Frag6_D, r_gr, show = show)


    def gen_27aqds_with_no2( self, Frag6_D, show = True):
        r_gr = '[N+]([O-])=O'
        return self.gen_27aqds_with_R( Frag6_D, r_gr, show = show)        

def get_r_list( N_Rgroup = 4, so3h = '(S(O)(=O)=O)', disp = False, pdForm = True):
    pdr_id, pdr_index, pdr_rgroups, pdr_no_r = [], [], [], []
    
    N_max_bin = '0b' + '1' * N_Rgroup
    for pos in range( int(N_max_bin, 2) + 1):
        pos_bin = bin( pos)[2:].rjust( N_Rgroup, '0')
        so_int_l = [int(x) for x in pos_bin]
        so_l = [so3h if x == 1 else '' for x in so_int_l ]
        no_r = sum( so_int_l)
        
        pdr_id.append( pos + 1)
        pdr_no_r.append( no_r)
        pdr_index.append( so_int_l)
        pdr_rgroups.append( so_l)
        
        if disp: print pos, no_r, so_int_l, '==>', so_l
        
    if pdForm:
        pdr = pd.DataFrame()
        pdr['ID'] = pdr_id
        pdr['Rgroup'] = [so3h] * len( pdr_id)   
        pdr['NoOfR'] = pdr_no_r
        pdr['Index'] = pdr_index
        pdr['Rgroups'] = pdr_rgroups
        return pdr
    else:
        return so_l

def gen_r_attach( mol = 'Oc1nc(O)c2nc3c{0}c{1}c{2}c{3}c3nc2n1', so3h = '(S(O)(=O)=O)', disp = False, graph = False):
    """
    generate molecules with R group fragment
    """
    N_group = len( re.findall( '{[0-9]*}', mol)) # find number of R group positions

    pdr = get_r_list( N_group, so3h, disp = disp, pdForm = True)
    so_l = pdr['Rgroups'].tolist()

    aso_l = []
    for so in so_l:        
        aso = mol.format(*so)
        aso_l.append( aso)
        if disp: print so, aso
        if graph: jchem.show_mol( aso)

    pdr['SMILES'] = aso_l
    pdr['BaseMol'] = [aso_l[0]] * len( aso_l)
    pdr['BaseStr'] = [mol] * len( aso_l)

    return pdr

def gen_r_attach_Alloxazine_R123457( so3h = '(S(O)(=O)=O)', disp = False, graph = False):
    """
    generate molecules with R group fragment
    """

    # n1{R5}c2nc3c{R1}c{R2}c{R3}c{R4}c3nc2c(=O)n{R7}c1=O
    # 
    N_group = 6 #R1234 5 7 -> 0123 4 5

    pdr = get_r_list( N_group, so3h, disp = disp, pdForm = True)
    so_l = pdr['Rgroups'].tolist()

    aso_l = []
    mol_l = []
    for so in so_l:        
        if so[4] != '' and so[5] != '':
            aso = 'n1{4}c2nc3c{0}c{1}c{2}c{3}c3nc2c(=O)n{5}c1=O'.format(*so)
            mol_l.append('n1{4}c2nc3c{0}c{1}c{2}c{3}c3nc2c(=O)n{5}c1=O')
        elif so[4] == '' and so[5] == '':
            aso = '[nH]1c2nc3c{0}c{1}c{2}c{3}c3nc2c(=O)[nH]c1=O'.format(*so[:4])
            mol_l.append('[nH]1c2nc3c{0}c{1}c{2}c{3}c3nc2c(=O)[nH]c1=O')
        elif so[4] == '':
            aso = '[nH]1c2nc3c{0}c{1}c{2}c{3}c3nc2c(=O)n{4}c1=O'.format(so[0],so[1],so[2],so[3], so[5])
            mol_l.append('[nH]1c2nc3c{0}c{1}c{2}c{3}c3nc2c(=O)n{4}c1=O')
        else: #so[5] == '':
            aso = 'n1{4}c2nc3c{0}c{1}c{2}c{3}c3nc2c(=O)[nH]c1=O'.format(*so[:5])
            mol_l.append('n1{4}c2nc3c{0}c{1}c{2}c{3}c3nc2c(=O)[nH]c1=O')

        aso_l.append( aso)
        if disp: print so, aso
        if graph: jchem.show_mol( aso)

    pdr['SMILES'] = aso_l
    pdr['BaseMol'] = [aso_l[0]] * len( aso_l)
    pdr['BaseStr'] = mol_l

    return pdr

def gen_r_attach_Flavins( mol = 'n1c2[nH]{5}c3c{4}c{3}c{2}c{1}c3nc2c(=O)[nH]{0}c1=O', so3h = '(S(O)(=O)=O)', disp = False, graph = False):
    # jchem.show_mol( 'n1c2[nH]{5}c3c{4}c{3}c{2}c{1}c3nc2c(=O)[nH]{0}c1=O'.format( so, so, so, so, so, so))
    return gen_r_attach( mol = mol, so3h = so3h, disp = disp, graph = graph)

# [nH]1c2[nH]c3ccccc3[nH]c2c(=O)[nH]c1=O
def gen_r_attach_Flavins_nH( mol = '[nH]1c2[nH]{5}c3c{4}c{3}c{2}c{1}c3[nH]c2c(=O)[nH]{0}c1=O', so3h = '(S(O)(=O)=O)', disp = False, graph = False):
    # jchem.show_mol( 'n1c2[nH]{5}c3c{4}c{3}c{2}c{1}c3nc2c(=O)[nH]{0}c1=O'.format( so, so, so, so, so, so))
    return gen_r_attach( mol = mol, so3h = so3h, disp = disp, graph = graph)


def gen_r_attach_Alloxazine( mol = '[nH]1{5}c2nc3c{4}c{3}c{2}c{1}c3nc2c(=O)[nH]{0}c1=O', so3h = '(S(O)(=O)=O)', disp = False, graph = False):
    # '[nH]1{5}c2nc3c{4}c{3}c{2}c{1}c3nc2c(=O)[nH]{0}c1=O'
    return gen_r_attach( mol = mol, so3h = so3h, disp = disp, graph = graph)

def gen_r_attach_lowpot_Flavins( disp = False, graph = False):
    oh = '(O)'
    h = ''
    oc = '(OC)'

    rl = []
    rl.append(([h,oh, oh, oh, h, h], -0.47))
    rl.append(([oh, oh, h,h,h,h], -0.47))
    rl.append(([oh, oh, oh, oh, oh, h], -0.47))
    rl.append(([oh, oh, oh, oh, h, h], -0.51))
    rl.append(([h, oh, h, oh, h, h], -0.50))
    rl.append(([h, oh, h, h, h, h], -0.45))
    rl.append(([oh, oh, h, oh, oh, h], -0.50))
    rl.append(([h, oh, h, oh, oh, h], -0.46))
    rl.append(([oh, oh, h, oh, h, h], -0.53))

    rl.append(([h, oc, oc, oc, h, h], -0.48))
    rl.append(([oc, oc, oc, oc, h, h], -0.48))
    rl.append(([oc, oc, h, oc, h, h], -0.47))
    rl.append(([h, oc, h, oc, oc, h], -0.46))
    rl.append(([oc, oc, h, oc, oc, h], -0.50))

    BaseStr = 'n1c2[nH]{5}c3c{4}c{3}c{2}c{1}c3nc2c(=O)[nH]{0}c1=O'
    N_group = len( re.findall( '{[0-9]*}', BaseStr))
    emptyR = [''] * N_group
    BaseMol = BaseStr.format( *emptyR)

    smiles_l = [ BaseStr.format(*r[0]) for r in rl]

    pdr = pd.DataFrame()
    pdr['ID'] = range( 1, len( smiles_l) + 1)

    R_group_l = []
    Index_l = []
    NoOfR_l = []
    for r in rl:
        # Whether it is oh or oc family is determined
        r_oh_test = [ x == oh for x in r[0]]
        print r[0], '-->', r_oh_test, '-->', any(r_oh_test)

        if any(r_oh_test): 
            r_type = oh
        else:
            r_type = oc
        R_group_l.append( r_type)

        r_groups = [ 0 if x == '' else 1 for x in r[0]]
        Index_l.append( r_groups)
        NoOfR_l.append( np.sum( r_groups))

    pdr['Rgroup'] = R_group_l # This is newly included.
    pdr['NoOfR'] = NoOfR_l 
    pdr['Index'] = Index_l
    pdr['Rgroups'] = [ r[0] for r in rl]
    pdr['SMILES'] = smiles_l
    pdr['BaseMol'] = [BaseMol] * len(rl)
    pdr['BaseStr'] = [BaseStr] * len(rl)
    pdr['RedoxPotential'] = [ r[1] for r in rl]

    for ix, s in enumerate( smiles_l):
        if disp: print ix+1, s
        if graph:
            jchem.show_mol( s)

    return pdr