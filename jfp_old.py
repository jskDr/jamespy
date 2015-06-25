"""
Finger print related codes are colected.
"""
import jchem

def find_cluster( fa_list, thr = 0.5):
    """
    find similar pattern with 
    the first element: fa0
    """
    fa0 = fa_list[0]
    fa0_group = [fa0]
    fa_other = fa_list[1:]
    for fa_o in fa_list[1:]:
    tm_d = jchem.calc_tm_dist_int( fa0, fa_o)
    if tm_d > thr:
    fa0_group.append( fa_o)
    fa_other.remove( fa_o)

    return fa0_group, fa_other


def find_cluster_all( fa_list, thr = 0.5):
    """
    all cluster are founded based on threshold of 
    fingerprint similarity using greedy methods
    """
    fa_o = fa_list
    fa0_g_all = []
    
    while len( fa_o) > 0:
        fa0_g, fa_o = find_cluster( fa_o, thr)
        fa0_g_all.append( fa0_g)
    
    return fa0_g_all

def _calc_tm_sim_int_r0( A_int, B_int):
    """
    Calculate tanimoto distance of A_int and B_int
    where X_int isinteger fingerprint vlaue of material A.
    """
    C_int = A_int & B_int

    A_str = bin(A_int)[2:]
    B_str = bin(B_int)[2:]
    C_str = bin(C_int)[2:]

    lmax = max( [len( A_str), len( B_str), len( C_str)])

    """ this shows calculation process 
    print "A:", A_str.ljust( lmax, '0')
    print "B:", B_str.ljust( lmax, '0')
    print "C:", C_str.ljust( lmax, '0')
    """

    a = A_str.count('1')
    b = B_str.count('1')
    c = C_str.count('1')

    # print a, b, c
    if a == 0 and b == 0:
        tm_dist = 1
    else:
        tm_dist = float(c) / float( a + b - c)

    return tm_dist

def _calc_tm_sim_int_r0( A_int, B_int):
    """
    Calculate tanimoto distance of A_int and B_int
    where X_int isinteger fingerprint vlaue of material A.
    """
    C_int = A_int & B_int
    
    A_str = bin(A_int)[2:]
    B_str = bin(B_int)[2:]
    C_str = bin(C_int)[2:]

    lmax = max( [len( A_str), len( B_str), len( C_str)])

    """ this shows calculation process 
    print "A:", A_str.ljust( lmax, '0')
    print "B:", B_str.ljust( lmax, '0')
    print "C:", C_str.ljust( lmax, '0')
    """

    a = A_str.count('1')
    b = B_str.count('1')
    c = C_str.count('1')

    # print a, b, c
    if a == 0 and b == 0:
        tm_dist = 1
    else:
        tm_dist = float(c) / float( a + b - c)

    return tm_dist


def calc_tm_sim( A_smiles, B_smiles):

    A_int = jchem.ff_int( A_smiles)
    B_int = jchem.ff_int( B_smiles)

    return calc_tm_sim_int( A_int, B_int)