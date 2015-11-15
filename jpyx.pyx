import numpy as np
cimport numpy as np

import jchem

cdef extern from "jc.h":
	int prt()
	int prt_str( char*)

def prt_c():
	prt()

def prt_str_c( str):
	prt_str( str)

def calc_corr( smiles_l, radius = 6, nBits = 4096):
	"""
	It emulate calc_corr in jchem using cython.
	"""
	xM = jchem.get_xM( smiles_l, radius = radius, nBits = nBits)
	A = calc_tm_sim_M( xM)

	return A

def calc_bin_sim_M( np.ndarray[np.long_t, ndim=2] xM, gamma = 1.0):
	"""
	Ex. A = calc_tm_sim_M( np.array( xM, dtype = long)
	"""

	cdef int ln = xM.shape[0]
	cdef int lm = xM.shape[1]
	cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
	cdef int ix, iy, ii
	cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
	cdef int a_ix = 0
	cdef int c
	
	for ix in range( ln):
		a_ix = 0
		for ii in range( lm):
			a_ix += xM[ix, ii]
		#print ix, a_ix
		a[ix] = a_ix
	
	for ix in range( ln):
		for iy in range( ln):
			c = 0
			for ii in range( lm):
				c += xM[ix, ii] & xM[iy, ii]   
				
			if a[ix] == 0 and a[iy] == 0:
				A[ ix, iy] = 0.0
			else:
				A[ ix, iy] = gamma * float( c) / ( gamma * float( c) + a[ix] + a[iy] - 2*c)
			
	return A

def calc_tm_sim_M( np.ndarray[np.long_t, ndim=2] xM):
	"""
	Ex. A = calc_tm_sim_M( np.array( xM, dtype = long)
	"""

	cdef int ln = xM.shape[0]
	cdef int lm = xM.shape[1]
	cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
	cdef int ix, iy, ii
	cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
	cdef int a_ix = 0
	cdef int c, d
	
	for ix in range( ln):
		a_ix = 0
		for ii in range( lm):
			a_ix += xM[ix, ii]
		#print ix, a_ix
		a[ix] = a_ix
	
	for ix in range( ln):
		for iy in range( ln):
			c = 0
			for ii in range( lm):
				c += xM[ix, ii] & xM[iy, ii]   
				
			if a[ix] == 0 and a[iy] == 0:
				A[ ix, iy] = 0.0
			else:
				d = a[ix] + a[iy] - c
				A[ ix, iy] = float( c) / d
			
	return A

def calc_RBF( np.ndarray[np.long_t, ndim=2] xM, float epsilon):
	"""
	calculate Radial basis function for xM with epsilon
	in terms of direct distance
	where distance is the number of different bits between the two vectors.
	"""
	d = calc_atm_dist_M( xM)
	return RBF(d, epsilon) 

def calc_rRBF( np.ndarray[np.long_t, ndim=2] xM, float epsilon):
	"""
	calculate Radial basis function for xM with epsilon
	in terms of relative distance
	where distance is the number of different bits between the two vectors.
	"""
	rd = calc_atm_rdist_M( xM)
	return RBF(rd, epsilon) 


def RBF( d, e = 1):
	return np.exp( - e*np.power(d,2))

def calc_atm_dist_M( np.ndarray[np.long_t, ndim=2] xM):
	"""
	Ex. A = calc_tm_sim_M( np.array( xM, dtype = long)
	"""

	cdef int ln = xM.shape[0]
	cdef int lm = xM.shape[1]
	cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
	cdef int ix, iy, ii
	cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
	cdef int a_ix = 0
	cdef int c, d
	
	for ix in range( ln):
		a_ix = 0
		for ii in range( lm):
			a_ix += xM[ix, ii]
		#print ix, a_ix
		a[ix] = a_ix
	
	for ix in range( ln):
		for iy in range( ln):
			c = 0
			for ii in range( lm):
				c += xM[ix, ii] & xM[iy, ii]   
				
			if a[ix] == 0 and a[iy] == 0:
				A[ ix, iy] = 0.0
			else:
				d = a[ix] + a[iy] - 2*c
				A[ ix, iy] = d
			
	return A

def calc_atm_rdist_M( np.ndarray[np.long_t, ndim=2] xM):
	"""
	Ex. A = calc_tm_sim_M( np.array( xM, dtype = long)
	Relative Tanimoto distance is calculated
	"""

	cdef int ln = xM.shape[0]
	cdef int lm = xM.shape[1]
	cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
	cdef int ix, iy, ii
	cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
	cdef int a_ix = 0
	cdef int c, d
	
	for ix in range( ln):
		a_ix = 0
		for ii in range( lm):
			a_ix += xM[ix, ii]
		#print ix, a_ix
		a[ix] = a_ix
	
	for ix in range( ln):
		for iy in range( ln):
			c = 0
			for ii in range( lm):
				c += xM[ix, ii] & xM[iy, ii]   
				
			if a[ix] == 0 and a[iy] == 0:
				A[ ix, iy] = 0.0
			else:
				d = a[ix] + a[iy] - c # A or B
				A[ ix, iy] = (float) (d-c) / d # A or B - A and B / A or B (relative distance)
			
	return A



def calc_tm_dist_M( np.ndarray[np.long_t, ndim=2] xM):
	"""
	Ex. A = calc_tm_sim_M( np.array( xM, dtype = long)
	"""

	cdef int ln = xM.shape[0]
	cdef int lm = xM.shape[1]
	cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
	cdef int ix, iy, ii
	cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
	cdef int a_ix = 0
	cdef int c, d
	
	for ix in range( ln):
		a_ix = 0
		for ii in range( lm):
			a_ix += xM[ix, ii]
		#print ix, a_ix
		a[ix] = a_ix
	
	for ix in range( ln):
		for iy in range( ln):
			c = 0
			for ii in range( lm):
				c += xM[ix, ii] & xM[iy, ii]   
				
			if a[ix] == 0 and a[iy] == 0:
				A[ ix, iy] = 0.0
			else:
				d = a[ix] + a[iy] - c
				A[ ix, iy] = float( d - c) / d
			
	return A

def _calc_ec_sim_M_r0( np.ndarray[np.long_t, ndim=2] xM):
	"""
	Euclidean-tanimoto distance
	"""

	cdef int ln = xM.shape[0]
	cdef int lm = xM.shape[1]
	cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
	cdef int ix, iy, ii
	cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
	cdef int a_ix = 0
	cdef int c, d
	
	for ix in range( ln):
		a_ix = 0
		for ii in range( lm):
			a_ix += xM[ix, ii]
		#print ix, a_ix
		a[ix] = a_ix
	
	for ix in range( ln):
		for iy in range( ln):
			c = 0
			for ii in range( lm):
				c += xM[ix, ii] & xM[iy, ii]   
				
			if a[ix] == 0 and a[iy] == 0:
				A[ ix, iy] = 0.0
			else:
				d = a[ix] + a[iy] - c
				A[ ix, iy] = float(lm - d + c) / lm
			
	return A        

def _calc_ec_dist_M_r0( np.ndarray[np.long_t, ndim=2] xM):
	"""
	Euclidean-tanimoto distance
	"""

	cdef int ln = xM.shape[0]
	cdef int lm = xM.shape[1]
	cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
	cdef int ix, iy, ii
	cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
	cdef int a_ix = 0
	cdef int c, d
	
	for ix in range( ln):
		a_ix = 0
		for ii in range( lm):
			a_ix += xM[ix, ii]
		#print ix, a_ix
		a[ix] = a_ix
	
	for ix in range( ln):
		for iy in range( ln):
			c = 0
			for ii in range( lm):
				c += xM[ix, ii] & xM[iy, ii]   
				
			if a[ix] == 0 and a[iy] == 0:
				A[ ix, iy] = 0.0
			else:
				d = a[ix] + a[iy] - c
				A[ ix, iy] = float( d - c) / lm
			
	return A    

def calc_ec_sim_M( np.ndarray[np.long_t, ndim=2] xM):
	"""
	Euclidean-tanimoto distance
	"""

	cdef int ln = xM.shape[0]
	cdef int lm = xM.shape[1]
	cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln), dtype = float)
	cdef int ix, iy, ii
	cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
	cdef int a_ix = 0
	cdef int c
	
	for ix in range( ln):
		a_ix = 0
		for ii in range( lm):
			a_ix += xM[ix, ii]
		#print ix, a_ix
		a[ix] = a_ix
	
	for ix in range( ln):
		for iy in range( ln):
			c = 0
			for ii in range( lm):
				c += xM[ix, ii] ^ xM[iy, ii]   
				
			if a[ix] == 0 and a[iy] == 0:
				A[ ix, iy] = 1.0
			else:
				#d = a[ix] + a[iy] - c
				A[ ix, iy] = float( lm - c) / lm
			
	return A    

def calc_ec_dist_M( np.ndarray[np.long_t, ndim=2] xM):
	"""
	Euclidean-tanimoto distance
	"""

	cdef int ln = xM.shape[0]
	cdef int lm = xM.shape[1]
	cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
	cdef int ix, iy, ii
	cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
	cdef int a_ix = 0
	cdef int c
	
	for ix in range( ln):
		a_ix = 0
		for ii in range( lm):
			a_ix += xM[ix, ii]
		#print ix, a_ix
		a[ix] = a_ix
	
	for ix in range( ln):
		for iy in range( ln):
			c = 0
			for ii in range( lm):
				c += xM[ix, ii] ^ xM[iy, ii]   
				
			if a[ix] == 0 and a[iy] == 0:
				A[ ix, iy] = 0.0
			else:
				#d = a[ix] + a[iy] - c
				A[ ix, iy] = float( c) / lm
			
	return A    

def bcalc_tm_sim_vec(int a, int b, int ln):
	cdef int ii
	cdef int a_and_b = a & b
	cdef int a_or_b = a | b
	cdef int a_and_b_sum = 0
	cdef int a_or_b_sum = 0
	
	for ii in range(ln):
		a_and_b_sum += a_and_b & 1
		a_and_b = a_and_b >> 1
		a_or_b_sum += a_or_b & 1
		a_or_b = a_or_b >> 1
	return float(a_and_b_sum) / float(a_or_b_sum)

def calc_tm_sim_vec(np.ndarray[np.long_t, ndim=1] a, np.ndarray[np.long_t, ndim=1] b):
	cdef int ii
	cdef int a_and_b_sum = 0
	cdef int a_or_b_sum = 0
	cdef int ln = a.shape[0]
	
	for ii in range( ln):
		a_and_b_sum += a[ii] & b[ii]
		a_or_b_sum += a[ii] | b[ii]
	return float(a_and_b_sum) / float(a_or_b_sum)

