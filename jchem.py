from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import DataStructs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This is James Sungjin Kim's library
import jutil

def show_mol( smiles = 'C1=CC=CC=C1'):
	"""
	This function shows the molecule defined by smiles code.
	The procedure follows:
	- 
	First, benzene can be defined as follows. 
    Before defining molecule, the basic library of rdkit can be loaded using the import command.

    Second, the 2D coordination of the molecule can be calculated. 
    For coordination calculation, AllChem sub-tool should be included.

	Third, the molecular graph is drawn and save it 
	so as to see in the picture manipulation tool. 
	To use Draw, we must include Draw tool from rdkit.Chem.

	Then,  it is time to load png file and show the image on screen.

	Input: smiles code
	"""
	m = Chem.MolFromSmiles( smiles)
	tmp = AllChem.Compute2DCoords( m)
	f_name = '{}.png'.format( 'smiles')
	Draw.MolToFile(m, f_name)

	img_m = plt.imread( f_name)
	plt.imshow( img_m)
	plt.show()

def calc_corr( smilesArr, radius = 2, nBits = 1024):
	ms_mid = [Chem.MolFromSmiles( m_sm) for m_sm in smilesArr]	
	f_m = [AllChem.GetMorganFingerprintAsBitVect(x, radius, nBits) for x in ms_mid]

	Nm = len(f_m)
	A = np.zeros( (Nm, Nm))

	for (m1, f1) in enumerate(f_m):
			for (m2, f2) in enumerate(f_m):
				# print( "Base:{0}, Target:{1}".format( ms_smiles_base.keys()[bx], ms_smiles_mid.keys()[mx]))
				A[m1, m2] =  DataStructs.DiceSimilarity( f1, f2)

	return A

def calc_corr_rad( smilesArr, radius = 2):
	ms_mid = [Chem.MolFromSmiles( m_sm) for m_sm in smilesArr]	
	f_m = [AllChem.GetMorganFingerprint(x, radius) for x in ms_mid]

	Nm = len(f_m)
	A = np.zeros( (Nm, Nm))

	for (m1, f1) in enumerate(f_m):
			for (m2, f2) in enumerate(f_m):
				# print( "Base:{0}, Target:{1}".format( ms_smiles_base.keys()[bx], ms_smiles_mid.keys()[mx]))
				A[m1, m2] =  DataStructs.DiceSimilarity( f1, f2)

	return A


class jfingerprt_circular():
	def __init__(self, radius = 2, nBits = 1024):
		self.radius = radius
		self.nBits = nBits

	def smiles_to_ff( self, smilesArr):
		"""
		smiles array will be transformed to fingerprint array
		"""
		ms_mid = [Chem.MolFromSmiles( m_sm) for m_sm in smilesArr]
		fps_mid = [AllChem.GetMorganFingerprintAsBitVect(x, self.radius, self.nBits) for x in ms_mid]
		return fps_mid

	def similarity( self, ms_smiles_mid, ms_smiles_base):
	    """
	    Input: dictionary type required such as {nick name: smiles code, ...}
	    """

	    """
	    # Processing for mid
	    print( "Target: {}".format( ms_smiles_mid.keys()))
	    fps_mid = self.smiles_to_ff( ms_smiles_mid.values())

	    #processing for base
	    print( "Base: {}".format( ms_smiles_base.keys()))
	    fps_base = self.smiles_to_ff( ms_smiles_base.values())
	    """

	    for idx in ["mid", "base"]:
	    	ms_smiles = eval( 'ms_smiles_{}'.format( idx))
	    	print( '{0}: {1}'.format( idx.upper(), ms_smiles.keys()))	    	
	    	exec( 'fps_{} = self.smiles_to_ff( ms_smiles.values())'.format( idx))

	    return fps_base, fps_mid	

	def return_similarity( self, ms_smiles_mid, ms_smiles_base, property_of_base = None):
		fps_base, fps_mid = self.similarity( ms_smiles_mid, ms_smiles_base)

		Nb, Nm = len(fps_base), len(fps_mid)
		A = np.zeros( (Nm, Nb))
		b = np.zeros( Nb)

		for (bx, f_b) in enumerate(fps_base):
			for (mx, f_m) in enumerate(fps_mid):
				# print( "Base:{0}, Target:{1}".format( ms_smiles_base.keys()[bx], ms_smiles_mid.keys()[mx]))
				A[mx, bx] =  DataStructs.DiceSimilarity( f_b, f_m)
				# print( A[mx, bx])
			if property_of_base:
				b[ bx] = property_of_base[ bx]
				# print( b[ bx])

		if property_of_base:
			# print "b is obtained."
			return A, b
		else:
			return A

	def get_w( self, ms_smiles_mid, ms_smiles_base, property_of_base):
		"""
		property_of_base, which is b, must be entered
		"""
		[A, b] = self.return_similarity( ms_smiles_mid, ms_smiles_base, property_of_base)
		w = np.dot( np.linalg.pinv(A), b)

		return w

	def get_w_full( self, ms_smiles_mid, ms_smiles_base, property_of_base):
		"""
		property_of_base, which is b, must be entered
		"""
		[A, b] = self.return_similarity( ms_smiles_mid, ms_smiles_base, property_of_base)
		B = A.transpose()
		w_full = np.dot( np.linalg.pinv(B), b)

		return w_full

def clean_smiles_vec( sv):
	"It removes bad smiles code elements in smiles code vector."
	new_sv = []
	for x in sv:
		y = Chem.MolFromSmiles(x)
		if y:
			new_sv.append( x)
	print "Vector size becomes: {0} --> {1}".format( len(sv), len(new_sv))
	return new_sv

def clean_smiles_vec_io( sv, out):
	""""
	It removes bad smiles code elements in smiles code vector
	as well as the corresponding outut value vector.
	"""
	new_sv = []
	new_out = []
	for x, o in zip( sv, out):
		y = Chem.MolFromSmiles(x)
		if y:
			new_sv.append( x)
			new_out.append( o)
	# print "Vector size becomes: {0} --> {1}".format( len(sv), len(new_sv))
	return new_sv, new_out

def gff( smiles = 'c1ccccc1O', rad = 2, nBits = 1024):
	"It generates fingerprint from smiles code"
	x = Chem.MolFromSmiles( smiles)
	return AllChem.GetMorganFingerprintAsBitVect( x, rad, nBits)

def gff_vec( smiles_vec, rad = 2, nBits = 1024):
	"It generates a fingerprint vector from a smiles code vector"
	return [gff(x, rad, nBits) for x in smiles_vec]

def _gff_binlist_r0( smiles_vec, rad = 2, nBits = 1024):
	"""
	It generates a binary list of fingerprint vector from a smiles code vector.
	Each string will be expanded to be the size of nBits such as 1024.
	- It shows error message when nBits < 1024 and len(x) > nBits.	
	"""
	ff_vec = gff_vec( smiles_vec, rad, nBits)
	ff_bin = [ bin(int(x.ToBinary().encode("hex"), 16)) for x in ff_vec]

	#Show error message when nBits < 1024 and len(x) > nBits	
	for x in ff_bin:
		if len(x[2:]) > nBits:
			print 'The length of x is {0}, which is larger than {1}'.format(len(x[2:]), nBits)
			print 'So, the minimal value of nBits must be 1024 generally.'
	return [ map( int, list( '0'*(nBits - len(x[2:])) + x[2:])) for x in ff_bin]

def gff_binlist( smiles_vec, rad = 2, nBits = 1024):
	"""
	It generates a binary list of fingerprint vector from a smiles code vector.
	Each string will be expanded to be the size of nBits such as 1024.
	- It shows error message when nBits < 1024 and len(x) > nBits.	
	- Now bits reduced to match input value of nBit eventhough the real output is large
	"""
	ff_vec = gff_vec( smiles_vec, rad, nBits)
	ff_bin = [ bin(int(x.ToBinary().encode("hex"), 16)) for x in ff_vec]

	#Show error message when nBits < 1024 and len(x) > nBits	
	"""
	for x in ff_bin:
		if len(x[2:]) > nBits:
			print 'The length of x is {0}, which is larger than {1}'.format(len(x[2:]), nBits)
			print 'So, the minimal value of nBits must be 1024 generally.'
	return [ map( int, list( '0'*(nBits - len(x[2:])) + x[2:])) for x in ff_bin]
	"""
	return [ map( int, list( jutil.sleast(x[2:], nBits))) for x in ff_bin]

def gff_binlist_bnbp( smiles_vec, rad = 2, nBits = 1024, bnbp = 'bn'):
	"""
	It generates a binary list of fingerprint vector from a smiles code vector.
	Each string will be expanded to be the size of nBits such as 1024.
	- It shows error message when nBits < 1024 and len(x) > nBits.	
	- Now bits reduced to match input value of nBit eventhough the real output is large
	bnbp --> if binary input, bnbp = 'bn', else if bipolar input, bnbp = 'bp'
	"""
	ff_vec = gff_vec( smiles_vec, rad, nBits)
	ff_bin = [ bin(int(x.ToBinary().encode("hex"), 16)) for x in ff_vec]

	if bnbp == 'bp': #bipolar input generation
		return [ map( jutil.int_bp, list( jutil.sleast(x[2:], nBits))) for x in ff_bin]
	else:
		return [ map( int, list( jutil.sleast(x[2:], nBits))) for x in ff_bin]


def gff_M( smiles_vec, rad = 2, nBits = 1024):
	"It generated a binary matrix from a smiles code vecor."
	return np.mat(gff_binlist( smiles_vec, rad, nBits))

def gff_M_bnbp( smiles_vec, rad = 2, nBits = 1024, bnbp = 'bn'):
	"It generated a binary matrix from a smiles code vecor."
	return np.mat(gff_binlist_bnbp( smiles_vec, rad, nBits, bnbp))


def ff_bin( smiles = 'c1ccccc1O'):
	"""
	It generates binary string fingerprint value
	Output -> '0b0010101...'
	"""

	mol = Chem.MolFromSmiles( smiles)
	fp = AllChem.GetMorganFingerprint(mol,2)
	
	fp_hex = fp.ToBinary().encode("hex")
	fp_bin = bin( int( fp_hex, 16))
	
	# print fp_bin

	return fp_bin

def ff_binstr( smiles = 'c1ccccc1O'):
	"""
	It generates binary string fingerprint value without head of 0b.
	So, in order to translate back into int value, the head should be attached 
	at the starting point. output_bin = '0b' + output_binstr
	Output -> '0010101...'
	"""

	mol = Chem.MolFromSmiles( smiles)
	fp = AllChem.GetMorganFingerprint(mol,2)
	
	fp_hex = fp.ToBinary().encode("hex")
	fp_bin = bin( int( fp_hex, 16))
	fp_binstr = fp_bin[2:]
	
	# print fp_bin

	return fp_binstr


def ff_int( smiles = 'c1ccccc1O'):
	"""
	It generates binary string fingerprint value
	Output -> long integer value
	which can be transformed to binary string using bin()
	"""

	mol = Chem.MolFromSmiles( smiles)
	fp = AllChem.GetMorganFingerprint(mol,2)
	
	fp_hex = fp.ToBinary().encode("hex")
	fp_int = int( fp_hex, 16)
	# fp_bin = bin( fp_int)
	# print fp_bin

	return fp_int

def calc_tm_dist_int( A_int, B_int):
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

	tm_dist = float(c) / float( a + b - c)

	return tm_dist

def calc_tm_dist( A_smiles, B_smiles):

	A_int = ff_int( A_smiles)
	B_int = ff_int( B_smiles)

	return calc_tm_dist_int( A_int, B_int)

def getw( Xs, Ys, N = 57, nBits = 400):
	"It calculate weight vector for specific N and nNBits."

	Xs50 = Xs[:N]
	Ys50 = Ys[:N]

	X = gff_M( Xs50, nBits=400)
	y = np.mat( Ys50).T

	print X.shape

	# Xw = y is assumed for Mutiple linear regression
	w = np.linalg.pinv( X) * y
	#print w

	plt.plot( w)
	plt.show()

	return w

def getw_clean( Xs, Ys, N = None, rad = 2, nBits = 1024):
	"Take only 50, each of which has safe smile code."
	nXs, nYs = clean_smiles_vec_io( Xs, Ys)

	# print len(nXs), len(nYs)

	if N is None:
		N = len( nXs)

	X = gff_M( nXs[:N], rad = rad, nBits = nBits)
	y = np.mat( nYs[:N]).T

	w = np.linalg.pinv( X) * y

	plt.plot( w)
	plt.title('Weight Vector')
	plt.show()

	y_calc = X*w
	e = y - y_calc
	se = (e.T * e)
	mse = (e.T * e) / len(e)
	print "SE =", se
	print "MSE =", mse
	print "RMSE =", np.sqrt( mse)
	
	plt.plot(e)
	plt.title("Error Vector: y - y_{calc}")
	plt.show()

	plt.plot(y, label='original')
	plt.plot(y_calc, label='predicted')
	plt.legend()
	plt.title("Output values: org vs. pred")
	plt.show()

	return w

def getw_clean_bnbp( Xs, Ys, N = None, rad = 2, nBits = 1024, bnbp = 'bn'):
	"""
	Take only 50, each of which has safe smile code.
	Translate the input into bipolar values.
	"""
	nXs, nYs = clean_smiles_vec_io( Xs, Ys)

	# print len(nXs), len(nYs)

	if N is None:
		N = len( nXs)

	X = gff_M_bnbp( nXs[:N], rad = rad, nBits = nBits, bnbp = bnbp)
	y = np.mat( nYs[:N]).T

	w = np.linalg.pinv( X) * y

	plt.plot( w)
	plt.title('Weight Vector')
	plt.show()

	y_calc = X*w
	e = y - y_calc
	se = (e.T * e)
	mse = (e.T * e) / len(e)
	print "SE =", se
	print "MSE =", mse
	print "RMSE =", np.sqrt( mse)
	
	plt.plot(e)
	plt.title("Error Vector: y - y_{calc}")
	plt.show()

	plt.plot(y, label='original')
	plt.plot(y_calc, label='predicted')
	plt.legend()
	plt.title("Output values: org vs. pred")
	plt.show()

	return w

def fpM_pat( xM):
	#%matplotlib qt
	xM_sum = np.sum( xM, axis = 0)

	plt.plot( xM_sum)
	plt.xlabel('fingerprint bit')
	plt.ylabel('Aggreation number')
	plt.show()

def gen_input_files( A, yV):
	"""
	Input files of ann_in.data and ann_run.dat are gerneated.
	The files are used in ann_aq.c (./ann_aq) 
	* Input: A is matrix, yV is vector
	"""
	# in file
	no_of_set = A.shape[0]
	no_of_input = A.shape[1]
	const_no_of_output = 1 # Now, only 1 output is considerd.
	with open("ann_in.data", "w") as f:
		f.write( "%d %d %d\n" % (no_of_set, no_of_input, const_no_of_output))
		for ix in range( no_of_set):
			for iy in range( no_of_input):
				f.write( "{} ".format(A[ix,iy]))
			f.write( "\n{}\n".format( yV[ix,0]))
		print("ann_in.data is saved")

	# run file 
	with open("ann_run.data", "w") as f:
		#In 2015-4-9, the following line is modified since it should not be 
		#the same to the associated line in ann_in data but it does not include the output length. 
		f.write( "%d %d\n" % (no_of_set, no_of_input))
		for ix in range( no_of_set):
			for iy in range( no_of_input):
				f.write( "{} ".format(A[ix,iy]))
			f.write( "\n") 
		print("ann_run.data is saved")

def gen_input_files_valid( At, yt, Av):
	"""
	Validation is also considerd.
	At and yt are for training while Av, yv are for validation.
	Input files of ann_in.data and ann_run.dat are gerneated.
	The files are used in ann_aq.c (./ann_aq) 
	* Input: At, Av is matrix, yt, yv is vector
	"""

	const_no_of_output = 1 # Now, only 1 output is considerd.

	# in file
	no_of_set = At.shape[0]
	no_of_input = At.shape[1]
	with open("ann_in.data", "w") as f:
		f.write( "%d %d %d\n" % (no_of_set, no_of_input, const_no_of_output))
		for ix in range( no_of_set):
			for iy in range( no_of_input):
				f.write( "{} ".format(At[ix,iy]))
			f.write( "\n{}\n".format( yt[ix,0]))
		print("ann_in.data with {0} sets, {1} inputs is saved".format( no_of_set, no_of_input))

	# run file 
	no_of_set = Av.shape[0]
	no_of_input = Av.shape[1]
	with open("ann_run.data", "w") as f:
		f.write( "%d %d\n" % (no_of_set, no_of_input))
		for ix in range( no_of_set):
			for iy in range( no_of_input):
				f.write( "{} ".format(Av[ix,iy]))
			f.write( "\n") 
		print("ann_run.data with {0} sets, {1} inputs is saved".format( no_of_set, no_of_input))


def get_valid_mode_data( aM, yV, rate = 3, more_train = True, center = None):
	"""
	Data is organized for validation. The part of them becomes training and the other becomes validation.
	The flag of 'more_train' represents tranin data is bigger than validation data, and vice versa.
	"""
	ix = range( len( yV))
	if center == None:
		center = int(rate/2)
	if more_train:
		ix_t = filter( lambda x: x%rate != center, ix)
		ix_v = filter( lambda x: x%rate == center, ix)
	else:
		ix_t = filter( lambda x: x%rate == center, ix)
		ix_v = filter( lambda x: x%rate != center, ix)

	aM_t, yV_t = aM[ix_t, :], yV[ix_t, 0]
	aM_v, yV_v = aM[ix_v, :], yV[ix_v, 0]

	return aM_t, yV_t, aM_v, yV_v	

def estimate_accuracy( yv, yv_ann, disp = False):
	"""
	The two column matrix is compared in this function and 
	It calculates RMSE and r_sqr.
	"""
	e = yv - yv_ann
	se = e.T * e
	RMSE = np.sqrt( se / len(e))

	# print "RMSE =", RMSE
	y_unbias = yv - np.mean( yv)
	s_y_unbias = y_unbias.T * y_unbias
	r_sqr = 1 - se/s_y_unbias

	if disp:
		print "r_sqr = {0}, RMSE = {1}".format( r_sqr[0,0], RMSE[0,0])

	return r_sqr[0,0], RMSE[0,0]

class FF_W:
	"""
	It calculrates the weight vector using MLR 
	when the input is fingerprint and the output is property.
	The main data X, Y is used for input and output for each member functions 
	while flags and options are used in self variables.
	"""
	def __init__(self, N = None, rad = 2, nBits = 1024, bnbp = 'bn', nBits_Max = False, smiles_clean = True):	
		"""
		nButs_cut is current impiled in the code, while it can be optional in order to 
		get the limit value as the largest length value of the input fingerprint vectors. 
		"""
		self.N = N
		self.rad = rad
		self.nBits = nBits
		self.bnbp = bnbp
		self.nBits_Max = nBits_Max
		self.smiles_clean = smiles_clean		

	def gff( self, x):
		x1 = Chem.MolFromSmiles( x)
		return AllChem.GetMorganFingerprintAsBitVect( x1, self.rad, self.nBits)

	def _getM_r0( self, X, Y, nBits_Max_val = None):
		if self.smiles_clean:
			X0, Y0 = clean_smiles_vec_io( X, Y)
		else:
			X0, Y0 = X, Y

		if self.N is None:
			N = len( X0)

		X1, Y1 = X0[:N], Y0[:N]	

		# fingerprint vectors
		X2 = [self.gff(x) for x in X1] 

		# vector of fingerprint binary string
		X3 = [bin(int(x.ToBinary().encode("hex"), 16)) for x in X2] 

		# Convert each binary string to binary character vectors
		if self.nBits_Max: # the maximum size is used for nBits
			if nBits_Max_val== None:
				len_X3 = map( len, [x[2:] for x in X3])
				nBits_Max_val = max( len_X3)
			X4 = [list( jutil.sleast(x[2:], nBits_Max_val)) for x in X3]
		else:
			X4 = [list( jutil.sleast(x[2:], self.nBits)) for x in X3]

		# Convert character (single element string)	to integer for computation
		if self.bnbp == 'bp': #bipolar input generation
			X5 = [ map( jutil.int_bp, x) for x in X4]
		else: #binary case
			X5 = [ map( int, x) for x in X4]

		X6, Y2 = np.mat( X5), np.mat( Y1).T

		return X6, Y2		

	def getM( self, X, Y, nBits_Max_val = None):
		"""
		Fingerprint matrix is generated.
		Regularization is considered depending on the flag. 
		"""
		if self.smiles_clean:
			X0, Y0 = clean_smiles_vec_io( X, Y)
		else:
			X0, Y0 = X, Y

		# self.clean_X = X0

		if self.N is None:
			N = len( X0)

		X1, Y1 = X0[:N], Y0[:N]	

		# fingerprint vectors
		X2 = [self.gff(x) for x in X1] 

		# vector of fingerprint binary string
		X3 = [bin(int(x.ToBinary().encode("hex"), 16)) for x in X2] 

		# Convert each binary string to binary character vectors
		if self.nBits_Max: # the maximum size is used for nBits
			if nBits_Max_val== None:
				len_X3 = map( len, [x[2:] for x in X3])
				nBits_Max_val = max( len_X3)
			X4 = [list( jutil.sleast(x[2:], nBits_Max_val)) for x in X3]
		else:
			X4 = [list( jutil.sleast(x[2:], self.nBits)) for x in X3]

		# Convert character (single element string)	to integer for computation
		if self.bnbp == 'bp': #bipolar input generation
			X5 = [ map( jutil.int_bp, x) for x in X4]
		elif self.bnbp == 'bn_reg':
			X5_tmp = [ map( float, x) for x in X4]
			X5 = []
			for x in X5_tmp:
				x_sum = np.sum(x)
				X5.append( map( lambda x_i: x_i / x_sum, x))

		else: #binary case
			X5 = [ map( int, x) for x in X4]

		X6, Y2 = np.mat( X5), np.mat( Y1).T

		return X6, Y2		


	def getw( self, smiles_vec, property_vec):
		"""
		Take only 50, each of which has safe smile code.
		Translate the input into bipolar values.
		"""

		X, y = self.getM( smiles_vec, property_vec)
		# print np.shape( X), np.shape( y)

		w = np.linalg.pinv( X) * y

		#===============================================
		# Code for showing the result as graphs 
		plt.plot( w)
		plt.title('Weight Vector')
		plt.xlabel('Index of weight vector')
		plt.ylabel('Weight value')
		plt.show()

		y_calc = X*w
		e = y - y_calc
		se = (e.T * e)
		mse = (e.T * e) / len(e)
		
		# print "SE =", se
		print "MSE =", mse
		print "RMSE =", np.sqrt( mse)
		

		y_unbias = y - np.mean( y)
		s_y_unbias = y_unbias.T * y_unbias
		r_sqr = 1 - se/s_y_unbias
		print "r_sqr = ", r_sqr

		
		#plt.plot(e)
		#plt.title("Error Vector: y - y_{calc}")
		#plt.show()

		plt.plot(y, label='original')
		plt.plot(y_calc, label='predicted')
		plt.legend()
		plt.title("[Training] Output values: org vs. pred")
		plt.xlabel("Molecule")
		plt.ylabel("Solubility (m/L)")
		plt.show()
		#===============================================

		return w

	def validw( self, smiles_vec, property_vec, w):
		"""
		Given w, the output values are evaluated with the original values. 
		The performace measuring values will be shown with various forms.
		"""

		if self.nBits_Max == True:
			X, y = self.getM( smiles_vec, property_vec, nBits_Max_val = len(w))
		else:
			X, y = self.getM( smiles_vec, property_vec)

		# print np.shape( X), np.shape( y)

		# w = np.linalg.pinv( X) * y

		#===============================================
		y_calc = X*w
		e = y - y_calc
		se = (e.T * e)
		mse = (e.T * e) / len(e)
		# print "SE =", se
		print "MSE =", mse
		print "RMSE =", np.sqrt( mse)
		

		y_unbias = y - np.mean( y)
		s_y_unbias = y_unbias.T * y_unbias
		r_sqr = 1 - se/s_y_unbias
		print "r_sqr = ", r_sqr

		
		#plt.plot(e)
		#plt.title("Error Vector: y - y_{calc}")
		#plt.show()

		plt.plot(y, label='original')
		plt.plot(y_calc, label='predicted')
		plt.legend()
		plt.title("[Validation] Output values: org vs. pred")
		plt.xlabel("Molecule")
		plt.ylabel("Solubility (m/L)")
		plt.show()
		#===============================================

		# print np.shape( X), np.shape( y), np.shape( y_calc)
		return X, y, y_calc


	def read_data( self, fname_csv = 'sheet/solubility-sorted-csv.csv',\
       x_field_name = 'Smile',\
       y_field_name = 'Water Solubility Estimate from Log Kow (WSKOW v1.41): Water Solubility at 25 deg C (mol/L)'):
		"""
		fname_csv = 'sheet/solubility-sorted-csv.csv'
		x_filed_name = 'Smile'
		y_field_name = 'Water Solubility Estimate from Log Kow (WSKOW v1.41): Water Solubility at 25 deg C (mol/L)'
		"""
		dfr = pd.read_csv( fname_csv)
		Xs = dfr[ x_field_name].tolist()
		Ys = dfr[ y_field_name].tolist()

		# Here, cleaning processing for smiles code list are performed mandatory because it read data from the file.
		# It it takes longtime, this part can be revised so that if statement can be included. 
		nXs, nYs = clean_smiles_vec_io( Xs, Ys)
		nLs = np.log( nYs) #log S is standard value to predict solubility
		#print [np.shape( x) for x in [nXs, nLs, nYs]]

		return nXs, nYs, nLs

	def write_data( self, nXs, nYs, nLs, fn = "sheet/cws_data_one.csv",\
			x_n = "Smile", y_n = "Solubility (mol/L)", l_n = "Log S"):
		"""
		The extracted data is saved so as to be used later on and to be seen from spreadsheet applications."
		data = { "Smile": nXs, "Solubility (mol/L)": nYs, "Log S": nLs}
		"""
		data = { x_n: nXs, y_n: nYs, l_n: nLs}
		dfw = pd.DataFrame( data)

		return dfw.to_csv( fn, index=False)

	def train_valid(self, X, Y):
		"It trains and validates modeling."

		#75% data will be used for modeling - 0, 2, 3 of 4 step elements
		X_train, Y_train = [], []
		for ii in jutil.prange( [0, 2, 3], 0, len( X), 4):
			X_train.append( X[ii])
			Y_train.append( Y[ii])
		# Define validation sequence - 25% of 2nd of 4 element collection
		X_v, Y_v = X[1::4], Y[1::4]  

		w = self.getw( X_train, Y_train) # traning with half of data set
		#xM, yV, calc_yV = self.validw( X_v, Y_v, w) #validation the other half of data set
		# The final output data is generated fro all input data including both traning and validation data
		self.validw( X_v, Y_v, w) #validation the other half of data set
		xM, yV, calc_yV = self.validw( X, Y, w)

		return w, xM, yV, calc_yV

	def train_valid2(self, X, Y):
		"It trains and validates modeling."

		#75% data will be used for modeling - 0, 2, 3 of 4 step elements
		X_train, Y_train = [], []
		for ii in jutil.prange( [0, 2], 0, len( X), 4):
			X_train.append( X[ii])
			Y_train.append( Y[ii])
		# Define validation sequence - 25% of 2nd of 4 element collection
		X_v, Y_v = [], []
		for ii in jutil.prange( [1, 3], 0, len( X), 4):
			X_v.append( X[ii])
			Y_v.append( Y[ii])

		w = self.getw( X_train, Y_train) # traning with half of data set
		print("Partial data validation")
		self.validw( X_v, Y_v, w) #validation the other half of data set
		print("Whole data vadidation")
		xM, yV, calc_yV = self.validw( X, Y, w)

		return w, xM, yV, calc_yV

	def train_valid_A( self, aM, yV):

		ix = range( len( yV))
		ix_t = filter( lambda x: x%3 == 0, ix)
		ix_v = filter( lambda x: x%3 != 0, ix)

		xMs_t = aM[ix_t, :]
		yVs_t = yV[ix_t, 0]
		w = np.linalg.pinv( xMs_t)*yVs_t

		yVs_t_calc = xMs_t * w
		e_t = yVs_t - yVs_t_calc
		#print "e(train) = ", e_t.T  
		RMSE = np.sqrt(e_t.T*e_t / len(e_t))
		print "RMSE(train) = ", RMSE

		plt.figure()
		plt.plot( yVs_t, label = 'Original')
		plt.plot( yVs_t_calc, label = 'Prediction')
		plt.legend()
		plt.grid()
		plt.title('Training')
		plt.show()		

		xMs_v = aM[ix_v, :]
		yVs_v = yV[ix_v, 0]
		yVs_v_calc = xMs_v * w

		e_v = yVs_v - yVs_v_calc
		#print e_v
		#print "yVs_v =", yVs_v.T 
		#print "yVs_v_calc =", yVs_v_calc.T 
		#print "e(valid) = ", e_v.T    
		RMSE = np.sqrt(e_v.T*e_v / len(e_v))
		print "RMSE(valid) = ", RMSE

		plt.figure()
		plt.plot( yVs_v, label = 'Original')
		plt.plot( yVs_v_calc, label = 'Prediction')
		plt.legend()
		plt.grid()
		plt.title('Validation')
		plt.show()

		print "All results"
		yV_calc = aM * w
		plt.figure()
		plt.plot( yV, label = 'Original')
		plt.plot( yV_calc, label = 'Prediction')
		plt.title( "All results")
		plt.legend()
		plt.show()

		return w

	def train_valid_rate( self, aM, yV, rate = 3, more_train = True):

		ix = range( len( yV))

		if more_train:
			ix_t = filter( lambda x: x%rate != int(rate/2), ix)
			ix_v = filter( lambda x: x%rate == int(rate/2), ix)
		else:
			ix_t = filter( lambda x: x%rate == int(rate/2), ix)
			ix_v = filter( lambda x: x%rate != int(rate/2), ix)

		xMs_t = aM[ix_t, :]
		yVs_t = yV[ix_t, 0]
		w = np.linalg.pinv( xMs_t)*yVs_t

		yVs_t_calc = xMs_t * w
		e_t = yVs_t - yVs_t_calc
		#print "e(train) = ", e_t.T  
		RMSE = np.sqrt(e_t.T*e_t / len(e_t))
		#print "RMSE(train) = ", RMSE
		estimate_accuracy( yVs_t, yVs_t_calc, disp = True)

		plt.figure()
		plt.plot( yVs_t, label = 'Original')
		plt.plot( yVs_t_calc, label = 'Prediction')
		plt.legend()
		plt.grid()
		plt.title('Training')
		plt.show()		

		xMs_v = aM[ix_v, :]
		yVs_v = yV[ix_v, 0]
		yVs_v_calc = xMs_v * w

		e_v = yVs_v - yVs_v_calc
		#print e_v
		#print "yVs_v =", yVs_v.T 
		#print "yVs_v_calc =", yVs_v_calc.T 
		#print "e(valid) = ", e_v.T    
		RMSE = np.sqrt(e_v.T*e_v / len(e_v))
		#print "RMSE(valid) = ", RMSE
		estimate_accuracy( yVs_v, yVs_v_calc, disp = True)

		plt.figure()
		plt.plot( yVs_v, label = 'Original')
		plt.plot( yVs_v_calc, '.-', label = 'Prediction')
		plt.legend()
		plt.grid()
		plt.title('Validation')
		plt.show()

		print "All results"
		yV_calc = aM * w
		estimate_accuracy( yV, yV_calc, disp = True)
		plt.figure()
		plt.plot( yV, label = 'Original')
		plt.plot( yV_calc, '.-', label = 'Prediction')
		plt.title( "All results")
		plt.legend()
		plt.show()

		return w

	#def get_clean_X(self):
	#	return self.clean_X

