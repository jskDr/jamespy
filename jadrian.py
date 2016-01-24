"""
adrin_rp.py
This file collects functions and classes used in Redox potential prediction in metabolism, 
which I am collaborating with Adrian. 
"""

import numpy as np
import pandas as pd

# Optimization codes are described
from sklearn import cross_validation
from scipy.optimize import minimize

import jpandas as jpd


#################################################
# Utilities
#################################################
def to_kegg_id( pdr, cn = "KEGG_ID_Adrian", fname = None, disp = False):
	"""
	New column is added. This new colulum is simplied KEGG_ID,
	which is generated from full KEGG_ID whether it is Adrian or Ed's.
	"""
	pdw = pdr.copy()
	KEGG_ID_l = list()
	for kid_a in pdr[cn]:        
		tx_all, rx_all = kid_a.split(" = ")
		kid = "{} = {}".format( tx_all[:6], rx_all[:6])
		if disp:
			print kid_a, "==>", kid
		KEGG_ID_l.append( kid) 
	pdw["KEGG_ID"] = KEGG_ID_l
	
	if fname:
		"""
		If file name is given, the results will be directly saved.
		"""
		pdw.to_csv( fname, index = False)

	return pdw

#################################################
# Application codes
#################################################
def init():
	"""
	The two data frames are loaded.
	The result data frames are allocated to global variables since it will used later.
	However, they are not loaded although realod() is performed. When reload() is performed,
	only code will be changed except the init() is performed. 
	"""
	global pdr_code, pdr_reaction, pdr2, pdr3
	pdr_code = jpd.pd.read_csv('KEGG_to_smiles.csv') 
	pdr_reaction = jpd.pd.read_csv('DataFrame_Redox_Reactions.csv')
	pdr2 = jpd.pd.read_csv( 'DataFrame_Redox_Reactions_with_smiles122.csv')
	pdr3 = jpd.pd.read_csv( 'redox_quantum_chemistry_v2+pred_comma.csv')

def getsm( mid = 'C00025'):    
	return pdr_code[ pdr_code['KEGG_ID'] == mid].SMILES.tolist()[0]

def sep_ids( pdw):
	x1 = list()
	x2 = list()
	for x in pdr_reaction['KEGG_ID']:
		x1.append( x[:6])
		x2.append( x[9:9+6])
	pdw['left_id'] = x1
	pdw['right_id'] = x2
	return pdw

def get_2smiles( pdw):
	left_smiles = map( getsm, pdw.left_id)     
	right_smiles = map( getsm, pdw.right_id) 
	
	pdw['left_smiles'] = left_smiles
	pdw['right_smiles'] = right_smiles
	
	return pdw

def gen_pdw():
	"""
	pdr_reaction will be extended to have the two smiles codes since
	they are needed to make a model for regression from SMILES to redox potential.
	The result data frame will be saved to a file for a later use.
	"""
	pdw = pdr_reaction.copy()
	pdw = sep_ids( pdw)
	pdw = get_2smiles( pdw)

	pdw.to_csv('DataFrame_Redox_Reactions_with_smiles.csv', index = False)
	return pdw

import os

from rdkit import Chem
from rdkit.Chem import AllChem
#from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import Descriptors #as Descriptors
from rdkit.Chem import PandasTools #as PandasTools
from rdkit.Chem import FragmentCatalog

class Frag():
	"""
	This class investigate molecules whether they have a specific fragment.  
	"""
	def __init__( self, FunctionalGroups_txt = "FunctionalGroups.txt"):
		fName=os.path.join(FunctionalGroups_txt)
		self.fparams = FragmentCatalog.FragCatParams(1,6,fName)

	def search( self, a_smiles = 'OC(=O)[C@H](CC(=O)O)N'):
		"""
		It results frag_map which indicates the map of matching fragments. 
		If only the first fragment is matched, only the first element of the
		vector is turned as True for example.  
		"""
		frag_map = list()
		for indx in range(self.fparams.GetNumFuncGroups()):
			patt = self.fparams.GetFuncGroup(indx)
			m = Chem.MolFromSmiles( a_smiles)
			match=m.HasSubstructMatch( patt)
			frag_map.append( match)

		return frag_map

	def search_idx( self, frag_idx, s_l):
		"""
		It searches all molecules in a vector whether the molecules have
		the given fragment. Hence, each element of the return vector is
		corresponding to each element of a vector of SMILES. 
		Moreover, exclusiveness is also tested by calculating a sum of  
		frag_map. If the sum is more than one, it is not exclusive for single
		fragment when the corresponding smiles_map is True. 
		"""
		smiles_map = list()
		exclusive_map = list()
		for s in s_l:
			frag_map = self.search( s)
			smiles_map.append(frag_map[ frag_idx] == True)
			exclusive_map.append( sum( frag_map))

		return smiles_map, exclusive_map

import scipy.io

class GroupContr():
	"""
	This is a group contribution method, which can preditcs, for example, redox potential
	by addiding up all contributions from fragments. 
	This will use one matlab data file, which includes index data of KEGG_ID, 
	and a collection of vectors with respect to the index data. 
	If you ask group information for a molecule, 
	it search the index of it first. Then, it extract the contribution vector of the 
	founded index.   
	"""

	def __init__( self, fname_mat = 'sheet/kegg_group_decomposition.mat', disp = False):	
		mat = scipy.io.loadmat( fname_mat)
		self.cids = mat['cids']
		self.group_mat = mat['gv_mat']

		self.disp = disp

	def get_index( self, subs_id = 'C00376'):
		"""
		It finds the index of a given kegg_id. 
		There will be a method to improve the speed of founding using C or C++ codes.
		It is also possible to make C++ class all this class later on. 
		"""

		cids_result = np.where( self.cids == subs_id)[0]
		if len(cids_result) > 0:
			return cids_result[0]
		else:
			# if no cids_id is founded. 
			if self.disp:
				print 'No found for', subs_id 
				print 'The search result is', cids_result
				print 'So, the output becomes -1 for notification.'
			return -1	

		#return np.where( self.cids == subs_id)[0][0]

	def get_2indices_kegg_id( self, kegg_id = 'C00376 = C00473'):
		"""
		The reaction will be separated to subs and prod. 
		"""
		sub_prod = kegg_id.split(' = ')
		return map( self.get_index, sub_prod)

	def get_group_descriptor( self, idx):
		"""
		It returns the group contribution descriptor of a given molecule. 
		"""
		return self.group_mat[ idx, :]

	def get_2group_descriptors_kegg_id( self, idx2):
		#print "DEBUG: idx2 -->", idx2

		return self.group_mat[ idx2[0], :], self.group_mat[ idx2[1], :]

	def kegg_id_list_to_2group_descriptors( self, kegg_id_list):
		"""
		It returns group contribution descriptors for subs and prods separately. 
		The return lists can be stored in the original csv file as additional information. 
		"""
		# list of subs's group descriptors, a list of prod's group descriptors
		subs_gd_list = list() 
		prod_gd_list = list()
		no_cids_list = list()  

		for kegg_id in kegg_id_list:
			idx2 = self.get_2indices_kegg_id( kegg_id)
			if -1 in idx2:
				no_cids_list.append( True)
				subs_gd_list.append( None)
				prod_gd_list.append( None)
			else:
				no_cids_list.append( False)
				gd2_list = self.get_2group_descriptors_kegg_id( idx2)
				subs_gd_list.append( gd2_list[0])
				prod_gd_list.append( gd2_list[1])

		return subs_gd_list, prod_gd_list, no_cids_list

class ML_GroupContr( GroupContr):
	"""
	This class performs machine learning based on group contribution. 
	"""
	def __init__(self, fname_csv = 'sheet/Redox_ForJames_TPSS0_2015_11_04_comma.csv', 
					fname_mat = 'sheet/kegg_group_decomposition.mat', disp = False):
		GroupContr.__init__( self, fname_mat, disp)
		self.pdr = pd.read_csv( fname_csv)

	def k2d( self, save_fname = None):
		"""
		It translate kegg_id_list to descriptors
		"""
		xM_subs, xM_prod, no_csid_list = self.kegg_id_list_to_2group_descriptors( self.pdr.KEGG_ID)

		self.pdw = self.pdr.copy()
		self.pdw[ 'xM_subs'] = xM_subs
		self.pdw[ 'xM_prod'] = xM_prod
		self.pdw[ 'no_csid'] = no_csid_list

		if save_fname != None:
			"""
			If filename is given as an argument, the result data 
			will be stored together with the original data.
			"""
			self.pdw.to_csv( save_fname, index = False)

		return self.pdw

	def k2d_valid( self, save_fname = None):

		self.k2d()
		self.pdw_valid = self.pdw[ self.pdw[ 'no_csid'] == False]

		if save_fname != None:
			"""
			If filename is given as an argument, the result data 
			will be stored together with the original data.
			"""
			self.pdw_valid.to_csv( save_fname, index = False)

		return self.pdw_valid

	def get_xM_yV(self, a_pdw_valid):
		"""
		From string arrays, original arrays will be recovered. 
		"""
		#xM_subs_list = [ eval(x) for x in a_pdw_valid[ 'xM_subs'].tolist()]
		xM_subs_list = a_pdw_valid[ 'xM_subs'].tolist()
		xM_subs = np.mat( xM_subs_list)
		#xM_prod_list = [ eval(x) for x in a_pdw_valid[ 'xM_prod'].tolist()]
		xM_prod_list = a_pdw_valid[ 'xM_prod'].tolist()
		xM_prod = np.mat( xM_prod_list)
		xM = np.concatenate( [xM_subs, xM_prod], axis = 1)

		if self.disp:
			print 'xM_subs.shape, xM_prod.shape, xM.shape =', xM_subs.shape, xM_prod.shape, xM.shape

		yV = np.mat( a_pdw_valid[ 'Em']).T

		return xM, yV

	def get_sM_yV(self, a_pdw_valid):
		"""
		sM is xM_prod - xM_subs 
		"""
		#xM_subs_list = [ eval(x) for x in a_pdw_valid[ 'xM_subs'].tolist()]
		xM_subs_list = a_pdw_valid[ 'xM_subs'].tolist()
		xM_subs = np.mat( xM_subs_list)
		#xM_prod_list = [ eval(x) for x in a_pdw_valid[ 'xM_prod'].tolist()]
		xM_prod_list = a_pdw_valid[ 'xM_prod'].tolist()
		xM_prod = np.mat( xM_prod_list)
		sM = xM_prod - xM_subs

		if self.disp:
			print 'xM_subs.shape, xM_prod.shape, xM.shape =', xM_subs.shape, xM_prod.shape, sM.shape

		yV = np.mat( a_pdw_valid[ 'Em']).T

		return sM, yV

#######################
# Regression codes
#######################

import jutil
from sklearn import linear_model, cross_validation
import matplotlib.pyplot as plt

def plot_per_type( pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4]):
	"""
	plot data for each group with different color
	pdr: pandas dataframe
	type_id: name of the type column
	type_l: list of illustration types
	"""
	for type_id in type_l:
		pdr_new = pdr[ pdr[ type_name] == type_id]
		xM = np.mat( pdr_new[ E_QC].values).T
		print xM.shape
		yV = np.mat( pdr_new[ Em].values).T
		print yV.shape
		plt.plot( yV, xM, '.', label = "Type={}".format(type_id))

	plt.xlabel('Experiment')
	plt.ylabel('Estimation')
	plt.legend(loc=2)

def single_regress( pdr, E_QC = "E_QC", Em = "Em"):
	xM_all = np.mat( pdr[E_QC].values).T
	print xM_all.shape

	yV_all = np.mat( pdr[Em].values).T
	print yV_all.shape

	print "No regression case"
	plt.figure()
	jutil.regress_show3( yV_all, xM_all)
	plt.show()

	print "Simple regression case"
	plt.figure()
	jutil.mlr3( xM_all, yV_all)
	plt.show()

def single_regress4( pdr, E_QC = "E_QC", Em = "Em", disp = True, graph = True):
	xM_all = np.mat( pdr[E_QC].values).T
	print xM_all.shape

	yV_all = np.mat( pdr[Em].values).T
	print yV_all.shape

	print "No regression case"
	#plt.figure()
	jutil.regress_show4( yV_all, xM_all, disp = disp, graph = graph)
	#plt.show()

	print "Simple regression case"
	#plt.figure()
	jutil.mlr4_coef( xM_all, yV_all, disp = disp, graph = graph)
	#plt.show()

def sep_noregress( pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4]):
	for type_id in type_l:
		print "Type", type_id
		pdr_new = pdr[ pdr[ type_name] == type_id]
		xM = np.mat( pdr_new[E_QC].values).T
		print xM.shape
		yV = np.mat( pdr_new[Em].values).T
		print yV.shape
		jutil.regress_show3( yV, xM)

def sep_noregress4( pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4]):
	for type_id in type_l:
		print "Type", type_id
		pdr_new = pdr[ pdr[ type_name] == type_id]
		xM = np.mat( pdr_new[E_QC].values).T
		print xM.shape
		yV = np.mat( pdr_new[Em].values).T
		print yV.shape
		jutil.regress_show4( yV, xM)

def sep_regress4( pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4]):
	# Select interesting parts only
	#pdr = pdr_org.query( "Type in {}".format( type_id_l))

	for type_id in type_l:
		print "Type", type_id
		pdr_new = pdr[ pdr[type_name] == type_id]
		xM = np.mat( pdr_new[E_QC].values).T
		print xM.shape
		yV = np.mat( pdr_new[Em].values).T
		print yV.shape
		jutil.mlr4_coef( xM, yV)

def sep_regress_merge( pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4]):
	# Select interesting parts only
	# pdr = pdr_org.query( "Type in {}".format( type_id_l))
	
	yV_pred_all_list = list()
	for type_id in type_l:
		print "Type", type_id
		pdr_new = pdr[ pdr[ type_name] == type_id]
		xM = np.mat( pdr_new[E_QC].values).T
		print xM.shape
		yV = np.mat( pdr_new[Em].values).T
		print yV.shape
		clf = linear_model.LinearRegression()
		clf.fit( xM, yV)
		yV_pred = clf.predict( xM).ravel()
		jutil.regress_show3( yV, yV_pred)
		yV_pred_all_list.append( yV_pred)

	print "Merging"
	yV_pred_all = np.mat( np.concatenate( yV_pred_all_list, axis = 0)).T
	yV_all = np.mat( pdr[Em].values).T

	plt.figure()
	jutil.regress_show3( yV_all, yV_pred_all)
	plt.show()
	
	plt.figure()
	jutil.mlr3( yV_pred_all, yV_all)
	plt.show()

def sep_regress_merge4( pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4]):
	# Select interesting parts only
	# pdr = pdr_org.query( "Type in {}".format( type_id_l))
	
	yV_pred_all_list = list()
	for type_id in type_l:
		print "Type", type_id
		pdr_new = pdr[ pdr[ type_name] == type_id]
		xM = np.mat( pdr_new[E_QC].values).T
		print xM.shape
		yV = np.mat( pdr_new[Em].values).T
		print yV.shape
		clf = linear_model.LinearRegression()
		clf.fit( xM, yV)
		yV_pred = clf.predict( xM).ravel()
		jutil.regress_show4( yV, yV_pred)
		yV_pred_all_list.append( yV_pred)

	print "Merging"
	yV_pred_all = np.mat( np.concatenate( yV_pred_all_list, axis = 0)).T
	yV_all = np.mat( pdr[Em].values).T

	plt.figure()
	jutil.regress_show4( yV_all, yV_pred_all)
	plt.show()
	
	plt.figure()
	jutil.mlr4_coef( yV_pred_all, yV_all)
	plt.show()	


def emul_get( pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4]):
	"""
	New descirptor = [x, 0, 0 ,0] for type_1, vice versa.
	"""
	
	xM_l = list()
	for ix, type_id in enumerate(type_l):
		#print "Type[{0}] -> {1}".format( ix, type_id)
		pdr_new = pdr[ pdr[ type_name] == type_id]
		x = pdr_new[E_QC].values
		x_ext = np.zeros( (len( type_l)*2, x.shape[0]), dtype = float)
		x_ext[ix][:] = x
		x_ext[len( type_l)+ix][:] = np.ones( x.shape[0], dtype = float)
		xM_l.append( x_ext)
	xM = np.mat( np.concatenate( xM_l, axis = 1)).T
	yV = np.mat( pdr[Em].values).T
	
	return xM, yV


def emul_regress_merge4( pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4]):
	"""
	New descirptor = [x, 0, 0 ,0] for type_1, vice versa.
	"""
	xM, yV = emul_get( pdr, E_QC = E_QC, Em = Em, type_name = type_name, type_l = type_l)
	
	jutil.mlr4_coef( xM, yV)


def _emul_regress_merge4( pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4]):
	"""
	New descirptor = [x, 0, 0 ,0] for type_1, vice versa.
	"""
	
	xM_l = list()
	for ix, type_id in enumerate(type_l):
		print "Type[{0}] -> {1}".format( ix, type_id)
		pdr_new = pdr[ pdr[ type_name] == type_id]
		x = pdr_new[E_QC].values
		x_ext = np.zeros( (len( type_l)*2, x.shape[0]), dtype = float)
		x_ext[ix][:] = x
		x_ext[len( type_l)+ix][:] = np.ones( x.shape[0], dtype = float)
		xM_l.append( x_ext)
	xM = np.mat( np.concatenate( xM_l, axis = 1)).T
	yV = np.mat( pdr[Em].values).T
	
	jutil.mlr4_coef( xM, yV)

def hybrid_get( pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4]):
	"""
	It generates hybrid descriptors. One additional descriptors for all x values regardless of 
	type. Using this, overfitting can be reduced.  
	"""

	xM_l = list()
	for ix, type_id in enumerate(type_l):
		# print "Type[{0}] -> {1}".format( ix, type_id)
		pdr_new = pdr[ pdr[ type_name] == type_id]
		x = pdr_new[E_QC].values
		x_ext = np.zeros( (len( type_l)*2+1, x.shape[0]), dtype = float)
		x_ext[ix][:] = x
		x_ext[len( type_l)+ix][:] = np.ones( x.shape[0], dtype = float)
		x_ext[len( type_l)*2][:] = x
		xM_l.append( x_ext)
	xM = np.mat( np.concatenate( xM_l, axis = 1)).T
	yV = np.mat( pdr[Em].values).T

	return xM, yV

def hybrid_regress_merge4( pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4]):
	"""
	A new descriptor is a sub function which is hybrid_get().
	"""
	
	xM, yV = hybrid_get( pdr, E_QC = E_QC, Em = Em, type_name = type_name, type_l = type_l)
	jutil.mlr4_coef( xM, yV)


def _hybrid_regress_merge4( pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4]):
	"""
	New descirptor = [x, 0, 0 ,0] for type_1, vice versa.
	"""
	
	xM_l = list()
	for ix, type_id in enumerate(type_l):
		print "Type[{0}] -> {1}".format( ix, type_id)
		pdr_new = pdr[ pdr[ type_name] == type_id]
		x = pdr_new[E_QC].values
		x_ext = np.zeros( (len( type_l)*2+1, x.shape[0]), dtype = float)
		x_ext[ix][:] = x
		x_ext[len( type_l)+ix][:] = np.ones( x.shape[0], dtype = float)
		x_ext[len( type_l)*2][:] = x
		xM_l.append( x_ext)
	xM = np.mat( np.concatenate( xM_l, axis = 1)).T
	yV = np.mat( pdr[Em].values).T
	
	jutil.mlr4_coef( xM, yV)

def cv_train_test( xMa, yVa, tr, ts):
	"""
	Regression and test is performed for given data
	with cross-validation streams
	"""		
	xM = xMa[ tr, :]
	yV = yVa[ tr, 0]
	
	clf = linear_model.LinearRegression()
	clf.fit( xM, yV)

	# The testing information is extracted.
	xM_test = xMa[ ts, :]
	yV_test = yVa[ ts, 0]

	return yV_test.A1, clf.predict( xM_test).ravel()

class Cv_sep_regress( object):
	def __init__( self, pdr, E_QC = "E_QC", Em = "Em", type_name = "Type", type_l = [1,2,3,4], 
					disp = False, graph = False):

		# This parameter will be used in the run() function. 
		self.type_l = type_l
		self.disp = disp
		self.graph = graph

		self.xMa = {}
		self.yVa = {}
		# self.kfa = {}
		for type_id in type_l:
			pdr_new = pdr[ pdr[ type_name] == type_id]
			self.xMa[type_id] = np.mat( pdr_new[ E_QC].values).T
			self.yVa[type_id] = np.mat( pdr_new[ Em].values).T
			# kfa[type_id] = cross_validation.KFold( np.shape(yVa[type_id])[0], n_folds=5, shuffle=True)

	def run(self, n_folds = 5):
		# Without initializatoin just before for-loop, the list is stacked continously. 
		yV_test_all_list = list()
		yV_pred_all_list = list()

		kfa = {}
		kfa_tr = {}
		kfa_ts = {}

		for type_id in self.type_l:
			kfa[type_id] = cross_validation.KFold( np.shape( self.yVa[type_id])[0], 
							n_folds = n_folds, shuffle=True)
			kfa_tr[type_id] = list()
			kfa_ts[type_id] = list()
			for tr, ts in kfa[ type_id]:
				kfa_tr[type_id].append( tr)
				kfa_ts[type_id].append( ts)

		r2_l, RMSE_l, MAE_l = list(), list(), list()
		for kf_id in range( n_folds):
			yV_test_all_list = list()
			yV_pred_all_list = list()
			for type_id in self.type_l:
				tr = kfa_tr[ type_id][ kf_id]
				ts = kfa_ts[ type_id][ kf_id]
				xM = self.xMa[ type_id]
				yV = self.yVa[ type_id]

				yV_test, yV_pred = cv_train_test( xM, yV, tr, ts)

				yV_test_all_list.append( yV_test)
				yV_pred_all_list.append( yV_pred)

			yV_test_all = np.mat( np.concatenate( yV_test_all_list, axis = 0)).T
			yV_pred_all = np.mat( np.concatenate( yV_pred_all_list, axis = 0)).T
			r2, RMSE, MAE = jutil.regress_show3( yV_test_all, yV_pred_all, disp = self.disp, graph = self.graph)
			r2_l.append( r2), RMSE_l.append( RMSE), MAE_l.append( MAE)

		if self.disp:
			print 'Mean and Std of R2 are', np.mean( r2_l), np.std( r2_l)
			print 'Mean and Std of RMSE are', np.mean( RMSE_l), np.std( RMSE_l)
			print 'Mean and Std of MAE are', np.mean( MAE_l), np.std( MAE_l)

		return r2_l, RMSE_l, MAE_l

	def run4(self, n_folds = 5):
		# Without initializatoin just before for-loop, the list is stacked continously. 
		yV_test_all_list = list()
		yV_pred_all_list = list()

		kfa = {}
		kfa_tr = {}
		kfa_ts = {}

		for type_id in self.type_l:
			kfa[type_id] = cross_validation.KFold( np.shape( self.yVa[type_id])[0], 
							n_folds = n_folds, shuffle=True)
			kfa_tr[type_id] = list()
			kfa_ts[type_id] = list()
			for tr, ts in kfa[ type_id]:
				kfa_tr[type_id].append( tr)
				kfa_ts[type_id].append( ts)

		r2_l, RMSE_l, MAE_l, DAE_l = list(), list(), list(), list()
		for kf_id in range( n_folds):
			yV_test_all_list = list()
			yV_pred_all_list = list()
			for type_id in self.type_l:
				tr = kfa_tr[ type_id][ kf_id]
				ts = kfa_ts[ type_id][ kf_id]
				xM = self.xMa[ type_id]
				yV = self.yVa[ type_id]

				yV_test, yV_pred = cv_train_test( xM, yV, tr, ts)

				yV_test_all_list.append( yV_test)
				yV_pred_all_list.append( yV_pred)

			yV_test_all = np.mat( np.concatenate( yV_test_all_list, axis = 0)).T
			yV_pred_all = np.mat( np.concatenate( yV_pred_all_list, axis = 0)).T
			r2, RMSE, MAE, DAE = jutil.regress_show4( yV_test_all, yV_pred_all, disp = self.disp, graph = self.graph)
			r2_l.append( r2), RMSE_l.append( RMSE), MAE_l.append( MAE), DAE_l.append( DAE)

		if self.disp:
			print 'Mean and Std of R2 are', np.mean( r2_l), np.std( r2_l)
			print 'Mean and Std of RMSE are', np.mean( RMSE_l), np.std( RMSE_l)
			print 'Mean and Std of MAE are', np.mean( MAE_l), np.std( MAE_l)
			print 'Mean and Std of DAE are', np.mean( DAE_l), np.std( DAE_l)

		return r2_l, RMSE_l, MAE_l, DAE_l

	def run_id(self, n_folds = 5):
		# Without initializatoin just before for-loop, the list is stacked continously. 
		yV_test_all_list = list()
		yV_pred_all_list = list()

		kfa = {}
		kfa_tr = {}
		kfa_ts = {}

		r2_id_d, RMSE_id_d, MAE_id_d = {}, {}, {}
		yV_test_d, yV_pred_d = {}, {}
		for type_id in self.type_l:
			kfa[type_id] = cross_validation.KFold( np.shape( self.yVa[type_id])[0], 
							n_folds = n_folds, shuffle=True)
			kfa_tr[type_id] = list()
			kfa_ts[type_id] = list()
			for tr, ts in kfa[ type_id]:
				kfa_tr[type_id].append( tr)
				kfa_ts[type_id].append( ts)

			for x in [yV_test_d, yV_pred_d, r2_id_d, RMSE_id_d, MAE_id_d]:
				x[type_id] = list()

		r2_l, RMSE_l, MAE_l = list(), list(), list()
		for kf_id in range( n_folds):
			yV_test_all_list = list()
			yV_pred_all_list = list()
			for type_id in self.type_l:
				tr = kfa_tr[ type_id][ kf_id]
				ts = kfa_ts[ type_id][ kf_id]
				xM = self.xMa[ type_id]
				yV = self.yVa[ type_id]

				yV_test, yV_pred = cv_train_test( xM, yV, tr, ts)

				yV_test_all_list.append( yV_test)
				yV_pred_all_list.append( yV_pred)

				yV_test_d[type_id].append( yV_test)
				yV_pred_d[type_id].append( yV_pred)

			yV_test_all = np.mat( np.concatenate( yV_test_all_list, axis = 0)).T
			yV_pred_all = np.mat( np.concatenate( yV_pred_all_list, axis = 0)).T
			r2, RMSE, MAE = jutil.regress_show3( yV_test_all, yV_pred_all, disp = self.disp, graph = self.graph)
			r2_l.append( r2), RMSE_l.append( RMSE), MAE_l.append( MAE)

		for type_id in self.type_l:
			for kf_id in range( n_folds):				
				r2, RMSE, MAE = jutil.regress_show3( yV_test_d[type_id][kf_id], yV_pred_d[type_id][kf_id],
												disp = self.disp, graph = self.graph)			
				r2_id_d[type_id].append( r2)
				RMSE_id_d[type_id].append( RMSE)
				MAE_id_d[type_id].append( MAE)

		if self.disp:
			print 'Mean and Std of R2 are', np.mean( r2_l), np.std( r2_l)
			print 'Mean and Std of RMSE are', np.mean( RMSE_l), np.std( RMSE_l)
			print 'Mean and Std of MAE are', np.mean( MAE_l), np.std( MAE_l)

		return (r2_l, RMSE_l, MAE_l), (r2_id_d, RMSE_id_d, MAE_id_d)

	def run4_id(self, n_folds = 5):
		# Without initializatoin just before for-loop, the list is stacked continously. 
		yV_test_all_list = list()
		yV_pred_all_list = list()

		kfa = {}
		kfa_tr = {}
		kfa_ts = {}

		r2_id_d, RMSE_id_d, MAE_id_d, DAE_id_d = {}, {}, {}, {}
		yV_test_d, yV_pred_d = {}, {}
		for type_id in self.type_l:
			kfa[type_id] = cross_validation.KFold( np.shape( self.yVa[type_id])[0], 
							n_folds = n_folds, shuffle=True)
			kfa_tr[type_id] = list()
			kfa_ts[type_id] = list()
			for tr, ts in kfa[ type_id]:
				kfa_tr[type_id].append( tr)
				kfa_ts[type_id].append( ts)

			for x in [yV_test_d, yV_pred_d, r2_id_d, RMSE_id_d, MAE_id_d, DAE_id_d]:
				x[type_id] = list()

		r2_l, RMSE_l, MAE_l, DAE_l = list(), list(), list(), list()
		for kf_id in range( n_folds):
			yV_test_all_list = list()
			yV_pred_all_list = list()
			for type_id in self.type_l:
				tr = kfa_tr[ type_id][ kf_id]
				ts = kfa_ts[ type_id][ kf_id]
				xM = self.xMa[ type_id]
				yV = self.yVa[ type_id]

				yV_test, yV_pred = cv_train_test( xM, yV, tr, ts)

				yV_test_all_list.append( yV_test)
				yV_pred_all_list.append( yV_pred)

				yV_test_d[type_id].append( yV_test)
				yV_pred_d[type_id].append( yV_pred)

			yV_test_all = np.mat( np.concatenate( yV_test_all_list, axis = 0)).T
			yV_pred_all = np.mat( np.concatenate( yV_pred_all_list, axis = 0)).T
			r2, RMSE, MAE, DAE = jutil.regress_show4( yV_test_all, yV_pred_all, disp = self.disp, graph = self.graph)
			r2_l.append( r2), RMSE_l.append( RMSE), MAE_l.append( MAE), DAE_l.append( DAE)

		for type_id in self.type_l:
			for kf_id in range( n_folds):				
				r2, RMSE, MAE, DAE = jutil.regress_show4( yV_test_d[type_id][kf_id], yV_pred_d[type_id][kf_id],
												disp = self.disp, graph = self.graph)			
				r2_id_d[type_id].append( r2)
				RMSE_id_d[type_id].append( RMSE)
				MAE_id_d[type_id].append( MAE)
				DAE_id_d[type_id].append( DAE)

		if self.disp:
			print 'Mean and Std of R2 are', np.mean( r2_l), np.std( r2_l)
			print 'Mean and Std of RMSE are', np.mean( RMSE_l), np.std( RMSE_l)
			print 'Mean and Std of MAE are', np.mean( MAE_l), np.std( MAE_l)
			print 'Mean and Std of DAE are', np.mean( DAE_l), np.std( DAE_l)

		return (r2_l, RMSE_l, MAE_l, DAE_l), (r2_id_d, RMSE_id_d, MAE_id_d, DAE_id_d)

	def run_iter(self, Niter = 10, n_folds = 5):
		r2_ll, RMSE_ll, MAE_ll = list(), list(), list()
		for ii in range( Niter):
			r2_l, RMSE_l, MAE_l = self.run( n_folds)
			r2_ll.extend( r2_l), RMSE_ll.extend( RMSE_l), MAE_ll.extend( MAE_l)

		if self.disp:
			print 'Mean and Std of R2 with 10 times 5-fold CV are', np.mean( r2_ll), np.std( r2_ll)
			print 'Mean and Std of RMSE with 10 times 5-fold CV are', np.mean( RMSE_ll), np.std( RMSE_ll)
			print 'Mean and Std of MAE with 10 times 5-fold CV are', np.mean( MAE_ll), np.std( MAE_ll)
			
		pdw = pd.DataFrame()
		pdw['Measure'] = ['R2', 'R2', 'RMSE', 'RMSE', 'MAE', 'MAE']
		pdw['Mode'] = ['Mean', 'STD'] * 3
		pdw['Value'] = [np.mean( r2_ll), np.std( r2_ll), 
						np.mean( RMSE_ll), np.std( RMSE_ll), 
						np.mean( MAE_ll), np.std( MAE_ll)]

		return pdw

	def run4_iter(self, Niter = 10, n_folds = 5):
		r2_ll, RMSE_ll, MAE_ll, DAE_ll = list(), list(), list(), list()
		for ii in range( Niter):
			r2_l, RMSE_l, MAE_l, DAE_l = self.run4( n_folds)
			r2_ll.extend( r2_l), RMSE_ll.extend( RMSE_l), MAE_ll.extend( MAE_l), DAE_ll.extend( DAE_l)

		if self.disp:
			print 'Mean and Std of R2 with 10 times 5-fold CV are', np.mean( r2_ll), np.std( r2_ll)
			print 'Mean and Std of RMSE with 10 times 5-fold CV are', np.mean( RMSE_ll), np.std( RMSE_ll)
			print 'Mean and Std of MAE with 10 times 5-fold CV are', np.mean( MAE_ll), np.std( MAE_ll)
			print 'Mean and Std of DAE with 10 times 5-fold CV are', np.mean( DAE_ll), np.std( DAE_ll)	

		pdw = pd.DataFrame()
		pdw['Measure'] = ['R2', 'R2', 'RMSE', 'RMSE', 'MAE', 'MAE', 'DAE', 'DAE']
		pdw['Mode'] = ['Mean', 'STD'] * 4
		pdw['Value'] = [np.mean( r2_ll), np.std( r2_ll), 
						np.mean( RMSE_ll), np.std( RMSE_ll), 
						np.mean( MAE_ll), np.std( MAE_ll), 
						np.mean( DAE_ll), np.std( DAE_ll)]

		return pdw


	def run_id_iter(self, Niter = 10, n_folds = 5):

		r2_ll, RMSE_ll, MAE_ll = list(), list(), list()
		
		r2_lld, RMSE_lld, MAE_lld = {}, {}, {}
		for type_id in self.type_l:
			r2_lld[type_id] = list()
			RMSE_lld[type_id] = list()
			MAE_lld[type_id] = list()

		for ii in range( Niter):
			(r2_l, RMSE_l, MAE_l), (r2_ld, RMSE_ld, MAE_ld) = self.run_id( n_folds)
			
			# Processing for merged results 
			r2_ll.extend( r2_l), RMSE_ll.extend( RMSE_l), MAE_ll.extend( MAE_l)

			# Processing for each result
			for type_id in self.type_l:
				r2_lld[type_id].extend( r2_ld[type_id])
				RMSE_lld[type_id].extend( RMSE_ld[type_id])
				MAE_lld[type_id].extend( MAE_ld[type_id])

		if self.disp:
			print 'I. Merged results:'
			print 'Mean and Std of R2 with 10 times 5-fold CV are', np.mean( r2_ll), np.std( r2_ll)
			print 'Mean and Std of RMSE with 10 times 5-fold CV are', np.mean( RMSE_ll), np.std( RMSE_ll)
			print 'Mean and Std of MAE with 10 times 5-fold CV are', np.mean( MAE_ll), np.std( MAE_ll)	

			print 'II. Each results:'
			for type_id in self.type_l:
				print "Type", type_id
				print 'Mean and Std of R2 with 10 times 5-fold CV are', np.mean( r2_ld[ type_id]), np.std( r2_ld[type_id])
				print 'Mean and Std of RMSE with 10 times 5-fold CV are', np.mean( RMSE_ld[ type_id]), np.std( RMSE_ld[type_id])
				print 'Mean and Std of MAE with 10 times 5-fold CV are', np.mean( MAE_ld[type_id]), np.std( MAE_ld[type_id])	

		pdw_l = []		
		pdw = pd.DataFrame()
		pdw['Measure'] = ['R2', 'R2', 'RMSE', 'RMSE', 'MAE', 'MAE']
		pdw['Mode'] = ['Mean', 'STD'] * 3
		pdw['Value'] = [np.mean( r2_ll), np.std( r2_ll), np.mean( RMSE_ll), np.std( RMSE_ll), np.mean( MAE_ll), np.std( MAE_ll)]
		pdw['Type'] = [ "Merged"] * 6 # checking whether different types can be included in a column vector.
		pdw_l.append( pdw)

		for type_id in self.type_l:
			pdw = pd.DataFrame()
			pdw['Measure'] = ['R2', 'R2', 'RMSE', 'RMSE', 'MAE', 'MAE']
			pdw['Mode'] = ['Mean', 'STD'] * 3
			pdw['Value'] = [np.mean( r2_ld[ type_id]), np.std( r2_ld[ type_id]), 
						np.mean( RMSE_ld[ type_id]), np.std( RMSE_ld[ type_id]), 
						np.mean( MAE_ld[ type_id]), np.std( MAE_ld[ type_id])]
			pdw['Type'] = [ type_id] * 6
			pdw_l.append( pdw)

		return pd.concat( pdw_l)

	def run4_id_iter(self, Niter = 10, n_folds = 5):

		r2_ll, RMSE_ll, MAE_ll, DAE_ll = list(), list(), list(), list()
		
		r2_lld, RMSE_lld, MAE_lld, DAE_lld = {}, {}, {}, {}
		for type_id in self.type_l:
			r2_lld[type_id] = list()
			RMSE_lld[type_id] = list()
			MAE_lld[type_id] = list()
			DAE_lld[type_id] = list()

		for ii in range( Niter):
			(r2_l, RMSE_l, MAE_l, DAE_l), (r2_ld, RMSE_ld, MAE_ld, DAE_ld) = self.run4_id( n_folds)
			
			# Processing for merged results 
			r2_ll.extend( r2_l), RMSE_ll.extend( RMSE_l), MAE_ll.extend( MAE_l), DAE_ll.extend( DAE_l)

			# Processing for each result
			for type_id in self.type_l:
				r2_lld[type_id].extend( r2_ld[type_id])
				RMSE_lld[type_id].extend( RMSE_ld[type_id])
				MAE_lld[type_id].extend( MAE_ld[type_id])
				DAE_lld[type_id].extend( DAE_ld[type_id])

		if self.disp:
			print 'I. Merged results:'
			print 'Mean and Std of R2 with 10 times 5-fold CV are', np.mean( r2_ll), np.std( r2_ll)
			print 'Mean and Std of RMSE with 10 times 5-fold CV are', np.mean( RMSE_ll), np.std( RMSE_ll)
			print 'Mean and Std of MAE with 10 times 5-fold CV are', np.mean( MAE_ll), np.std( MAE_ll)	
			print 'Mean and Std of MAE with 10 times 5-fold CV are', np.mean( DAE_ll), np.std( DAE_ll)	

			print 'II. Each results:'
			for type_id in self.type_l:
				print "Type", type_id
				print 'Mean and Std of R2 with 10 times 5-fold CV are', np.mean( r2_ld[ type_id]), np.std( r2_ld[type_id])
				print 'Mean and Std of RMSE with 10 times 5-fold CV are', np.mean( RMSE_ld[ type_id]), np.std( RMSE_ld[type_id])
				print 'Mean and Std of MAE with 10 times 5-fold CV are', np.mean( MAE_ld[type_id]), np.std( MAE_ld[type_id])
				print 'Mean and Std of DAE with 10 times 5-fold CV are', np.mean( DAE_ld[type_id]), np.std( DAE_ld[type_id])	

		pdw_l = []		
		pdw = pd.DataFrame()
		pdw['Measure'] = ['R2', 'R2', 'RMSE', 'RMSE', 'MAE', 'MAE', 'DAE', 'DAE']
		pdw['Mode'] = ['Mean', 'STD'] * 4
		pdw['Value'] = [np.mean( r2_ll), np.std( r2_ll), 
						np.mean( RMSE_ll), np.std( RMSE_ll), 
						np.mean( MAE_ll), np.std( MAE_ll),
						np.mean( DAE_ll), np.std( DAE_ll)]
		pdw['Type'] = [ "Merged"] * 8 # checking whether different types can be included in a column vector.
		pdw_l.append( pdw)

		for type_id in self.type_l:
			pdw = pd.DataFrame()
			pdw['Measure'] = ['R2', 'R2', 'RMSE', 'RMSE', 'MAE', 'MAE', 'DAE', 'DAE']
			pdw['Mode'] = ['Mean', 'STD'] * 4
			pdw['Value'] = [np.mean( r2_ld[ type_id]), np.std( r2_ld[ type_id]), 
						np.mean( RMSE_ld[ type_id]), np.std( RMSE_ld[ type_id]), 
						np.mean( MAE_ld[ type_id]), np.std( MAE_ld[ type_id]),
						np.mean( DAE_ld[ type_id]), np.std( DAE_ld[ type_id])]
			pdw['Type'] = [ type_id] * 8
			pdw_l.append( pdw)

		return pd.concat( pdw_l)


def opt_func(w, X,y): 
	U = np.linalg.norm(X.dot(w) - y)
	return U

def coefNormalized(w):
	return sum(w)-1

class Becca( object):
	def __init__(self, X_shape1, opt_func, coefNormalized):
		self.init_cond = np.ones(X_shape1)/X_shape1

		self.cons = []
		self.cons.append({'type': 'eq', 'fun': coefNormalized })

		self.bounds = []
		for i in range(0,len(self.init_cond)):
			self.bounds.append((0,100))
			
		self.opt_func = opt_func
		self.coefNormalized = coefNormalized
			
	def run(self, X, Gr):
		#Optimize ensemble

		kf = cross_validation.KFold(Gr.shape[0], n_folds=5)
		errors = []
		coeffs = []

		for train_index, test_index in kf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Gr[train_index], Gr[test_index]

			res = minimize(self.opt_func, self.init_cond, args=(X_train, y_train),
					   method='SLSQP',
					   constraints = self.cons, bounds = self.bounds)

			error = np.abs(X_test.dot(res['x'])-y_test)
			errors.append(error)

			coeff = res['x']
			coeffs.append(coeff)

		print "Average Mean-AE:", np.mean(np.mean(errors,axis=1))
		print "Average Std-AE:", np.mean(np.std(errors, axis = 1))
		print "Average Median-AE:", np.mean(np.median(errors, axis = 1))
		
		return errors, coeffs
	
def alphaNormalized(w):
	alphas= w[::2]
	return sum(alphas)-1

def betaNormalized(w):
	betas= w[1::2]
	return sum(betas)-1
	
class Becca_half( object):
	def __init__(self, X_shape1, opt_func, alphaNormalized, betaNormalized):
		
		nhalf = X_shape1/2
		self.init_cond = np.ones(X_shape1)/nhalf
		
		# Add constraints
		self.cons = []
		self.cons.append({'type': 'eq', 'fun': alphaNormalized })
		self.cons.append({'type': 'eq', 'fun': betaNormalized })               

		self.bounds = []
		for i in range(0,len(self.init_cond)):
			self.bounds.append((0,100))
			
		self.opt_func = opt_func
		self.coefNormalized = coefNormalized
			
	def run(self, X, Gr):
		#Optimize ensemble

		kf = cross_validation.KFold(Gr.shape[0], n_folds= 5)
		errors = []
		coeffs = []

		for train_index, test_index in kf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Gr[train_index], Gr[test_index]

			res = minimize(self.opt_func, self.init_cond, args=(X_train, y_train),
					   method='SLSQP',
					   constraints = self.cons, bounds = self.bounds)
			
			error = np.abs(X_test.dot(res['x'])-y_test)
			
			errors.append(error)

			coeff = res['x']
			coeffs.append(coeff)

		print "Average Mean-AE:", np.mean(np.mean(errors,axis=1))
		print "Average Std-AE:", np.mean(np.std(errors, axis = 1))
		print "Average Median-AE:", np.mean(np.median(errors, axis = 1))
		
		return errors, coeffs
	
	def run_base(self, X, Gr):
		#Optimize ensemble

		kf = cross_validation.KFold(Gr.shape[0], n_folds= 5)
		errors = []
		coeffs = []

		for train_index, test_index in kf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Gr[train_index], Gr[test_index]

			res = {}
			res['x'] = np.array( [0, 0, 1, 1.0]).T
			error = np.abs(X_test.dot( res['x'])-y_test)
			print error.shape
			
			errors.append(error)

			coeff = res['x']
			coeffs.append(coeff)

		print "Average Mean-AE:", np.mean(np.mean(errors,axis=1))
		print "Average Std-AE:", np.mean(np.std(errors, axis = 1))
		print "Average Median-AE:", np.mean(np.median(errors, axis = 1))
		
		return errors, coeffs
	
# Becca codes

def convert_GtoE( G):

	kcal_to_eV = 0.0433641
	z = 2.0
	V_to_mV = 1000

	E = G * -kcal_to_eV * V_to_mV / z

	return E

def opt_func(w, X,y): 
	U = np.linalg.norm(X.dot(w) - y)
	return U

def coefNormalized(w):
	return sum(w)-1

class Becca( object):
	def __init__(self, X_shape1, 
		opt_func = opt_func, 
		coefNormalized = coefNormalized):

		self.init_cond = np.ones(X_shape1)/X_shape1

		self.cons = []
		self.cons.append({'type': 'eq', 'fun': coefNormalized })

		self.bounds = []
		for i in range(0,len(self.init_cond)):
			self.bounds.append((0,100))
			
		self.opt_func = opt_func
		self.coefNormalized = coefNormalized
			
	def run(self, X, Gr):
		#Optimize ensemble

		kf = cross_validation.KFold(Gr.shape[0], n_folds=5)
		errors = []
		coeffs = []

		for train_index, test_index in kf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Gr[train_index], Gr[test_index]

			res = minimize(self.opt_func, self.init_cond, args=(X_train, y_train),
					   method='SLSQP',
					   constraints = self.cons, bounds = self.bounds)

			error = np.abs(X_test.dot(res['x'])-y_test)
			errors.append(error)

			coeff = res['x']
			coeffs.append(coeff)

		print "Average Mean-AE:", np.mean(np.mean(errors,axis=1))
		print "Average Std-AE:", np.mean(np.std(errors, axis = 1))
		print "Average Median-AE:", np.mean(np.median(errors, axis = 1))
		
		return errors, coeffs
	
def alphaNormalized(w):
	alphas= w[::2]
	return sum(alphas)-1

def betaNormalized(w):
	betas= w[1::2]
	return sum(betas)-1
	
class Becca_half( object):
	def __init__(self, X_shape1, 
		opt_func = opt_func, 
		alphaNormalized = alphaNormalized, 
		betaNormalized = betaNormalized):
		
		nhalf = X_shape1/2
		self.init_cond = np.ones(X_shape1)/nhalf
		
		# Add constraints
		self.cons = []
		self.cons.append({'type': 'eq', 'fun': alphaNormalized })
		self.cons.append({'type': 'eq', 'fun': betaNormalized })               

		self.bounds = []
		for i in range(0,len(self.init_cond)):
			self.bounds.append((0,100))
			
		self.opt_func = opt_func
		self.coefNormalized = coefNormalized
			
	def run(self, X, Gr):
		#Optimize ensemble

		kf = cross_validation.KFold(Gr.shape[0], n_folds= 5)
		errors = []
		coeffs = []

		for train_index, test_index in kf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Gr[train_index], Gr[test_index]

			res = minimize(self.opt_func, self.init_cond, args=(X_train, y_train),
					   method='SLSQP',
					   constraints = self.cons, bounds = self.bounds)
			
			error = np.abs(X_test.dot(res['x'])-y_test)
			
			errors.append(error)

			coeff = res['x']
			coeffs.append(coeff)

		print "Average Mean-AE:", np.mean(np.mean(errors,axis=1))
		print "Average Std-AE:", np.mean(np.std(errors, axis = 1))
		print "Average Median-AE:", np.mean(np.median(errors, axis = 1))
		
		return errors, coeffs

	def run_base(self, X, Gr):
		#Optimize ensemble

		kf = cross_validation.KFold(Gr.shape[0], n_folds= 5)
		errors = []
		coeffs = []

		for train_index, test_index in kf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Gr[train_index], Gr[test_index]

			res = {}
			res['x'] = np.array( [0, 0, 1.0, 1.0]).T
			error = np.abs(X_test.dot( res['x'])-y_test)
			# print error.shape
			
			errors.append(error)

			coeff = res['x']
			coeffs.append(coeff)

		print "Average Mean-AE:", np.mean(np.mean(errors,axis=1))
		print "Average Std-AE:", np.mean(np.std(errors, axis = 1))
		print "Average Median-AE:", np.mean(np.median(errors, axis = 1))
		
		return errors, coeffs

	def run_alpha(self, X, Gr, alpha = convert_GtoE(1.0)):
		"""
		The results will be multiplied by alpha so that 
		the value will be equivalent to the corresponding target value.
		"""

		kf = cross_validation.KFold(Gr.shape[0], n_folds= 5)
		errors = []
		coeffs = []

		for train_index, test_index in kf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Gr[train_index], Gr[test_index]

			res = minimize(self.opt_func, self.init_cond, args=(X_train, y_train),
					   method='SLSQP',
					   constraints = self.cons, bounds = self.bounds)
			
			error = np.abs(alpha * X_test.dot(res['x'])-y_test)
			
			errors.append(error)

			coeff = res['x']
			coeffs.append(coeff)

		print "Average Mean-AE:", np.mean(np.mean(errors,axis=1))
		print "Average Std-AE:", np.mean(np.std(errors, axis = 1))
		print "Average Median-AE:", np.mean(np.median(errors, axis = 1))
		
		return errors, coeffs

	def run_alpha_base(self, X, Gr, alpha = convert_GtoE(1.0)):
		"""
		The results will be multiplied by alpha so that 
		the value will be equivalent to the corresponding target value.
		"""

		kf = cross_validation.KFold(Gr.shape[0], n_folds= 5)
		errors = []
		coeffs = []

		for train_index, test_index in kf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Gr[train_index], Gr[test_index]

			res = {}
			res['x'] = np.array( [0, 0, 1.0, 1.0]).T
			
			error = np.abs(alpha * X_test.dot(res['x'])-y_test)
			
			errors.append(error)

			coeff = res['x']
			coeffs.append(coeff)

		print "Average Mean-AE:", np.mean(np.mean(errors,axis=1))
		print "Average Std-AE:", np.mean(np.std(errors, axis = 1))
		print "Average Median-AE:", np.mean(np.median(errors, axis = 1))
		
		return errors, coeffs

	

def get_xM4_yV( pdr, Em_type = "Em_in"):
	"""
	read saved arrays in each element. 
	"""
	xM4_str = pdr.xM4 
	xM4_l = [ eval( x) for x in xM4_str]
	xM = np.mat( xM4_l)
	
	yV = np.mat( pdr[Em_type]).T
	
	return xM, yV