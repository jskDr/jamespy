"""
some utility which I made.
Editor - Sungjin Kim, 2015-4-17
"""

#Common library
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
#import subprocess
import pandas as pd
import itertools
import random

#My personal library
import jchem


def sleast( a = '1000', ln = 10):
	"It returns 0 filled string with the length of ln."
	if ln > len(a):
		return '0'*(ln - len(a)) + a
	else:
		return a[-ln:]

def int_bp( b_ch):
	"map '0' --> -1, '1' --> -1"
	b_int = int( b_ch)
	return 1 - 2 * b_int

def prange( pat, st, ed, ic=1):
	ar = []
	for ii in range( st, ed, ic):
		ar.extend( map( lambda jj: ii + jj, pat))

	return filter( lambda x: x < ed, ar)

import time

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

def mlr( RM, yE, disp = True, graph = True):
	clf = linear_model.LinearRegression()
	clf.fit( RM, yE)
	mlr_show( clf, RM, yE, disp = disp, graph = graph)

def ann_pre( RM, yE, disp = True, graph = True):
	"""
	In ann case, pre and post processing are used
	while in mlr case, all processing is completed by one function (mlr).
	ann processing will be performed by shell command
	"""
	jchem.gen_input_files_valid( RM, yE, RM)

def ann_post( yv, disp = True, graph = True):
	"""
	After ann_pre and shell command, ann_post can be used.
	"""
	df_ann = pd.read_csv( 'ann_out.csv')
	yv_ann = np.mat( df_ann['out'].tolist()).T
	
	r_sqr, RMSE = ann_show( yv, yv_ann, disp = disp, graph = graph)

	return r_sqr, RMSE

def ann_show( yEv, yEv_calc, disp = True, graph = True):
	r_sqr, RMSE = jchem.estimate_accuracy( yEv, yEv_calc, disp = disp)
	if graph:
		plt.scatter( yEv.tolist(), yEv_calc.tolist())
		ax = plt.gca()
		lims = [
			np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
			np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
		]
		# now plot both limits against eachother
		#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		ax.plot(lims, lims, '-', color = 'pink')
		plt.xlabel('Target')
		plt.ylabel('Prediction')
		plt.show()
	return r_sqr, RMSE

def mlr_show( clf, RMv, yEv, disp = True, graph = True):
	yEv_calc = clf.predict( RMv)
	r_sqr, RMSE = jchem.estimate_accuracy( yEv, yEv_calc, disp = disp)
	if graph:
		plt.scatter( yEv.tolist(), yEv_calc.tolist())
		ax = plt.gca()
		lims = [
			np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
			np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
		]
		# now plot both limits against eachother
		#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		ax.plot(lims, lims, '-', color = 'pink')
		plt.xlabel('Target')
		plt.ylabel('Prediction')
		plt.show()

	return r_sqr, RMSE

def _mlr_val_r0( RM, yE, disp = True, graph = True):
	clf = linear_model.LinearRegression()
	clf.fit( RM[::2,:], yE[::2,0])

	print 'Training result'
	mlr_show( clf, RM[::2, :], yE[::2, 0], disp = disp, graph = graph)

	print 'Validation result'
	mlr_show( clf, RM[1::2, :], yE[1::2, 0], disp = disp, graph = graph)

def mlr_val( RM, yE, disp = True, graph = True, rate = 2, more_train = True, center = None):
	"""
	Validation is peformed as much as the given ratio.
	"""
	RMt, yEt, RMv, yEv = jchem.get_valid_mode_data( RM, yE, rate = rate, more_train = more_train, center = center)

	clf = linear_model.LinearRegression()	
	clf.fit( RMt, yEt)

	print 'Training result'
	mlr_show( clf, RMt, yEt, disp = disp, graph = graph)

	print 'Validation result'
	r_sqr, RMSE = mlr_show( clf, RMv, yEv, disp = disp, graph = graph)

	return r_sqr, RMSE

def mlr_val_ridge( RM, yE, rate = 2, more_train = True, center = None, alpha = 0.5, disp = True, graph = True):
	"""
	Validation is peformed as much as the given ratio.
	"""
	RMt, yEt, RMv, yEv = jchem.get_valid_mode_data( RM, yE, rate = rate, more_train = more_train, center = center)

	print "Ridge: alpha = {}".format( alpha)
	clf = linear_model.Ridge( alpha = alpha)	
	clf.fit( RMt, yEt)

	print 'Training result'
	mlr_show( clf, RMt, yEt, disp = disp, graph = graph)

	print 'Validation result'
	r_sqr, RMSE = mlr_show( clf, RMv, yEv, disp = disp, graph = graph)

	return r_sqr, RMSE

def mlr_val_avg_2( RM, yE, disp = False, graph = False):
	"""
	Validation is peformed as much as the given ratio.
	"""
	r_sqr_list, RMSE_list = [], []
	vseq_list = []
	org_seq = range( len( yE))
	for	v_seq in itertools.combinations( org_seq, 2):
		t_seq = filter( lambda x: x not in v_seq, org_seq)

		RMt, yEt = RM[ t_seq, :], yE[ t_seq, 0]
		RMv, yEv = RM[ v_seq, :], yE[ v_seq, 0]

		#RMt, yEt, RMv, yEv = jchem.get_valid_mode_data( RM, yE, rate = rate, more_train = more_train, center = center)	

		clf = linear_model.LinearRegression()	
		clf.fit( RMt, yEt)

		#print 'Training result'
		mlr_show( clf, RMt, yEt, disp = disp, graph = graph)

		#print 'Validation result'
		r_sqr, RMSE = mlr_show( clf, RMv, yEv, disp = disp, graph = graph)

		"""
		#This is blocked since vseq_list is returned.
		if r_sqr < 0:
			print 'v_seq:', v_seq, '--> r_sqr = ', r_sqr
		"""

		r_sqr_list.append( r_sqr)
		RMSE_list.append( RMSE)
		vseq_list.append( v_seq)

	print "average r_sqr = {0}, average RMSE = {1}".format( np.average( r_sqr_list), np.average( RMSE_list))

	return r_sqr_list, RMSE_list, v_seq

def mlr_val_vseq( RM, yE, v_seq, disp = True, graph = True):
	"""
	Validation is peformed using vseq indexed values.
	"""
	org_seq = range( len( yE))
	t_seq = filter( lambda x: x not in v_seq, org_seq)

	RMt, yEt = RM[ t_seq, :], yE[ t_seq, 0]
	RMv, yEv = RM[ v_seq, :], yE[ v_seq, 0]

	clf = linear_model.LinearRegression()	
	clf.fit( RMt, yEt)

	if disp: print 'Training result'
	mlr_show( clf, RMt, yEt, disp = disp, graph = graph)

	if disp: print 'Validation result'
	r_sqr, RMSE = mlr_show( clf, RMv, yEv, disp = disp, graph = graph)

	#if r_sqr < 0:
	#	print 'v_seq:', v_seq, '--> r_sqr = ', r_sqr

	return r_sqr, RMSE

def mlr_val_vseq_rand(RM, yE, disp = True, graph = True, rate = 5):
	"""
	Validation is peformed using vseq indexed values.
	vseq is randmly selected with respect to rate. 
	"""
	vseq = choose( len( yE), int(len( yE) / rate));

	r_sqr, RMSE = mlr_val_vseq( RM, yE, vseq, disp = disp, graph = graph)

	return r_sqr, RMSE

def mlr_val_vseq_ridge_rand( RM, yE, alpha = .5, rate = 2, disp = True, graph = True):
	vseq = choose( len( yE), int(len( yE) / rate));

	r_sqr, RMSE = mlr_val_vseq_ridge( RM, yE, vseq, alpha = alpha, disp = disp, graph = graph)

	return r_sqr, RMSE

def mlr_val_vseq_ridge( RM, yE, v_seq, alpha = .5, disp = True, graph = True):
	"""
	Validation is peformed using vseq indexed values.
	"""
	org_seq = range( len( yE))
	t_seq = filter( lambda x: x not in v_seq, org_seq)

	RMt, yEt = RM[ t_seq, :], yE[ t_seq, 0]
	RMv, yEv = RM[ v_seq, :], yE[ v_seq, 0]

	clf = linear_model.Ridge( alpha = alpha)
	clf.fit( RMt, yEt)

	if disp: print 'Training result'
	mlr_show( clf, RMt, yEt, disp = disp, graph = graph)

	if disp: print 'Validation result'
	r_sqr, RMSE = mlr_show( clf, RMv, yEv, disp = disp, graph = graph)

	#if r_sqr < 0:
	#	print 'v_seq:', v_seq, '--> r_sqr = ', r_sqr

	return r_sqr, RMSE

def _ann_val_pre_r0( RM, yE, disp = True, graph = True):
	"""
	In ann case, pre and post processing are used
	while in mlr case, all processing is completed by one function (mlr).
	ann processing will be performed by shell command
	"""
	jchem.gen_input_files_valid( RM[::2,:], yE[::2,0], RM)

def ann_val_pre( RM, yE, rate = 2, more_train = True, center = None):
	"""
	In ann case, pre and post processing are used
	while in mlr case, all processing is completed by one function (mlr).
	ann processing will be performed by shell command

	Now, any percentage of validation will be possible. 
	Later, random selection will be included, while currently 
	deterministic selection is applied. 
	"""
	RMt, yEt, RMv, yEv = jchem.get_valid_mode_data( RM, yE, rate = rate, more_train = more_train, center = center)
	jchem.gen_input_files_valid( RMt, yEt, RM)

def _ann_val_post_r0( yE, disp = True, graph = True):
	"""
	After ann_pre and shell command, ann_post can be used.
	"""
	df_ann = pd.read_csv( 'ann_out.csv')
	yv_ann = np.mat( df_ann['out'].tolist()).T
	
	print 'Trainig result'
	ann_show( yE[::2,0], yv_ann[::2,0], disp = disp, graph = graph)

	print 'Validation result'
	r_sqr, RMSE = ann_show( yE[1::2,0], yv_ann[1::2,0], disp = disp, graph = graph)

	return r_sqr, RMSE

def ann_val_post( yE, disp = True, graph = True, rate = 2, more_train = True, center = None):
	"""
	After ann_pre and shell command, ann_post can be used.
	"""
	df_ann = pd.read_csv( 'ann_out.csv')
	yE_c = np.mat( df_ann['out'].tolist()).T

	yEt, yEt_c, yEv, yEv_c = jchem.get_valid_mode_data( yE, yE_c, rate = rate, more_train = more_train, center = center)
	
	print 'Trainig result'
	ann_show( yEt, yEt_c, disp = disp, graph = graph)

	print 'Validation result'
	r_sqr, RMSE = ann_show( yEv, yEv_c, disp = disp, graph = graph)

	return r_sqr, RMSE

def writeparam_txt( fname = 'param.txt', dic = {"num_neurons_hidden": 4, "desired_error": 0.00001}):
	"save param.txt with dictionary"
	
	with open(fname, 'w') as f:
		print  "Saving", fname
		for di in dic:
			f.write("{} {}\n".format( di, dic[di]))

def choose(N, n):
	"""
	Returns n randomly chosen values between 0 to N-1.
	"""
	x = range( N)
	n_list = []

	for ii in range( n):
		xi = random.choice( x)
		n_list.append( xi)
		x.remove( xi)

	return n_list