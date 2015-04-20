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
	jchem.estimate_accuracy( yEv, yEv_calc, disp = disp)
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

def mlr_val( RM, yE, disp = True, graph = True):
	clf = linear_model.LinearRegression()
	clf.fit( RM[::2,:], yE[::2,0])

	print 'Training result'
	mlr_show( clf, RM[::2, :], yE[::2, 0], disp = disp, graph = graph)

	print 'Validation result'
	mlr_show( clf, RM[1::2, :], yE[1::2, 0], disp = disp, graph = graph)

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
	RMv, yEv = get_valid_mode_data( RM, yE, rate = rate, more_train = True, center = None):
	jchem.gen_input_files_valid( RMv, yEv, RM)

def ann_val_post( yE, disp = True, graph = True):
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