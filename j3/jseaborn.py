# this is extension of seaborn by James for machine learning.
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The following libraries are designed by James
import j3.jgrid as jgrid

def set_pdi_d( pdi_d, method, grid_scores_):
	pdi_d[ method] = pd.DataFrame()
	#print type( val.cv_validation_scores)

	for val in grid_scores_:
		r2_a = val.cv_validation_scores

		pdx = pd.DataFrame()
		pdx["Method"] = [ method]
		pdx["alpha"] = [ val.parameters["alpha"]]    
		pdx["mean(r2)"] = [np.mean( r2_a)]
		pdx["std(r2)"] = [np.std( r2_a)]
		pdx["r2_a"] = [r2_a]

		pdi_d[ method] = pdi_d[ method].append( pdx, ignore_index = True)

	return pdi_d

def set_pdi_d_full( pdi_d, method, xM_l, yV):
	xM = np.concatenate( xM_l, axis = 1)
	gs = jgrid.gs_Ridge( xM, yV, (-3, 2, 10), n_folds=20)
	# gs.grid_scores_

	set_pdi_d( pdi_d, method, gs.grid_scores_)
	pdi_d[ method].plot( kind ='line', x = 'alpha', y = 'mean(r2)', yerr = 'std(r2)', logx = True)
	plt.ylabel( r"E[$r^2$]")

	return pdi_d[method]

def _pdi_gs_r0( method, grid_scores_, expension = False):
	pdi = pd.DataFrame()
	#print type( val.cv_validation_scores)

	for val in grid_scores_:
		r2_a = val.cv_validation_scores

		pdx = pd.DataFrame()
		if expension:
			pdx["Method"] = [ method] * r2_a.shape[0]
			pdx["alpha"] = [ val.parameters["alpha"]] * r2_a.shape[0]    
			pdx["unit"] = range( r2_a.shape[0])
			pdx["r2"] = r2_a            
		else:
			pdx["Method"] = [ method]
			pdx["alpha"] = [ val.parameters["alpha"]]    
			pdx["mean(r2)"] = [np.mean( r2_a)]
			pdx["std(r2)"] = [np.std( r2_a)]
			pdx["r2_a"] = [r2_a]

		pdi = pdi.append( pdx, ignore_index = True)

	return pdi

def _pdi_gs_full_r0( method, xM_l, yV, expension = False):
	xM = np.concatenate( xM_l, axis = 1)
	gs = jgrid.gs_Ridge( xM, yV, (-3, 2, 10), n_folds=20)
	# gs.grid_scores_

	if expension:
		pdi = pdi_gs( method, gs.grid_scores_, expension = expension)
	else:
		pdi = pdi_gs( pdi_d, method, gs.grid_scores_)
		pdi.plot( kind ='line', x = 'alpha', y = 'mean(r2)', yerr = 'std(r2)', logx = True)
		plt.ylabel( r"E[$r^2$]")

	return pdi

def pdi_gs( method, grid_scores_, expension = False):
	pdi = pd.DataFrame()
	#print type( val.cv_validation_scores)

	for val in grid_scores_:
		r2_a = val.cv_validation_scores

		pdx = pd.DataFrame()
		if expension:
			pdx["Method"] = [ method] * r2_a.shape[0]
			pdx["alpha"] = [ val.parameters["alpha"]] * r2_a.shape[0]    
			pdx["unit"] = range( r2_a.shape[0])
			pdx["r2"] = r2_a            
		else:
			pdx["Method"] = [ method]
			pdx["alpha"] = [ val.parameters["alpha"]]    
			pdx["mean(r2)"] = [np.mean( r2_a)]
			pdx["std(r2)"] = [np.std( r2_a)]
			pdx["r2_a"] = [r2_a]

		pdi = pdi.append( pdx, ignore_index = True)

	return pdi

def pdi_gs_full( method, xM_l, yV, X_concat = None, mode = "Ridge", expension = False, n_folds=20):
	if mode == "Ridge":
		xM = np.concatenate( xM_l, axis = 1)
		gs = jgrid.gs_Ridge( xM, yV, (-3, 2, 10), n_folds=n_folds)
	elif mode == "BIKE_Ridge":
		# print "BIKE_Ridge mode is working now."
		A_l = xM_l
		gs = jgrid.gs_BIKE_Ridge( A_l, yV, alphas_log=(-3, 2, 10), X_concat = X_concat, n_folds=n_folds)
	else:
		print( "Mode {} is not supported.".format( mode))
	# gs.grid_scores_

	if expension:
		pdi = pdi_gs( method, gs.grid_scores_, expension = expension)
	else:
		pdi = pdi_gs( method, gs.grid_scores_)
		pdi.plot( kind ='line', x = 'alpha', y = 'mean(r2)', yerr = 'std(r2)', logx = True)
		plt.ylabel( r"E[$r^2$]")

	return pdi