# using PyCall
# pyinitialize("python3")
"""
# For reload - jjulia can be changed for appropriate type
jjulia = pyimport(:jjulia)

# This can be used for import and reload both.
# If @pyimport is used, reload is not workinng since lib names are assigned as constant variales. 
jjulia = pywrap(PyObject(ccall(pysym(:PyImport_ReloadModule), PyPtr, (PyPtr,), pyimport("jjulia"))))
"""

import numpy as np
from sklearn import linear_model

def hello( name):
	print("hi {}".format( name))
	print("This is James")
	print("Julia calls a python function.")
	print("Now reload module is working.")

def _regression_r0( X, y):
	print X
	print X.shape
	print y
	print y.shape

	xM = np.mat( X)
	yV = np.mat( y).T

	w = np.linalg.pinv( xM) * yV

	return np.array(w)

def regression(X, y):
	clf = linear_model.LinearRegression()
	clf.fit(X, y)
	return clf.coef_


