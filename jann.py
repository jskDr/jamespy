"""
Python code for artificial neural networks
"""

def gen_input_files_valid_overfit( At, yt, Av, yv):
	"""
	Validation is also considerd.
	At and yt are for training while Av, yv are for validation.
	Input files of ann_in.data and ann_run.dat are gerneated.
	The files are used in ann_aq.c (./ann_aq) 
	* Input: At, Av is matrix, yt, yv is vector
	"""

	print "For overfitting testing, a validation file includes desired value."

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
			f.write( "\n{}\n".format( yv[ix,0]))
		print("ann_run.data with {0} sets, {1} inputs is saved with desired values".format( no_of_set, no_of_input))