"""
this is a file to compile pyx files and copy to the associated directory 
which is jamespyx_linux and jamespyx_mac
"""

import platform
import os
from os.path import expanduser

def run(): 
	cur_path = os.getcwd()

	# This is the platform dependent part.
	if platform.system() == "Linux":
		home = expanduser("~")
		os.chdir( home + '/Dropbox/Aspuru-Guzik/python_lab/jamespy/')
	elif platform.system() == "Darwin":
		home = expanduser("~")
		os.chdir( home + '/Dropbox/Aspuru-Guzik/python_lab/jamespy/')

	if os.path.isfile('jpyx.so'):
		os.remove('jpyx.so')
	os.system('python setup_pyx.py build_ext --inplace')

	if platform.system() == "Linux":
		os.system('mv jpyx.so ../jamespyx_linux/.')
	elif platform.system() == "Darwin":
		os.system('mv jpyx.so ../jamespyx_mac/.')

	os.chdir(cur_path)
	os.getcwd()

	print "pyx code compilation is completed."

if __name__ == '__main__':
	run()