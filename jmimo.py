from __future__ import print_function
# I started to use __future__ so as to be compatible with Python3

import numpy as np
from sklearn import linear_model

# To improve the speed, I using pyx. 
import jpyx
import jutil

def mld( r_l, mod_l = [-0.70710678, 0.70710678]):
	"""
	maximum likelihood detection
	
	r_l: received signals after reception processing
	mod_l: list of all modulation signals
		BPSK: [-0.70710678, 0.70710678]
	
	return the demodulated signals (0, 1, ...)
	"""
	sd_l = list() # store demodulated signal
	for r in r_l:
		dist = list() #Store distance
		for m in mod_l:
			d = np.power( np.abs( r - m), 2)
			dist.append( d)
		sd = np.argmin( dist)
		sd_l.append( sd)
	return np.array( sd_l)

def calc_BER( r_l, x_l):
	"""
	calculate bit error rate (BER)
	r_l: demodulated signals (ndarray, 1D)
	x_l: transmitted signals (ndarray, 1D)
	"""
	err_l = r_l - x_l
	errs = np.where( err_l != 0)[0]
	# print 'err_l =', err_l 
	# print 'errs =', errs
	Nerr = len(np.where( err_l != 0)[0])
	return float( Nerr) / len( err_l), Nerr 

def db2var( SNRdB):
	return np.power( 10.0, SNRdB / 10.0)

def gen_BPSK(Nx, Nt):
	"""
	Generate BPSK modulated signals
	"""
	BPSK = np.array( [1, -1]) / np.sqrt( 2.0)
	s_a = np.random.randint( 0, 2, Nx * Nt)
	x_flat_a = BPSK[ s_a]
	x_a = np.reshape( x_flat_a, (Nx, Nt))

	return BPSK, s_a, x_flat_a, x_a

def gen_H( Nr, Nt):
	return np.random.randn( Nr, Nt)

def gen_Rx( Nr, Nx, SNR, H_a, x_a):
	"""
	The received signals are modeled.
	"""
	n_a = np.random.randn( Nr, Nx) / np.sqrt( SNR)
	y_a = np.dot( H_a, x_a.T) + n_a	

	return y_a

class MIMO(object):
	"""
	Modeling for a MIMO wireless communication system.
	"""
	def __init__(self, Nt = 2, Nr = 4, Nx = 10, SNRdB = 10):

		self.set_param( (Nt, Nr, Nx, SNRdB))

	def set_param( self, param_NtNrNxSNRdB):

		Nt, Nr, Nx, SNRdB	 = param_NtNrNxSNRdB
		
		# The antenna configuration is conducted.
		self.Nt = Nt
		self.Nr = Nr
		# No of streams is fixed.
		self.Nx = Nx

		# Initial SNR is defined
		self.SNRdB = SNRdB
		self.SNR = db2var(SNRdB)

	def _gen_BPSK_r0(self):
		"""
		Generate BPSK modulated signals
		"""
		self.BPSK = np.array( [1, -1]) / np.sqrt( 2.0)
		self.s_a = np.random.randint( 0, 2, self.Nx * self.Nt)
		self.x_flat_a = self.BPSK[ self.s_a]
		self.x_a = np.reshape( self.x_flat_a, (self.Nx, self.Nt))

	def gen_BPSK( self):
		"""
		Generate BPSK signals using global function gen_BPSK().
		This function will be used to generate pilot signal as well. 
		"""
		self.BPSK, self.s_a, self.x_flat_a, self.x_a = gen_BPSK( self.Nx, self.Nt)

	def gen_H(self):
		"""
		The MIMO channel is generated.
		"""
		self.H_a = gen_H( self.Nr, self.Nt)

	def _gen_Rx_r0(self):
		"""
		The received signals are modeled.
		"""
		self.n_a = np.random.randn( self.Nr, self.Nx) / np.sqrt( self.SNR)
		self.y_a = np.dot( self.H_a, self.x_a.T) + self.n_a

	def gen_Rx(self):
		"""
		The received signals are modeled.
		"""
		self.y_a = gen_Rx( self.Nr, self.Nx, self.SNR, self.H_a, self.x_a)

	def gen_WR_ideal(self): 
		"""
		The reception process with ideal channel estimation 
		is conducted.
		"""
		self.W_a = np.linalg.pinv( self.H_a)
		# The reception signal vector is transposed.

		self.gen_Decoding()

	def gen_WR_pilot(self, pilot_SNRdB):

		"""
		The reception process with pilot channel estimation
		is conducted.
		Pilot will be transmitted through random information channel.
		"""
		pilot_SNR = db2var(pilot_SNRdB)
		N_a = np.random.randn( *self.H_a.shape) / np.sqrt( pilot_SNR)
		Hp_a = self.H_a + N_a
		self.W_a = np.linalg.pinv( Hp_a)

		self.gen_Decoding()

	def gen_WR_pilot_channel(self, pilot_SNRdB):

		"""
		The reception process with pilot channel estimation
		is conducted.
		"""
		Npilot = 10
		SNRpilot = db2var( pilot_SNRdB)

		BPSK, s_a, x_flat_a, x_a = gen_BPSK( Npilot, self.Nt)
		# H_a = gen_H( self.Nr, self.Nt)
		# H_a = self.H_a
		y_a = gen_Rx( self.Nr, Npilot, SNRpilot, self.H_a, x_a)

		yT_a = y_a.T

		# print( x_a.shape, yT_a.shape)

		lm = linear_model.LinearRegression()
		lm.fit( yT_a, x_a)
		"""
		Power normalization should be considered 
		unless it is multiplied with both sinal and noise. 
		In this case, MMSE weight is calculated while
		pinv() obtain ZF filter.  
		"""
		self.W_a = lm.coef_

		# print( "np.dot( W_a, H_a) =", np.dot( self.W_a, self.H_a))

		self.gen_Decoding()

	def gen_WR_pilot_ch(self, pilot_SNRdB, alpha = 0):

		"""
		The reception process with pilot channel estimation
		is conducted.
		"""
		Npilot = 10
		SNRpilot = db2var( pilot_SNRdB)

		BPSK, s_a, x_flat_a, x_a = gen_BPSK( Npilot, self.Nt)
		# H_a = gen_H( self.Nr, self.Nt)
		# H_a = self.H_a
		y_a = gen_Rx( self.Nr, Npilot, SNRpilot, self.H_a, x_a)

		yT_a = y_a.T

		# print( x_a.shape, yT_a.shape)

		lm = linear_model.Ridge( alpha)
		lm.fit( yT_a, x_a)
		self.W_a = lm.coef_

		# print( "np.dot( W_a, H_a) =", np.dot( self.W_a, self.H_a))

		self.gen_Decoding()

	def gen_WR( self, pilot_SNRdB = None):
		if pilot_SNRdB:
			gen_WR_pilot( pilot_SNRdB)
		else:
			gen_WR_ideal()

	def gen_Decoding(self): 
		"""
		The reception process is conducted.
		"""
		self.rT_a = np.dot( self.W_a, self.y_a)

		self.r_flat_a = self.rT_a.T.flatten()
		#print( "type( self.r_flat_a), type( self.BPSK)")
		#print( type( self.r_flat_a), type( self.BPSK))
		# self.sd_a = jpyx.mld( self.r_flat_a, self.BPSK)
		self.sd_a = jpyx.mld_fast( self.r_flat_a, self.BPSK)
		self.BER, self.Nerr = calc_BER( self.s_a, self.sd_a)


	def run_ideal( self, param_NtNrNxSNRdB = None, Nloop = 10, disp = False):
		"""
		A system is run from the transmitter to the receiver.

		"""
		return self.run_pilot( param_NtNrNxSNRdB = param_NtNrNxSNRdB, Nloop = Nloop, disp = disp)


	def run_pilot( self, pilot_SNRdB = None, param_NtNrNxSNRdB = None, Nloop = 10, disp = False):
		"""
		A system is run from the transmitter to the receiver.

		"""
		if param_NtNrNxSNRdB:
			self.set_param( param_NtNrNxSNRdB)

		self.gen_BPSK()

		BER_l = list()
		Nerr_total = 0
		for nloop in range( Nloop):
			self.gen_H()
			self.gen_Rx()

			if pilot_SNRdB:
				self.gen_WR_pilot( pilot_SNRdB)
			else: 
				self.gen_WR_ideal()

			BER_l.append( self.BER)
			Nerr_total += self.Nerr

		self.BER = np.mean( BER_l)

		if disp:
			Ntot = self.Nt * self.Nx * Nloop
			print( "BER is {} with {}/{} errors at {} SNRdB ".format( self.BER, Nerr_total, Ntot, self.SNRdB))

		return self.BER

	def run_pilot_channel( self, pilot_SNRdB = None, param_NtNrNxSNRdB = None, Nloop = 10, disp = False):
		"""
		A system is run from the transmitter to the receiver.

		"""
		if param_NtNrNxSNRdB:
			self.set_param( param_NtNrNxSNRdB)

		self.gen_BPSK()

		BER_l = list()
		Nerr_total = 0
		for nloop in range( Nloop):
			self.gen_H()
			self.gen_Rx()

			if pilot_SNRdB:
				# self.gen_WR_pilot( pilot_SNRdB)
				self.gen_WR_pilot_channel( pilot_SNRdB)
				# self.gen_WR_pilot_ch( pilot_SNRdB, alpha)
			else: 
				self.gen_WR_ideal()

			BER_l.append( self.BER)
			Nerr_total += self.Nerr

		self.BER = np.mean( BER_l)

		if disp:
			Ntot = self.Nt * self.Nx * Nloop
			print( "BER is {} with {}/{} errors at {} SNRdB ".format( self.BER, Nerr_total, Ntot, self.SNRdB))

		return self.BER

	def run_pilot_ch( self, pilot_SNRdB = None, param_NtNrNxSNRdB = None, Nloop = 10, alpha = 0, disp = False):
		"""
		A system is run from the transmitter to the receiver.

		"""
		if param_NtNrNxSNRdB:
			self.set_param( param_NtNrNxSNRdB)

		self.gen_BPSK()

		BER_l = list()
		Nerr_total = 0
		for nloop in range( Nloop):
			self.gen_H()
			self.gen_Rx()

			if pilot_SNRdB:
				# self.gen_WR_pilot( pilot_SNRdB)
				# self.gen_WR_pilot_channel( pilot_SNRdB)
				self.gen_WR_pilot_ch( pilot_SNRdB, alpha)
			else: 
				self.gen_WR_ideal()

			BER_l.append( self.BER)
			Nerr_total += self.Nerr

		self.BER = np.mean( BER_l)

		if disp:
			Ntot = self.Nt * self.Nx * Nloop
			print( "BER is {} with {}/{} errors at {} SNRdB ".format( self.BER, Nerr_total, Ntot, self.SNRdB))

		return self.BER

def get_BER( SNRdB_l = [5,6,7], param_NtNrNx = (2,4,100), Nloop = 1000, pilot_SNRdB = None):
	BER_pilot = list()

	Nt, Nr, Nx = param_NtNrNx	
	for SNRdB in SNRdB_l:
		ber = MIMO().run_pilot( pilot_SNRdB = pilot_SNRdB, 
			param_NtNrNxSNRdB =(Nt, Nr, Nx, SNRdB), Nloop = Nloop, disp = True)
		BER_pilot.append( ber)

	# print( "List of average BERs =", BER_pilot)

	return BER_pilot

def get_BER_pilot_ch( SNRdB_l = [5,6,7], param_NtNrNx = (2,4,100), Nloop = 1000, pilot_SNRdB = None, alpha = 0):
	"""
	Ridge regression will be using to estimate channel.
	If alpha is zero, linear regression will be applied.
	If alpha is more than zero, Ridge regression will be applied.
	The default value of alpha is zero. 
	"""
	BER_pilot = list()

	Nt, Nr, Nx = param_NtNrNx	
	if alpha > 0:
		"""
		LinearRegression is using.
		"""
		for SNRdB in SNRdB_l:
			ber = MIMO().run_pilot_ch( pilot_SNRdB = pilot_SNRdB, 
				param_NtNrNxSNRdB =(Nt, Nr, Nx, SNRdB), Nloop = Nloop, alpha = alpha, disp = True)
			BER_pilot.append( ber)
	else:
		"""
		Ridge is using.
		"""
		for SNRdB in SNRdB_l:
			ber = MIMO().run_pilot_channel( pilot_SNRdB = pilot_SNRdB, 
				param_NtNrNxSNRdB =(Nt, Nr, Nx, SNRdB), Nloop = Nloop, disp = True)
			BER_pilot.append( ber)

	# print( "List of average BERs =", BER_pilot)

	return BER_pilot
