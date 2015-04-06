"some utility which I made."

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