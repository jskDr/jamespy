import numpy as np
import matplotlib.pyplot as plt

xs = np.mat( np.arange( -10, 10+0.2, 0.2)).T
keps = 1e-9

m = lambda x: np.power( 0.25 * x, 2)
K = lambda p, q: np.exp( -0.5 * np.power(p.T.repeat( q.shape[0], axis=0) - q.repeat( p.T.shape[1], axis=1), 2))
ns = lambda x: np.shape( x)[0]
# ns = np.shape( xs)[0]
fs = lambda xs, keps: m(xs) + np.linalg.cholesky( K(xs,xs) + keps * np.mat(np.eye(ns(xs)))).T * np.mat(np.random.randn(ns(xs), 1))

plt.plot( xs, fs(xs, keps), '.')
plt.plot( xs, fs(xs, keps), '-')
plt.plot( xs, fs(xs, keps), '--')
plt.show()