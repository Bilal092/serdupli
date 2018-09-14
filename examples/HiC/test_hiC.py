import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

data_file = '/Users/antlaplante/THESE/SeriationDuplications/data/simul_cnv3_500000_noNA.matrix'
data_file = '/Users/antlaplante/THESE/SeriationDuplications/data/MCF10a_WT_250000.matrix'

aa = np.loadtxt(data_file, dtype='int')
# (iis, jjs, vvs) = np.loadtxt(data_file, dtype='int', unpack=True)
# mat = sp.coo_matrix((aa[:, 0, (iis, jjs)))
n = max(aa[:, 0].max(), aa[:, 1].max())
n += 1
mat = sp.coo_matrix((1 + np.log(aa[:, 2]), (aa[:, 0], aa[:, 1])), shape=(n, n))

fig = plt.figure()
plt.matshow(mat.toarray())
plt.show()

