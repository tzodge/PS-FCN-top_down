import numpy as np

mu = 0.1
nu = 0.2
lam = 0.3
G = np.array([[1,0,0],[0,1,0],[mu,nu,lam]])
G_inv_T = np.linalg.inv(G.T)
print (G,"G")
print (G_inv_T,"G_inv_T")
# print(G.dot(G_inv_T.T),"G.dot(G_inv_T)")