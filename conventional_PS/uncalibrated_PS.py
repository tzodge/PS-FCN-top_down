# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from calibrated_PS import loadData, estimateAlbedosNormals, displayAlbedosNormals
from calibrated_PS import estimateShape, plotSurface 
from utils import enforceIntegrability

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    # print (I)
    U,S,Vt = np.linalg.svd(I,full_matrices=False)
    k=3
    # print(U.shape,"U.shape")
    # print(S.shape,"S.shape")
    # print(Vt.shape,"Vt.shape")

    S_mat_sqrt = np.sqrt(np.diag(S[0:k]))
    L = (U[:,0:k].dot(S_mat_sqrt)).T
    B = S_mat_sqrt.dot(Vt[0:k,:])

    # print (U,"U")
    # print (np.max(U),np.min(U), "np.max(U),np.min(U)")
    # print (S,"S")
    # print (np.max(S),np.min(S), "np.max(S),np.min(S)")
    # print (Vt,"Vt")
    # print (np.max(Vt),np.min(Vt), "np.max(Vt),np.min(Vt)")
    # print(L.shape,"L.shape")
    # print(B.shape,"B.shape")

    # B = None
    # L = None

    return B, L

def bas_relief (B_hat,s,mu=0.,nu=0.,lam=1.):
    G = np.array([[1,0,0],[0,1,0],[mu,nu,lam]])
    G_inv_T = np.linalg.inv(G.T)
    # print (G,"G")
    print ("mu: ",mu, "nu: ",nu, "lam: ",lam)
    # print(G.dot(G_inv_T.T),"G.dot(G_inv_T)")
    B_hat_integrable = enforceIntegrability(B_hat,s)
    B_hat_integrable = G_inv_T.dot(B_hat_integrable)
     
    albedos, normals = estimateAlbedosNormals(B_hat_integrable)
    # displayAlbedosNormals(albedos, normals, s)

    surface_integrable = estimateShape(normals,s)
    plotSurface(surface_integrable)


if __name__ == "__main__":

    # Put your main code here
    I,s = loadData()

    ## 2b
    B_hat,L_hat = estimatePseudonormalsUncalibrated(I)

    albedos, normals = estimateAlbedosNormals(B_hat)
    print(albedos,"albedos")
    print(normals,"normals")
    # print(albedos,"albedos")
    # displayAlbedosNormals(albedos, normals, s)
 
    # 2c
    # print (L,"L")
    # print (L_hat,"L_hat")
    # print (np.linalg.norm(L_hat-L,axis=0),"np.linalg.norm(L_hat-L,axis=1)")

    # import matplotlib.pyplot as plt 
    # from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(L[0,:], L[1,:], L[2,:], color='r')
    # ax.scatter(L_hat[0,:], L_hat[1,:], L_hat[2,:], color='b')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

 
    ## 2d
    # surface = estimateShape(normals,s)
    # plotSurface(surface)

    # ## 2e
    B_hat_integrable = enforceIntegrability(B_hat,s)
    albedos, normals = estimateAlbedosNormals(B_hat_integrable)
    displayAlbedosNormals(albedos, normals, s)

    surface_integrable = estimateShape(normals,s)
    plotSurface(surface_integrable)    


    ### 2f
    # print("bas relief ")
    # Mu = [0,10,50]
    # Nu = [0,100,500]
    # Lam = [1, 0.01,5]

    # for mu in Mu:
    #     print ("mu= ", mu)
    #     bas_relief (B_hat,s,mu=mu)

    # for nu in Nu:
    #     print ("nu= ", nu)
    #     bas_relief (B_hat,s,nu=nu)

    # for lam in Lam:
    #     print ("lam= ", lam)
    #     bas_relief (B_hat,s,lam=lam)



'''
 The Bas-relief ambiguity arises not due to insufficient light source views. It's because of only one camera view. 
 SO I don't think just adding more images, would solve the problem

'''
