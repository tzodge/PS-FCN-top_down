# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
import skimage 
from skimage  import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import cv2

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    shifted_orig = [int(res[0]/2),int(res[1]/2)]
    image = np.zeros(res)
    i_coords, j_coords = np.meshgrid(range(res[0]), range(res[1]))
    pix_idx = np.hstack((i_coords.reshape(-1,1),j_coords.reshape(-1,1)))

    pix_coords = (pix_idx - shifted_orig)*pxSize  
    pix_coords[:,0] = -pix_coords[:,0]
    pix_coords = np.fliplr(pix_coords)
    # print(pix_coords,"pix_coords")
    circle_idx = np.where(np.sqrt(pix_coords[:,0]**2+pix_coords[:,1]**2) < rad)[0]
    # circle_idx = np.where(pix_coords[:,0]**2 >=0)[0]
    
    circle_pix_idx = pix_idx[circle_idx,:]
    circle_pix_coords = pix_coords[circle_idx,:]
    z_coord_sphere = np.sqrt(rad**2 - circle_pix_coords[:,0]**2 - circle_pix_coords[:,1]**2) 
    # print (z_coord_sphere,"z_coord_sphere")

    image[circle_pix_idx[:,0] , circle_pix_idx[:,1]] = z_coord_sphere
    surface_normals = np.hstack((circle_pix_coords, z_coord_sphere.reshape(-1,1)))
    # print(surface_normals,"surface_normals")

    intensity = surface_normals.dot(light)
    intensity= np.clip(intensity,a_min=0,a_max = 1e7)    
 
    image[circle_pix_idx[:,0] , circle_pix_idx[:,1]] = intensity.flatten()
    # image[circle_pix_idx[10:10000,0] , circle_pix_idx[10:10000,1]] = 100
    # print(circle_pix_idx[0,:],"circle_pix_idx[0,:]")
    # print(surface_normals[0,:],"surface_normals[0,:]")
    # image = None
    return image

def process_image(img):
    img_val = np.linalg.norm(img,axis=2) 
    ids = np.where(img_val<=0.2)
    img_out = np.copy(img)
    img_out[ids[0],ids[1],:] = 0
    return img_out

def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    # datasets/PS_Blobby_Dataset/Images/blob01_s-0.06_x-000_y-000/delrin
    # path = "../data/datasets/PS_Blobby_Dataset/Images/blob01_s-0.06_x-000_y-000/delrin"
    # path = "../data/datasets/DiLiGenT/pmsData/bearPNG"
    path = "../data/datasets/DiLiGenT/pmsData/pot1PNG"
    # path = "../data/datasets/DiLiGenT/pmsData/harvestPNG"
    # path = "../data/datasets/DiLiGenT/pmsData/gobletPNG"
    # path = "../data/datasets/DiLiGenT/pmsData/readingPNG"
    # path = "../data/datasets/DiLiGenT/pmsData/cowPNG"
    # path = "../data/datasets/DiLiGenT/pmsData/buddhaPNG"
    # path = "../data/datasets//PS_Sculpture_Dataset/Images/two-wrestlers-in-combat-repost_Two_wrestlersincombat_s-0.16_x-000_y-000_000/blue-metallic-paint"
# /PS_Sculpture_Dataset/Images/two-wrestlers-in-combat-repost_Two_wrestlersincombat_s-0.16_x-000_y-000_000/blue-metallic-paint

    # n=7
    I =[]
    num_images = 0
    for file in glob.glob(path+"/*.png"):
        # print(file)
        # img = io.imread(path+"input_{}.tif".format(i+1))
        img = io.imread(file)
        size_min = np.min(img.shape[0:2])
        img = cv2.resize(img, dsize=(size_min, size_min), interpolation=cv2.INTER_CUBIC)
        # print(np.max(img),"np.max(img)")
        # print (img.shape,"img.shape")
        # print( img.dtype," img.dtype")
        if img.shape[-1] == 3:

            img = img/np.max(img)
            if num_images==2:   
                plt.figure("input image")             
                plt.imshow(img)
                plt.show(block=False)

            img=process_image(img)*255
            num_images +=1


            img = skimage.color.rgb2xyz(img)[:,:,1]
            s = img.shape
            I.append(img.flatten())


    
    # for i in range(n):
    #     img = io.imread(path+"input_{}.tif".format(i+1))
    #     # print( img.dtype," img.dtype")
    #     img = skimage.color.rgb2xyz(img)[:,:,1]
    #     s = img.shape
    #     I.append(img.flatten())
        
    I = np.array(I) 
    # print(I.shape,"I.shape")
    # U,S,Vt = np.linalg.svd(I,full_matrices=False)
    # print(S,"S")
    # print(I.shape,"I.shape")
    # L = np.load(path+"sources.npy").T
    # print(L.shape,"L.shape")
    # s = None
    print (s,"s")
    return I,s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    U,S,Vt = np.linalg.svd(L.T)
    zero_padding = U.shape[0]-len(S)
    S_inv_diag = np.concatenate(( 1/S, np.zeros((zero_padding,)) ))
    S_inv_diag = np.diag(S_inv_diag)
    S_inv_diag = S_inv_diag[0:U.shape[0] ,0:Vt.shape[0] ]

    # S_inv_diag = np.concatenate(( 1/S, np.zeros((zero_padding,)) ))
    # print(S_inv_diag,"S_inv_diag")
    # print (U.shape,"U.shape")
    # print (S_inv_diag.shape,"S_inv_diag.shape")
    # print (Vt.shape,"Vt.shape")
    pseudo_inv = Vt.T.dot(S_inv_diag.T).dot(U.T)
    # print   (pseudo_inv.shape,"pseudo_inv.shape")
    # print   (I.shape,"I.shape")
    B = pseudo_inv.dot(I)
    # print(B.shape,"B.shape")
    # B = None
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    albedos = np.linalg.norm(B,axis=0)
    zero_idx = np.where(albedos ==0)[0]

    normals = B/albedos
    print (normals.shape,"normals.shape before")
    normals[:,zero_idx] = np.array([0,0,1]).reshape(-1,1)
    print (normals.shape,"normals.shape after")
    # albedos = None
    # normals = None
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    albedoIm = albedos.reshape(s)
    # plt.figure()
    plt.figure("albedo image")             

    cmap=plt.cm.gray
    plt.imshow(albedoIm,cmap=cmap)
    # plt.savefig("Q1f_albedo.jpg")
    plt.show(block=False)
 
    normalIm = np.zeros((s[0],s[1],3))
    normalIm[:,:,0] = normals[0,:].reshape(s)
    normalIm[:,:,1] = normals[1,:].reshape(s)
    normalIm[:,:,2] = normals[2,:].reshape(s)
    
    normalIm_disp = np.copy(normalIm)
    normalIm_disp = normalIm_disp - np.min(normalIm_disp)
    normalIm_disp = normalIm_disp/np.max(normalIm_disp)
    

    # print (normalIm,"normalIm")
    # print (np.max(normalIm),"np.max(normalIm)")
    # plt.figure()
    plt.figure("normals map")             
    cmap=plt.cm.rainbow
    plt.imshow(normalIm_disp,cmap=cmap)
    # plt.savefig("Q1f_normals.jpg")
    plt.show(block=False)
    # plt.show()

    # cmap=plt.cm.rainbow
    # plt.imshow(normalIm,cmap=cmap)
    # # plt.savefig("Q1f_normals_2.jpg")
    # plt.show(block=False)
    # plt.show()

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    f_x = (-normals[0,:]/normals[2,:]).reshape(s)
    f_y = (-normals[1,:]/normals[2,:]).reshape(s)
    z = integrateFrankot(f_x,f_y)
    # print(z)
    i_coords, j_coords = np.meshgrid(range(s[1]), range(s[0]))
    print (i_coords.shape,"i_coords.shape")
    print (j_coords.shape,"j_coords.shape")
    print (z.shape,"z.shape")
    print (f_y.shape,"f_y.shape")
    print (f_x.shape,"f_x.shape")
    surface = np.array([i_coords, j_coords, -z])
    # surface = None
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """

    fig = plt.figure("surface plot")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(surface[0], surface[1], surface[2], cmap='coolwarm')
    # plt.savefig("Q1i_surface.jpg")
    # ax.grid(False)
    # plt.axis('off')
    # ax.set_aspect('equal')
    plt.show()


    # pass


if __name__ == '__main__':

    # Put your main code here

    ## q1.b
    # center = np.zeros(( 3,))
    # rad = 0.75

    # # light = np.array([[1,1,1]])/np.sqrt(3)
    # # light = np.array([1,1,1])/np.sqrt(3)
    # light = np.array([-1,-1,1])/np.sqrt(3)
    # # light = np.array([1,1,1])/np.sqrt(3)
     
    # pxSize = 0.0007
    
    # res = np.array([2160,3840])
    # rendered_img = renderNDotLSphere(center, rad, light, pxSize, res)
    # im=plt.imshow(1-rendered_img,cmap='Greys')
    # plt.savefig("Q1b_-1-11.jpg")
    # plt.show()

    ### q1.d
    I,L,s = loadData()
 
    ### q1.e
    I,L,s = loadData()
    B = estimatePseudonormalsCalibrated(I,L)
    albedos, normals = estimateAlbedosNormals(B)
    displayAlbedosNormals(albedos, normals, s)
 
    surface = estimateShape(normals,s)
    plotSurface(surface)