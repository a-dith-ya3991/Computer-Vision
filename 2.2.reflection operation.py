""" For Reflection Operations use matrix 
[1,0]    for y-axis reflection,
[0,-1]

[-1,0] for x-axis reflection and 
[0,1]

[-1,0] for diagonal reflection
[0,-1]

"""
import cv2
import numpy as np
import matplotlib.pyplot as pyp
def rot(img,m):
    
    sh,sw=img.shape[0],img.shape[1]#source image height , width
    corners=np.array([[0,0],[0,sw-1],[sh-1,0],[sh-1,sw-1]])
    newcord=m.dot(corners.T)#here we take transpose because to change dimensions
    rmin,cmin=np.floor(min(newcord[0])),np.floor(min(newcord[1]))
    rmax,cmax=np.ceil(max(newcord[0])),np.ceil(max(newcord[1]))
    dh=np.int32(rmax-rmin+1)#destination image height,width
    dw=np.int32(cmax-cmin+1)
    img2=np.zeros((dh,dw,3),dtype="uint8")
    for i in range(sh):
        for j in range(sw):
            p_dash=m.dot(np.array([i,j]))
            new_i,new_j=np.int32(np.round(p_dash[0])-rmin),np.int32(np.round(p_dash[1])-cmin)
            img2[new_i][new_j]=img[i][j]
    return img2
img=cv2.imread(r"PATH")# Replace "PATH" with your image Path
m=np.array([[1,0],# Replace the matrix according to required form
           [0,-1]])
img2=rot(img,m)
fig,ax=pyp.subplots(1,2)
ax[0].imshow(img[:,:,::-1])
ax[1].imshow(img2[:,:,::-1])

