import cv2 
import numpy as np
import matplotlib.pyplot as ply
def scale(img,x,y):
    s=np.array([[x,0],
               [0,y]])
    h,w=img.shape[0],img.shape[1]
    img2=np.zeros((h*x,w*y,3),dtype="uint8")
    
    for i in range(h):
        for j in range(w):
            p=s.dot(np.array([i,j]))
            new_i,new_j=np.floor(p[0]),np.floor(p[1])
            img2[new_i,new_j]=img[i,j]
    return img2
img=cv2.imread(r"PATH")#PASTE IMAGE PATH
img2=scale(img,0.2,0.2)
ply.imshow(img2[:,:,::-1])
