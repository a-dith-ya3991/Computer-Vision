import cv2 
import numpy as np
import matplotlib.pyplot as ply
def scale(img,x,y):
    s=np.array([[x,0],
               [0,y]])
    s_inv=np.linalg.inv(s)
    h,w=img.shape[0],img.shape[1]
    img2=np.zeros((int(h*x),int(w*y),3),dtype="uint8")
    
    for new_i in range(int(h*x)):
        for new_j in range(int(w*y)):
            p=s_inv.dot(np.array([new_i,new_j]))
            i,j=np.floor(p[0]),np.floor(p[1])
            img2[new_i,new_j]=img[int(i),int(j)]
    return img2
img=cv2.imread(r"PATH")#replace PATH with image path
img2=scale(img,2,2)
ply.imshow(img2[:,:,::-1])
