import cv2 
import numpy as np
import matplotlib.pyplot as ply
def bi_linear(img,x,y):
    g=0
    if x>0 and x<img.shape[0]-1 and y>0 and y<img.shape[1]-1:
        right_weight=y-int(y)
        left_weight=1-right_weight
        a=img[int(x),int(y)]*left_weight+img[int(x),int(y)+1]*right_weight
        b=img[int(x)+1,int(y)]*left_weight+img[int(x)+1,int(y)+1]*right_weight
        bottom_weight=x-int(x)
        top_weight=1-bottom_weight
        g=a*top_weight+b*bottom_weight
    elif (x<0 or x>img.shape[0]-1)and(y>0 or y<img.shape[1]-1):
        right_weight=y-abs(y)
        left_weight=1-right_weight
        if x<0:
            g=img[0,int(y)]*left_weight+img[0,int(y)+1]*right_weight
        else :
            g=img[img.shape[0]-1,int(y)]*left_weight+img[img.shape[0]-1,int(y)+1]*right_weight
    elif (x>0 and x<img.shape[0]-1) and (y<0 or y>img.shape[1]-1):
        bottom_weight=x-int(x)
        top_weight=1-bottom_weight
        if y<0:
            g=img[int(x),0]*top_weight+img[int(x)+1,0]*bottom_weight
        else:
            g=img[int(x),img.shape[0]-1]*top_weight+img[int(x)+1,img.shape[0]-1]*bottom_weight
    else :
        g=img[int(x),int(y)]
    return np.uint8(g)

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
            img2[new_i,new_j]=bi_linear(img,int(i),int(j))
    return img2
img=cv2.imread(r"PATH")#replace PATH with image path
img2=scale(img,2,2)
ply.imshow(img2[:,:,::-1])
