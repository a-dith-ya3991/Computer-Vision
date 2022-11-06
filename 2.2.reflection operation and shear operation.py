""" For Reflection Operations use matrix 
[1,0]    for y-axis reflection,
[0,-1]

[-1,0] for x-axis reflection and 
[0,1]

[-1,0] for diagonal reflection
[0,-1]

For shear operations use same code but change matrix as
[1,k] for shear operation among row wise
[0,1] 
          where k=factor of shear
[1,0]
[k,1] for shear operation among column wise

"""
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

def get_cords(img,m):
    corners=np.array([[0,0],[0,img.shape[1]-1],[img.shape[0]-1,0],[img.shape[0]-1,img.shape[1]-1]])
    corners_dash=m.dot(corners.T)

    (rmin,cmin)=np.floor([min(corners_dash[0]),min(corners_dash[1])])
    
    (rmax,cmax)=np.ceil([max(corners_dash[0]),max(corners_dash[1])])
    
    h,w=int(rmax+abs(rmin)),int(cmax+abs(cmin))
    print(int(rmin),int(cmin),int(rmax),int(cmax),h+1,w+1)
    return int(rmin),int(cmin),int(rmax),int(cmax),h+1,w+1

def transform(img,m):
    rmin,cmin,rmax,cmax,h,w=get_cords(img,m)
    m_inv=np.linalg.inv(m)
    img2=np.zeros((abs(h),abs(w),3),dtype="uint8")
    for newi in range(rmin,rmax):
        for newj in range(cmin,cmax):
            p=m_inv.dot(np.array([newi,newj]))
            i,j=p[0],p[1]
            if i<=-1 or i>=img.shape[0]-1 or j<=-1 or j>=img.shape[1]-1:
                pass
            else:
                img2[int(newi-rmin),int(newj-cmin)]=bi_linear(img,i,j)
    return img2

img =cv2.imread(r"Path")#change path 
m=np.array([[1,0],
            [0,1]])  #change matrix
img2=transform(img,m)
ply.imshow(img2[:,:,::-1])
