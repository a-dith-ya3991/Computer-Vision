import cv2
import numpy as np
import matplotlib.pyplot as pyp
def rot(img,angle):
    x=angle
    si=np.sin(np.deg2rad(x))
    co=np.cos(np.deg2rad(x))
    m=np.array([[co,-si],
                [si,co]])
    sh,sw=img.shape[0],img.shape[1]#source iumage height , width
    corners=np.array([[0,0],[0,sw-1],[sh-1,0],[sh-1,sw-1]])
    newcord=m.dot(corners.T)#here we take transpose because to cnahge dimensions
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
img=cv2.imread(r"PATH")#replace "PATH" with image path
n=int(input("enter rotation degree:-"))
img2=rot(img,n)
fig,ax=pyp.subplots(1,2)
#ax[0].axis("off")
ax[0].imshow(img[:,:,::-1])
#ax[1].axis("off")
ax[1].imshow(img2[:,:,::-1])

